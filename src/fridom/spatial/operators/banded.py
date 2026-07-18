"""
Banded (tridiagonal) solve primitives on a device-local axis.

Description
-----------
The grid-free kernel shared by the IMEX implicit vertical-diffusion
solve (``model/implicit.py``) and the mixed Fourier x Chebyshev
spectral banded solve (``operators/spectral_solve.py``, designed-for):
assemble a dense band, apply it along one storage axis (batched over
the off-axis columns), and solve ``system @ x = rhs`` along that axis.

Iteration-1 scope: the band is materialized dense and solved with
``jnp.linalg.solve`` (batched over the off-axis columns); the solve
axis must stay device-local (a tridiagonal is serial along it). A
Thomas / ``jax.lax.linalg.tridiagonal_solve`` kernel is the production
optimization; both respect the same true-shape ``data`` /
``with_data`` halo contract of the caller.

The multigrid vertical-line smoother (``operators/multigrid.py``,
decision MG-D7) needs a *diagonal-varying* tridiagonal per off-axis
column, so it does not materialize the dense band:
:func:`tridiagonal_solve_along_axis` solves one distinct tridiagonal
per column, batched over the off-axis columns, through one of three
interchangeable kernels chosen by its ``method`` argument. All three
compute the same ``T^{-1}`` to machine precision (agreement ~1e-18, the
multigrid kernel study) and are natively reverse-mode differentiable,
no ``custom_vjp``:

- ``"scan"`` — the reference batched Thomas algorithm: a forward-
  elimination and a back-substitution :func:`jax.lax.scan`. Portable
  and exact, but its ``2 * N`` sequential scan-loop launches make it
  latency-bound on the GPU (~2.75 ms/solve, batch-independent).
- ``"pcr"`` — pure-jax parallel cyclic reduction: ``ceil(log2 N)``
  data-parallel passes, portable to any backend, ~7x the ``scan`` on
  the GPU. The portable default off the GPU.
- ``"cusparse"`` — the batched
  :func:`jax.lax.linalg.tridiagonal_solve` (cuSPARSE
  ``gtsv2StridedBatch``), one distinct system per column; the fastest
  kernel, but it requires a CUDA GPU jax backend.

``method="auto"`` resolves — host-side, never a traced branch — to
``"cusparse"`` on a GPU backend and ``"pcr"`` elsewhere. No kernel
pivots: the caller guarantees a non-singular column (the line smoother
substitutes ``diag -> 1`` on dry/zero-diagonal cells). The older
warning that ``jax.lax.linalg.tridiagonal_solve`` must be avoided for
uneven autodiff support is refuted by the kernel study — it
differentiates with respect to the rhs and the diagonal in jax 0.10.2.
"""
# Wave 9B: lifted out of model/implicit.py (plan section 5, decision C)
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from fridom.framework.utils import dtype_real

# ================================================================
#  Band assembly
# ================================================================
#: the per-side corner main-diagonal value each boundary condition
#: contributes to :func:`second_difference_matrix`. Neumann (zero-flux,
#: even ghost mirror ``u_{-1} = u_0``) gives ``-1``; Dirichlet (no-slip,
#: odd ghost mirror across the half-cell ``u_{-1} = -u_0``) gives ``-3``
#: (the wall-adjacent row becomes ``(u_1 - 3 u_0) / dz^2``).
_CORNER: dict[str, float] = {"neumann": -1.0, "dirichlet": -3.0}


def validate_boundary_conditions(
    bc: tuple[str, str],
) -> tuple[str, str]:
    """
    Return the ``(low, high)`` boundary-condition pair or raise.

    Description
    -----------
    The single source of truth for the per-side boundary vocabulary of
    :func:`second_difference_matrix` (and the ``VerticalDiffusion``
    operator that carries one): each side must be ``"neumann"`` (the
    zero-flux corner) or ``"dirichlet"`` (the no-slip odd-mirror
    corner). The choice is static (host-side), so it is validated in
    plain Python — never on a traced value.

    Parameters
    ----------
    bc : tuple[str, str]
        The per-side ``(low, high)`` boundary conditions.

    Returns
    -------
    tuple[str, str]
        The validated ``(low, high)`` pair.

    Raises
    ------
    ValueError
        If `bc` is not a length-2 pair of accepted condition names.
    """
    pair = tuple(bc)
    valid = tuple(_CORNER)
    if len(pair) != 2 or any(  # noqa: PLR2004 — a (low, high) pair
            side not in _CORNER for side in pair):
        raise ValueError(
            "boundary conditions must be a (low, high) pair drawn from "
            f"{valid}, got {bc!r}")
    return pair


def second_difference_matrix(
    coords: jax.Array,
    bc: tuple[str, str] = ("neumann", "neumann"),
) -> jax.Array:
    r"""
    Dense ``(N, N)`` second-difference ``d2/dz2`` on uniform nodes.

    Description
    -----------
    The tridiagonal second-difference stencil divided by ``dz^2``
    (uniform spacing inferred from the first two nodes) with per-side
    boundary rows selected by `bc`. A **Neumann** (zero-flux) side puts
    ``-1`` on its corner diagonal (the even ghost mirror ``u_{-1} =
    u_0``); a **Dirichlet** (no-slip) side puts ``-3`` (the odd ghost
    mirror across the half-cell ``u_{-1} = -u_0``, so the wall-adjacent
    row is ``(u_1 - 3 u_0) / dz^2``). The default ``("neumann",
    "neumann")`` reproduces the historical zero-flux band. Grid-free: it
    reads only the 1D node line, so the same band feeds the diffusion
    operator (scaled by ``kappa``) and, with a per-mode scalar shift,
    the spectral banded z-solve; the ``bc`` choice is static (host-side)
    and jit-friendly (it selects Python constants, never a traced
    branch).

    Parameters
    ----------
    coords : jax.Array
        The evaluation nodes along the solve axis (any shape; flattened
        to the 1D line).
    bc : tuple[str, str], optional
        The per-side ``(low, high)`` boundary conditions, each
        ``"neumann"`` or ``"dirichlet"`` (default: ``("neumann",
        "neumann")``).

    Returns
    -------
    jax.Array
        The ``(N, N)`` second-difference matrix.

    Raises
    ------
    ValueError
        If `bc` is not a length-2 pair of accepted condition names.
    """
    low, high = validate_boundary_conditions(bc)
    real = dtype_real()
    line = jnp.reshape(jnp.asarray(coords), (-1,)).astype(real)
    size = line.shape[0]
    dz = line[1] - line[0]
    main = jnp.full((size,), -2.0, dtype=real)
    main = main.at[0].set(_CORNER[low]).at[size - 1].set(_CORNER[high])
    off = jnp.ones((size - 1,), dtype=real)
    return (jnp.diag(main) + jnp.diag(off, 1) + jnp.diag(off, -1)
            ) / (dz * dz)


# ================================================================
#  Batched apply / solve along one storage axis
# ================================================================
def apply_along_axis(
    operator: jax.Array, data: jax.Array, axis_index: int,
) -> jax.Array:
    """
    Apply the ``(N, N)`` band along ``axis_index`` (batched columns).

    Parameters
    ----------
    operator : jax.Array
        The ``(N, N)`` band matrix.
    data : jax.Array
        The operand array; its ``axis_index`` axis has length ``N``.
    axis_index : int
        The storage-frame index of the solve axis.

    Returns
    -------
    jax.Array
        ``operator @ data`` along ``axis_index`` (same shape as ``data``).
    """
    moved = jnp.moveaxis(data, axis_index, -1)
    shape = moved.shape
    flat = moved.reshape(-1, shape[-1])
    out = flat @ operator.T
    return jnp.moveaxis(out.reshape(shape), -1, axis_index)


def solve_along_axis(
    system: jax.Array, data: jax.Array, axis_index: int,
) -> jax.Array:
    """
    Solve ``system @ x = data`` along ``axis_index`` (batched columns).

    Parameters
    ----------
    system : jax.Array
        The ``(N, N)`` band system to invert.
    data : jax.Array
        The right-hand side; its ``axis_index`` axis has length ``N``.
    axis_index : int
        The storage-frame index of the solve axis.

    Returns
    -------
    jax.Array
        The solution ``x`` (same shape as ``data``).
    """
    moved = jnp.moveaxis(data, axis_index, -1)
    shape = moved.shape
    flat = moved.reshape(-1, shape[-1])
    solved = jnp.linalg.solve(system, flat.T).T
    return jnp.moveaxis(solved.reshape(shape), -1, axis_index)


# ================================================================
#  Per-column tridiagonal solve (distinct tridiagonal per column)
# ================================================================
#: the interchangeable per-column tridiagonal kernels; ``"auto"``
#: resolves against the jax backend (host-static, never traced)
_TRIDIAGONAL_METHODS: tuple[str, ...] = (
    "auto", "cusparse", "pcr", "scan")


def validate_tridiagonal_method(method: str) -> str:
    """
    Return `method` if it names a known kernel, else raise.

    Description
    -----------
    The name-only validation of the :func:`tridiagonal_solve_along_axis`
    ``method`` argument — no backend query, so a solver can validate the
    knob eagerly at construction (a typo fails at assembly, not at first
    trace) without prematurely rejecting ``"cusparse"`` on a host that
    happens to build the model on the CPU. The backend resolution lives
    in :func:`_resolve_tridiagonal_method`, called at solve time.

    Parameters
    ----------
    method : str
        The requested kernel name.

    Returns
    -------
    str
        The validated `method`.

    Raises
    ------
    ValueError
        If `method` is not one of the accepted kernel names.
    """
    if method not in _TRIDIAGONAL_METHODS:
        raise ValueError(
            "tridiagonal method must be one of "
            f"{_TRIDIAGONAL_METHODS}, got {method!r}")
    return method


def _resolve_tridiagonal_method(method: str) -> str:
    """
    Resolve `method` to a concrete kernel against the jax backend.

    Description
    -----------
    Validates the name, then maps ``"auto"`` to ``"cusparse"`` on a GPU
    backend and ``"pcr"`` elsewhere, and rejects an explicit
    ``"cusparse"`` request when the default backend is not a CUDA GPU.
    Host-side (the backend query is static), so the kernel choice is a
    Python constant, never a traced branch.

    Parameters
    ----------
    method : str
        The requested kernel name (``"auto"`` / ``"cusparse"`` /
        ``"pcr"`` / ``"scan"``).

    Returns
    -------
    str
        A concrete kernel name (``"cusparse"`` / ``"pcr"`` / ``"scan"``).

    Raises
    ------
    ValueError
        If `method` is unknown, or ``"cusparse"`` is requested without a
        CUDA GPU jax backend.
    """
    validate_tridiagonal_method(method)
    backend = jax.default_backend()
    if method == "auto":
        return "cusparse" if backend == "gpu" else "pcr"
    if method == "cusparse" and backend != "gpu":
        raise ValueError(
            "the cusparse tridiagonal kernel requires a CUDA GPU jax "
            f"backend, but the current backend is {backend!r}; use "
            "method='pcr' (pure-jax, portable to any backend) or "
            "method='auto' (cusparse on a GPU, pcr elsewhere)")
    return method


def _shift_down(arr: jax.Array, span: int, fill: float) -> jax.Array:
    """Bring row ``i - span`` to row ``i`` (fill the top ``span`` rows)."""
    size = arr.shape[0]
    return jnp.pad(
        arr, ((span, 0), (0, 0)), constant_values=fill)[:size]


def _shift_up(arr: jax.Array, span: int, fill: float) -> jax.Array:
    """Bring row ``i + span`` to row ``i`` (fill the last ``span`` rows)."""
    return jnp.pad(
        arr, ((0, span), (0, 0)), constant_values=fill)[span:]


def _tridiagonal_scan(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    """
    Solve per column with the reference batched Thomas scan.

    Description
    -----------
    The reference kernel on the ``(n, batch)`` layout:
    one forward-elimination :func:`jax.lax.scan` (the modified
    super-diagonal and right-hand side) and one reverse
    back-substitution ``scan``. Kept verbatim as the reference kernel
    (exact reproducibility of the prior scan-Thomas results); the ends
    ``lower[0]`` / ``upper[-1]`` are no-ops against the zero carries.
    """
    batch = rhs.shape[1]
    zero = jnp.zeros((batch,), dtype=rhs.dtype)

    def eliminate(
        carry: tuple[jax.Array, jax.Array],
        row: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]
    ]:
        c_prev, d_prev = carry
        low, dia, upp, right = row
        denom = dia - low * c_prev
        c_new = upp / denom
        d_new = (right - low * d_prev) / denom
        return (c_new, d_new), (c_new, d_new)

    _, (c_star, d_star) = lax.scan(
        eliminate, (zero, zero), (lower, diag, upper, rhs))

    def substitute(
        x_next: jax.Array, row: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        c_new, d_new = row
        x_cur = d_new - c_new * x_next
        return x_cur, x_cur

    _, solved = lax.scan(
        substitute, zero, (c_star, d_star), reverse=True)
    return solved


def _tridiagonal_pcr(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    r"""
    Parallel cyclic reduction on the ``(n, batch)`` layout.

    Description
    -----------
    Log-depth (``ceil(log2 n)`` passes, host-unrolled at trace time)
    cyclic reduction for arbitrary ``n`` — not only powers of two. Each
    pass eliminates the ``i - span`` and ``i + span`` neighbours of
    every row against the shifted equations; the out-of-range ends are
    filled ``diag -> 1``, ``lower/upper/rhs -> 0`` so those eliminations
    are exact no-ops and every division stays non-singular (no
    ``jnp.where`` guard needed, so the reverse pass is NaN-free). The
    ends ``lower[0]`` / ``upper[-1]`` are zeroed up front, which makes
    the row-0 / row-``(n - 1)`` eliminations no-ops too. Pure jax, so it
    is portable to any backend and partitions cleanly along the batch.
    """
    a = lower.at[0].set(0.0)
    b = diag
    c = upper.at[-1].set(0.0)
    d = rhs
    size = a.shape[0]
    span = 1
    while span < size:
        a_m = _shift_down(a, span, 0.0)
        b_m = _shift_down(b, span, 1.0)
        c_m = _shift_down(c, span, 0.0)
        d_m = _shift_down(d, span, 0.0)
        a_p = _shift_up(a, span, 0.0)
        b_p = _shift_up(b, span, 1.0)
        c_p = _shift_up(c, span, 0.0)
        d_p = _shift_up(d, span, 0.0)
        alpha = -a / b_m
        gamma = -c / b_p
        a = alpha * a_m
        c = gamma * c_p
        b = b + alpha * c_m + gamma * a_p
        d = d + alpha * d_m + gamma * d_p
        span *= 2
    return d / b


def _tridiagonal_cusparse(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    r"""
    Batched cuSPARSE ``gtsv2StridedBatch`` on the ``(n, batch)`` layout.

    Description
    -----------
    The batched :func:`jax.lax.linalg.tridiagonal_solve` — one distinct
    system per column, the batch as the leading dim (**never** the
    single-system form with the batch folded into the RHS width, a
    pathological ``O(nrhs)`` trap). The bands transpose to ``(batch,
    n)`` and the rhs to ``(batch, n, 1)``; the API requires ``dl[.., 0]
    == du[.., -1] == 0``, so both are zeroed explicitly (our contract
    lets the caller leave those ends arbitrary). Requires a CUDA GPU
    backend; caught host-side by :func:`_resolve_tridiagonal_method`.
    """
    dl = lower.T.at[:, 0].set(0.0)
    dd = diag.T
    du = upper.T.at[:, -1].set(0.0)
    b = rhs.T[..., None]
    solved = lax.linalg.tridiagonal_solve(dl, dd, du, b)
    return solved[..., 0].T


def tridiagonal_solve_along_axis(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    data: jax.Array,
    axis_index: int,
    method: str = "auto",
) -> jax.Array:
    r"""
    Solve one tridiagonal per column along ``axis_index``.

    Description
    -----------
    A distinct tridiagonal ``T`` (the per-cell ``lower``/``diag``/
    ``upper`` bands) is solved against ``data`` along ``axis_index``,
    batched over every off-axis column. Unlike :func:`solve_along_axis`
    no dense band is materialized — the multigrid vertical-line
    smoother's ``T`` varies from column to column (MG-D7). Three
    interchangeable kernels compute the same ``T^{-1}`` to machine
    precision (agreement ~1e-18, the multigrid kernel study) and are all
    natively reverse-mode differentiable (no ``custom_vjp``), selected
    by `method`:

    - ``"scan"`` — the reference batched Thomas algorithm, one
      forward-elimination and one back-substitution
      :func:`jax.lax.scan`. Portable and exact, but its ``2 * N``
      sequential scan launches make it latency-bound on the GPU (kept
      verbatim as the reference kernel).
    - ``"pcr"`` — pure-jax parallel cyclic reduction: ``ceil(log2 N)``
      data-parallel passes, portable to any backend (the portable
      default off the GPU).
    - ``"cusparse"`` — the batched
      :func:`jax.lax.linalg.tridiagonal_solve` (cuSPARSE
      ``gtsv2StridedBatch``), one distinct system per column; the
      fastest kernel, but it requires a CUDA GPU jax backend.

    ``method="auto"`` (the default) resolves — host-side, never a traced
    branch — to ``"cusparse"`` on a GPU backend and ``"pcr"`` elsewhere.

    The band arrays broadcast against ``data``: ``lower[i]`` is the
    sub-diagonal coupling of cell ``i`` to ``i - 1`` and ``upper[i]``
    the super-diagonal coupling to ``i + 1``, both along ``axis_index``.
    The ends ``lower[0]`` and ``upper[N - 1]`` are unused — the ``scan``
    seeds them against a zero carry, ``pcr`` / ``cusparse`` zero them
    explicitly — so the caller may leave them at any value.

    Precondition: **no pivoting** is performed, so every column's
    tridiagonal must be non-singular (diagonally dominant is
    sufficient; the reduced systems of a DD system stay DD). A dry /
    zero-diagonal column must be sanitized by the caller — the
    vertical-line smoother substitutes ``diag -> 1`` and ``rhs -> 0``
    there through the double-``jnp.where`` guard, which also keeps the
    reverse pass NaN-free.

    Multi-device note: on a sharded (multi-device) run the
    ``"cusparse"`` path lowers to a custom call, and XLA partitions it
    cleanly along the sharded batch axes — validated 2026-07-18 on
    4x A100 (jax 0.10.2): the custom call receives per-shard operands at
    every multigrid level with no feeding collective (no all-gather),
    both in a minimal standalone jit and in the in-model
    ``jit__chunk_body`` step (record:
    ``design/research/multigrid_kernel_study.md`` Addendum 2). This is an
    observed XLA lowering behaviour, not an API contract; the pure-jax
    ``"pcr"`` / ``"scan"`` kernels partition cleanly by construction, so
    ``method="pcr"`` stays the portable choice for a run that ever sees
    an unexpected all-gather feed the batch axes.

    Parameters
    ----------
    lower : jax.Array
        The sub-diagonal band (broadcasts against ``data``; ``lower[0]``
        along ``axis_index`` unused).
    diag : jax.Array
        The main diagonal band (broadcasts against ``data``; non-zero on
        every solved cell).
    upper : jax.Array
        The super-diagonal band (broadcasts against ``data``;
        ``upper[N - 1]`` along ``axis_index`` unused).
    data : jax.Array
        The right-hand side; its ``axis_index`` axis has length ``N``.
    axis_index : int
        The storage-frame index of the solve axis.
    method : str, optional
        The kernel: ``"auto"``, ``"cusparse"``, ``"pcr"`` or ``"scan"``
        (default: ``"auto"``).

    Returns
    -------
    jax.Array
        The per-column solution ``x`` (same shape as ``data``).

    Raises
    ------
    ValueError
        If `method` is not a known kernel name, or ``method="cusparse"``
        is requested without a CUDA GPU jax backend.
    """
    resolved = _resolve_tridiagonal_method(method)
    lo = jnp.moveaxis(lower, axis_index, 0)
    di = jnp.moveaxis(diag, axis_index, 0)
    up = jnp.moveaxis(upper, axis_index, 0)
    rhs = jnp.moveaxis(data, axis_index, 0)
    shape = rhs.shape
    size = shape[0]
    lo = jnp.broadcast_to(lo, shape).reshape(size, -1)
    di = jnp.broadcast_to(di, shape).reshape(size, -1)
    up = jnp.broadcast_to(up, shape).reshape(size, -1)
    rhs = rhs.reshape(size, -1)
    if resolved == "scan":
        solved = _tridiagonal_scan(lo, di, up, rhs)
    elif resolved == "pcr":
        solved = _tridiagonal_pcr(lo, di, up, rhs)
    else:  # "cusparse"
        solved = _tridiagonal_cusparse(lo, di, up, rhs)
    solved = solved.reshape(shape)
    return jnp.moveaxis(solved, 0, axis_index)
