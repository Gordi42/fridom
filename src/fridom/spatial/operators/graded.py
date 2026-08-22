"""
Shared graded near-wall closure machinery (interior DOFs only).

Description
-----------
Owning plan: ``design/plans/active/boundary_plan.md`` (rule R1). The
graded closure is the explicit opt-in that makes a *wide* stencil legal
on a bounded axis: in the interior the operator runs its full-width
kernel; at the ``K`` output faces adjacent to each physical wall it
substitutes progressively narrower stencils that read **no exterior
value at all**. This module owns the pieces every graded operator
shares — the ladder arithmetic, the wall-window builder, and the
multi-device ``patch_physical_ends`` seam — so the FV
(:mod:`~fridom.spatial.operators.fallback`) and the nodal C-grid
(``model.modules.advection``) families use one implementation.

The cell frame
--------------
Every graded operator here maps a **cell** family onto the **interior
faces** of the same 1D lattice:

- ``n_cells`` lattice cells, indexed ``c = 0 .. n_cells - 1``;
- ``n_faces = n_cells - 1`` interior faces, indexed ``F = 1 .. n_faces``
  (face ``F`` sits between cells ``F - 1`` and ``F``), which is exactly
  the operator's output DOF set (output true index ``k = F - 1``).

Two staggerings realize this frame, distinguished by the integer
``shift`` (the number of lattice cells that live *outside* the operand's
true DOF region, per side):

- ``shift = 0`` — the cells **are** the operand's true DOFs
  (``CellAvg -> Inner`` for the FV family, ``Center -> Inner`` for the
  nodal C-grid): ``n_cells = t_in``, cell ``c`` is operand true DOF
  ``c``. The bounded space is BC-free, so nothing outside ``[0,
  n_cells - 1]`` exists (R1) and the closure is interior-only by
  construction.
- ``shift = 1`` — the *dual* direction (``Inner -> Center``: a
  wall-normal velocity reconstructed onto the cell centers). The
  lattice cells are the ``n + 1`` faces of the mesh; the ``n - 1``
  interior faces are the operand's true DOFs and the two **wall** cells
  (``c = 0`` and ``c = n_cells - 1``) are the homogeneous Dirichlet
  boundary values — exact zeros, a boundary condition rather than a
  DOF. ``n_cells = t_in + 2``, cell ``c`` is operand true DOF
  ``c - 1``. The wall cells are **synthesized** (``jnp.zeros_like``)
  rather than read from the Dirichlet ghost slot, so a graded operator
  reads no ghost storage on a bounded axis whatever its BC tag.

In both cases the operand storage index of cell ``c`` is
``width + c - shift`` and the output storage slot of face ``F`` is
``width_out + (F - 1)``; and ``t_out = n_cells - 1``, so the two wall
blocks can index their local cells off the **local** ``t_out`` alone
(the decomposition's last shard absorbs the staggered true-count
deficit, so ``t_out - t_in = 2 * shift - 1`` holds on the right-wall
block exactly as it does globally).

The ladder
----------
A rung is a (``size``, ``offset``, ``kernel``) triple: at face ``F`` it
reads the ``size`` lattice cells ``F - 1 - offset .. F - 2 - offset +
size`` and its ``kernel`` collapses that window to the single face
value. ``K = len(rungs)`` reduced faces per side, widest rung first and
the wall-adjacent rung last, exactly as in ``fallback.py``.

``K`` is the count of output faces per side whose *full-width* window
leaves the operand's true-DOF cell range ``[shift, n_cells - 1 -
shift]`` (maximized over the left/right bias, so one ``K`` serves both
members of an upwind pair):

- biased odd-order kernels: ``K = order // 2 + shift``, and the rung at
  distance ``d`` from the wall (``d = 1`` wall-adjacent) has order
  ``min(order, 2 * d - 1)``;
- centered even-size kernels: ``K = size // 2 - 1 + shift``, and the
  rung at distance ``d`` has size ``min(size, 2 * d)``.

At ``shift = 1`` the outermost rung keeps the interior width (its window
merely reaches the exact wall value); at ``shift = 0`` it genuinely
narrows, because a BC-free bounded space offers nothing outside its true
DOFs. Both ladders bottom out at the wall-adjacent face: the 1st-order
upwind cell (biased) or the two-point mean (centered). Every index above
is a compile-time constant, so the interior-vs-wall split is a **static**
index partition, never a data-dependent ``jnp.where``.

The bottom rung of a biased ladder (``wall=``)
----------------------------------------------
The wall-adjacent rung of a *biased* ladder is the one place where the
closure has a genuine choice, and it is the user's
(:func:`biased_specs`):

- ``wall="upwind1"`` (default) — the 1st-order upwind cell. Monotone
  and dissipative exactly where a boundary layer or a front may sit,
  but its face value carries an :math:`O(h)` error, which the FD-flux
  difference turns into an :math:`O(h)` tendency error at the wall
  cell (the wall-normal velocity vanishes linearly, which is what
  saves it from :math:`O(1)`). The global rate on a walled axis is
  therefore ~1, whatever the interior order.
- ``wall="centered2"`` — the two-point mean of the two cells that
  straddle the wall-adjacent face. It reads exactly the same cells the
  upwind rung's window is a subset of, so it is just as interior-only
  (R1), and its :math:`O(h^2)` face value lifts the global rate to ~2.
  The price is that the wall-adjacent face carries **no upwind bias**
  and hence no numerical dissipation: both members of an upwind pair
  return the same value there, so a front sitting on the wall is
  reconstructed by a centered stencil and may ring.

Both rungs are exact on constants (free-stream preservation) and
neither changes the impermeability argument: a graded operator only
ever writes the *interior* faces, and the wall flux is a structural
zero of the flux space, not something a rung computes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Sequence

    from jax import Array

    from fridom.spatial.operators.base import FieldLike, Operator


class Rung(NamedTuple):

    """
    One rung of a graded ladder (a narrow wall-side stencil).

    Description
    -----------
    At face ``F`` the rung reads the ``size`` lattice cells starting at
    ``F - 1 - offset`` (the cell frame of the module docstring) and
    ``kernel`` collapses that window — whose ``axis`` length is exactly
    ``size`` — to the single face value (``axis`` length 1).

    A kernel that needs **geometry** (the non-uniform reconstruction
    rows of a stretched factor read the lattice-cell widths) takes it
    as trailing ``co_windows``: one window per ``co_storages`` entry of
    :func:`apply_graded_walls`, sliced with the very same window as the
    operand but **without** the wall-value synthesis, since the wall
    slots of a width array carry the real (half-cell) widths.

    Parameters
    ----------
    size : int
        The rung's window width in lattice cells.
    offset : int
        The window's start cell relative to the face: the first cell is
        ``F - 1 - offset``.
    kernel : Callable[..., Array]
        The array kernel ``(window, axis_index, *co_windows) ->
        face_value``.
    """

    size: int
    offset: int
    kernel: Callable[..., Array]


# ================================================================
#  Ladder arithmetic (static; see the module docstring)
# ================================================================
def biased_offset(order: int, bias: Literal["left", "right"]) -> int:
    """
    Window start offset of a biased odd-order rung (cell frame).

    Description
    -----------
    The left-biased window of order ``order`` at face ``F`` is centered
    one cell to the left of the face (``offset = order // 2``), the
    right-biased one one cell to the right (``offset = order // 2 - 1``)
    — the bias that breaks the tie of an odd stencil between staggered
    offsets. Order 1 degenerates to the single upwind cell: ``F - 1``
    (left) or ``F`` (right).

    Parameters
    ----------
    order : int
        The rung's odd formal order.
    bias : Literal["left", "right"]
        The upwind bias side.

    Returns
    -------
    int
        The window start offset.
    """
    return order // 2 if bias == "left" else order // 2 - 1


def centered_offset(size: int) -> int:
    """
    Window start offset of a centered even-size rung (cell frame).

    Parameters
    ----------
    size : int
        The rung's even window width.

    Returns
    -------
    int
        The window start offset (``size // 2 - 1``): the window
        ``F - size // 2 .. F + size // 2 - 1`` straddles the face.
    """
    return size // 2 - 1


def biased_rows(order: int, shift: int) -> int:
    """
    Reduced faces per side ``K`` of a biased odd-order kernel.

    Parameters
    ----------
    order : int
        The interior odd formal order.
    shift : int
        The cell-frame shift (0 or 1; module docstring).

    Returns
    -------
    int
        ``K = order // 2 + shift``.
    """
    return order // 2 + shift


def centered_rows(size: int, shift: int) -> int:
    """
    Reduced faces per side ``K`` of a centered even-size kernel.

    Parameters
    ----------
    size : int
        The interior even window width.
    shift : int
        The cell-frame shift (0 or 1; module docstring).

    Returns
    -------
    int
        ``K = size // 2 - 1 + shift`` (zero for the two-point mean on a
        BC-free operand: its window is already interior-only).
    """
    return size // 2 - 1 + shift


def biased_ladder(order: int, shift: int) -> tuple[int, ...]:
    """
    Rung orders of a biased graded ladder, widest first.

    Description
    -----------
    The rung at distance ``d`` from the wall carries order
    ``min(order, 2 * d - 1)``: the widest odd window whose cells stay
    inside ``[0, n_cells - 1]`` at face ``d``. Returned wall-adjacent
    **last** (the ``Fallback`` convention), so the tuple ends in the
    1st-order upwind rung.

    Parameters
    ----------
    order : int
        The interior odd formal order.
    shift : int
        The cell-frame shift (0 or 1).

    Returns
    -------
    tuple[int, ...]
        The ``K`` rung orders, widest first.
    """
    k = biased_rows(order, shift)
    return tuple(min(order, 2 * d - 1) for d in range(k, 0, -1))


def centered_ladder(size: int, shift: int) -> tuple[int, ...]:
    """
    Rung sizes of a centered graded ladder, widest first.

    Description
    -----------
    The rung at distance ``d`` from the wall carries width
    ``min(size, 2 * d)``. Returned wall-adjacent **last**, so the tuple
    ends in the two-point mean.

    Parameters
    ----------
    size : int
        The interior even window width.
    shift : int
        The cell-frame shift (0 or 1).

    Returns
    -------
    tuple[int, ...]
        The ``K`` rung sizes, widest first (empty when ``K == 0``).
    """
    k = centered_rows(size, shift)
    return tuple(min(size, 2 * d) for d in range(k, 0, -1))


#: the bottom-rung options of a biased graded ladder (``wall=``):
#: the 1st-order upwind cell (monotone, globally 1st order) or the
#: two-point centered mean (globally 2nd order, no upwind dissipation
#: on the wall-adjacent face) — see the module docstring
WALL_RUNGS = ("upwind1", "centered2")


class RungSpec(NamedTuple):

    """
    One rung of a graded ladder, before its kernel is built.

    Description
    -----------
    The kernel-free half of a `Rung`: which *family* of stencil the
    rung is (an upwind-biased odd-order row, or a symmetric even-size
    one) and how wide its window is. The holding operator turns a spec
    into a `Rung` by attaching its own array kernel (the biased rows
    are WENO- or linear-weighted, which ``graded`` does not know).

    Parameters
    ----------
    family : Literal["biased", "centered"]
        The stencil family: "biased" (odd ``width`` = its formal
        order) or "centered" (even ``width``).
    width : int
        The rung's window width in lattice cells.
    """

    family: Literal["biased", "centered"]
    width: int


def biased_specs(
    order: int,
    shift: int,
    wall: Literal["upwind1", "centered2"] = "upwind1",
) -> tuple[RungSpec, ...]:
    """
    Rung specs of a biased graded ladder, widest first.

    Description
    -----------
    `biased_ladder` with its bottom (wall-adjacent) rung made
    explicit: the ladder's reduced orders ``min(order, 2 * d - 1)``,
    and at the wall-adjacent face either the 1st-order upwind cell
    (``wall="upwind1"``, the default and the historical behavior) or
    the two-point centered mean (``wall="centered2"``). Both windows
    live at distance 1 from the wall and stay inside the lattice, so
    the ladder is interior-only either way; they differ in accuracy
    and in monotonicity (module docstring).

    Parameters
    ----------
    order : int
        The interior odd formal order.
    shift : int
        The cell-frame shift (0 or 1).
    wall : Literal["upwind1", "centered2"], optional
        The bottom (wall-adjacent) rung (default: "upwind1").

    Returns
    -------
    tuple[RungSpec, ...]
        The ``K`` rung specs, widest first and wall-adjacent last.
    """
    if wall not in WALL_RUNGS:
        raise ValueError(
            f"wall must be one of {WALL_RUNGS}, got {wall!r}")
    specs = [RungSpec("biased", rung)
             for rung in biased_ladder(order, shift)]
    if specs and wall == "centered2":
        specs[-1] = RungSpec("centered", 2)
    return tuple(specs)


def spec_offset(
    spec: RungSpec, bias: Literal["left", "right"],
) -> int:
    """
    Window start offset of a rung spec (cell frame).

    Parameters
    ----------
    spec : RungSpec
        The rung spec.
    bias : Literal["left", "right"]
        The upwind bias side (ignored by a centered rung).

    Returns
    -------
    int
        `biased_offset` of an odd biased rung, `centered_offset` of a
        symmetric one.
    """
    if spec.family == "centered":
        return centered_offset(spec.width)
    return biased_offset(spec.width, bias)


def n_interior_faces(n_cells: int, shift: int) -> int:
    """
    Output faces of a graded ladder on a walled axis of ``n_cells``.

    Description
    -----------
    The ladder's output DOF count in the cell frame: ``n_cells - 1``
    interior faces on the primal staggering (``shift = 0``) and
    ``n_cells`` on the dual one (``shift = 1``, where the outputs are
    the cell centers between the ``n_cells + 1`` lattice faces).

    Parameters
    ----------
    n_cells : int
        The mesh cell count of the walled axis.
    shift : int
        The cell-frame shift (0 or 1; module docstring).

    Returns
    -------
    int
        The number of output faces.
    """
    return n_cells - 1 + shift


def fully_patched(n_cells: int, rows: int, shift: int) -> bool:
    """
    Whether every output face of a walled axis is a reduced face.

    Description
    -----------
    A graded operator keeps its wide interior pass only on the faces
    farther than ``rows`` (its ``K``) from **both** walls; the rest is
    rebuilt by the ladder from true DOFs alone. When the axis has at
    most ``2 * rows`` output faces no face keeps the interior pass, so
    the operator reads **no ghost slot at all** along that axis — its
    honest ghost demand there is zero, and the storage need not carry
    the halo its interior window would otherwise ask for. This is the
    predicate the graded kernels' ``requirements`` and the advection
    modules' bind-time demand consult; ``apply_graded_walls`` needs no
    such gate (every face is assigned to its nearer wall whatever the
    count), and the shared stencil tails skip their halo guard and, when
    the window does not even fit the storage, the kernel itself for a
    fully patched axis (``patched=True``).

    A one-cell walled axis is the limiting case: zero output faces on
    the primal staggering (nothing to reconstruct), one on the dual
    (the wall-adjacent rung reading synthesized wall values only).

    Parameters
    ----------
    n_cells : int
        The mesh cell count of the walled axis.
    rows : int
        The ladder's reduced faces per side (`biased_rows` /
        `centered_rows`).
    shift : int
        The cell-frame shift (0 or 1).

    Returns
    -------
    bool
        True iff no output face keeps the interior pass.
    """
    return n_interior_faces(n_cells, shift) <= 2 * rows


# ================================================================
#  Wall-window assembly and the decomposition seam
# ================================================================
def _wall_cells(
    rung: Rung, side: int, distance: int, shift: int, n_faces: int,
) -> tuple[int, int]:
    """
    Synthesized wall cells at the head/tail of a rung window.

    Description
    -----------
    Static (the ladder's index arithmetic). In the lattice frame the
    wall cells are cell ``0`` and cell ``n_faces`` (the last lattice
    index: ``n_cells - 1 + shift``). The rung at distance ``d`` from
    the left wall starts at cell ``d - 1 - offset`` and ends
    ``size - 1`` cells later; mirrored for the right wall. Each end is
    a synthesized wall cell exactly when it lands on cell ``0`` /
    ``n_faces`` — **both** ends are checked, because on a short axis a
    face is patched from its nearer wall with a window that may still
    touch the far wall (the equidistant face of a dual-direction ladder
    under the right bias). Only a ``shift = 1`` operand *has* wall
    cells; on a BC-free (``shift = 0``) operand both counts are zero and
    the window is a plain true-DOF slice.

    Parameters
    ----------
    rung : Rung
        The ladder rung.
    side : int
        0 for the left wall, 1 for the right.
    distance : int
        The face's distance ``d`` from that wall (1 = wall-adjacent).
    shift : int
        The cell-frame shift (0 or 1).
    n_faces : int
        The output face count of the axis (`n_interior_faces`), i.e.
        the last lattice cell index.

    Returns
    -------
    tuple[int, int]
        The (leading, trailing) synthesized wall-cell counts.
    """
    if side == 0:
        start = distance - 1 - rung.offset
    else:
        start = n_faces - distance - rung.offset
    end = start + rung.size - 1
    return ((shift if start == 0 else 0),
            (shift if end == n_faces else 0))


def _rung_value(
    storage: Array,
    axis_index: int,
    width: int,
    face: int | Array,
    rung: Rung,
    shift: int,
    walls: tuple[int, int],
    co_storages: Sequence[tuple[Array, int]] = (),
) -> Array:
    """
    Evaluate one rung at a local ``face`` (interior DOFs + wall zeros).

    Description
    -----------
    The window's true-DOF stretch is taken with
    ``lax.dynamic_slice_in_dim`` — ``face`` may be a traced scalar (the
    right wall under ``shard_map``), and the slice degenerates to a
    static one when it is a Python int, so the single-shard result is
    bitwise the plain slice. The ``walls`` counts prepend/append exact
    zeros (the homogeneous Dirichlet wall values); no ghost slot is ever
    read. The result keeps a size-1 ``axis_index``.

    A ``co_storages`` entry (geometry the kernel reads alongside the
    operand — the lattice-cell widths of a stretched factor) is sliced
    with the **same** window and **no** synthesis: its wall slots hold
    real data (the clipped half cells), which the rung must read rather
    than replace by zeros. The slice therefore spans the full
    ``rung.size``, starting at the operand's un-synthesized window start
    plus the entry's frame offset.

    Parameters
    ----------
    storage : Array
        The operand's storage-shaped block (halos included).
    axis_index : int
        The stencil axis.
    width : int
        The operand's leading ghost width along the axis.
    face : int | Array
        The **local** face index (1-based; may be traced).
    rung : Rung
        The ladder rung to evaluate.
    shift : int
        The cell-frame shift (0 or 1).
    walls : tuple[int, int]
        The (leading, trailing) synthesized wall-cell counts.
    co_storages : Sequence[tuple[Array, int]], optional
        Extra storage arrays with their frame offset relative to the
        operand's frame (default: ()).

    Returns
    -------
    Array
        The single face value (size-1 along ``axis_index``).
    """
    lead, trail = walls
    true_cells = rung.size - lead - trail
    start = width + (face - 1 - rung.offset) - shift
    pieces: list[Array] = []
    zero = None
    if lead or trail:
        zero = jnp.zeros_like(jax.lax.dynamic_slice_in_dim(
            storage, width, 1, axis_index))
    if lead:
        pieces.append(zero)
    if true_cells:
        pieces.append(jax.lax.dynamic_slice_in_dim(
            storage, start + lead, true_cells, axis_index))
    if trail:
        pieces.append(zero)
    window = (pieces[0] if len(pieces) == 1
              else jnp.concatenate(pieces, axis=axis_index))
    co_windows = tuple(
        jax.lax.dynamic_slice_in_dim(
            co, start + offset, rung.size, axis_index)
        for co, offset in co_storages)
    return rung.kernel(window, axis_index, *co_windows)


def _set_slot(
    data: Array, axis_index: int, slot: int | Array, value: Array,
) -> Array:
    """
    Overwrite one output face slot (``value`` keeps a size-1 axis).

    Description
    -----------
    ``slot`` may be a traced scalar (the right wall under ``shard_map``);
    a ``lax.dynamic_update_slice_in_dim`` writes the size-1 ``value`` at
    ``slot`` (a static update when ``slot`` is a Python int, so the
    single-shard result is bitwise the plain ``.at[slot].set``).

    Parameters
    ----------
    data : Array
        The output storage block.
    axis_index : int
        The stencil axis.
    slot : int | Array
        The output storage index to overwrite.
    value : Array
        The size-1 replacement value.

    Returns
    -------
    Array
        The updated block.
    """
    return jax.lax.dynamic_update_slice_in_dim(
        data, value, slot, axis_index)


def apply_graded_walls(
    f: FieldLike,
    axis: str,
    interior: FieldLike,
    rungs: Sequence[Rung],
    shift: int,
    co_storages: Sequence[tuple[Array, int]] = (),
) -> FieldLike:
    """
    Overwrite the ``K`` wall faces per side of an interior pass.

    Description
    -----------
    The shared tail of every graded operator: the caller has already run
    its wide interior kernel over the halo-extended storage (whose
    near-wall output slots are computed from exterior/ghost values and
    are discarded here), and hands the two physical-wall ends to the
    decomposition seam ``patch_physical_ends`` with a shard-blind
    ``patch`` closure. Per wall the closure overwrites the ``K`` reduced
    faces of one local block from that block's wall-side interior cells
    (indexed off the **local** true count ``t_out``, never the global
    ``n``), synthesizing the exact-zero Dirichlet wall cells where the
    ladder reaches them (``shift = 1``). Nothing outside the operand's
    true DOFs is read, so the result is finite even when every ghost slot
    holds a NaN.

    On a **short** axis (fewer than ``2K`` output faces) the two walls'
    reduced faces overlap. Every face is then patched from its
    **nearer** wall (ties go to the left wall), with that wall's rung at
    the face's distance: its window fits between the two walls for
    either bias (`biased_offset`: a left-biased rung of order
    ``2d - 1`` at face ``d`` spans cells ``0 .. 2d - 2``, a right-biased
    one ``1 .. 2d - 1 <= n_faces``), and a window touching the far wall
    cell synthesizes it like the near one (`_wall_cells`). No face is
    written twice, so the result is independent of the patch order, and
    on an axis with no output face at all (the primal staggering of a
    one-cell axis) nothing runs. Keyed on the static global face count
    (`n_interior_faces` of the mesh), so it is correct under
    ``shard_map`` too — an axis short enough to overlap is never sharded
    (negotiation keeps every shard at least ``width + 1`` cells wide).

    Parameters
    ----------
    f : FieldLike
        The operand field (storage-shaped ``_data``).
    axis : str
        The resolved coordinate axis (bounded).
    interior : FieldLike
        The interior kernel's output field (on the graded codomain).
    rungs : Sequence[Rung]
        The ladder, widest first and wall-adjacent last
        (``K = len(rungs)``); an empty ladder returns ``interior``.
    shift : int
        The cell-frame shift (0 or 1; module docstring).
    co_storages : Sequence[tuple[Array, int]], optional
        Geometry the rung kernels read alongside the operand, each with
        its storage-frame offset relative to the operand's frame (e.g.
        ``((widths, 0),)`` for a width array in the operand's own
        frame). Sliced with the operand's window and handed to
        ``Rung.kernel`` as trailing arguments; **not** wall-synthesized
        (`_rung_value`) (default: ()).

    Returns
    -------
    FieldLike
        ``interior`` with its two wall ends replaced (metadata and
        halo-validity claim carried over).
    """
    k = len(rungs)
    if k == 0:
        return interior
    space = f.function_space.bare
    axis_index = space.names.index(axis)
    n_faces = n_interior_faces(space.factor(axis).mesh.n_cells, shift)
    if n_faces == 0:
        return interior

    offsets = tuple(offset for _co, offset in co_storages)

    def patch(
        in_block: Array,
        out_block: Array,
        side: int,
        width_in: int,
        t_in: int | Array,  # noqa: ARG001 — windows anchor on the face
        width_out: int,
        t_out: int | Array,
        *co_blocks: Array,
    ) -> Array:
        """Overwrite one wall's reduced faces of a block."""
        co_blocked = tuple(zip(co_blocks, offsets, strict=True))
        for d in range(1, min(k, n_faces) + 1):
            # a face nearer to the other wall is that wall's to patch
            # (an equidistant face is the left wall's)
            other = n_faces - d + 1
            if other < d or (side == 1 and other == d):
                continue
            rung = rungs[k - d]
            face = d if side == 0 else t_out - d + 1
            walls = _wall_cells(rung, side, d, shift, n_faces)
            value = _rung_value(in_block, axis_index, width_in, face,
                                rung, shift, walls, co_blocked)
            slot = width_out + (face - 1)  # output DOF k = face - 1
            out_block = _set_slot(out_block, axis_index, slot, value)
        return out_block

    data = f.grid.decomposition.patch_physical_ends(
        interior._data,  # noqa: SLF001 — plumbing seam
        f._data,  # noqa: SLF001 — documented storage seam
        interior.function_space.bare, space, axis, patch,
        layout=f.function_space.layout,
        co_arrays=tuple(co for co, _offset in co_storages))

    return type(f)(f.grid, interior.function_space, data, f.metadata,
                   halo_valid=interior.halo_valid)


# ================================================================
#  Mask-keyed graded closure (immersed grids)
# ================================================================
#: the all-wet product of a ``{0, 1}`` present-mask window is 1.0 iff
#: every slot is present; anything below this threshold has a dry slot
_PRESENT_THRESHOLD = 0.5


def _window_all_wet(size: int) -> Callable[[Array, int], Array]:
    """
    Windowed all-wet product kernel over a boolean-as-float mask.

    Description
    -----------
    The mask-keyed sibling of the value kernels: a ``size``-wide sliding
    window whose output is the product of the window slots (``1.0`` iff
    every slot is wet, ``0.0`` otherwise). The axis shrinks by
    ``size - 1`` exactly like the value rungs, so the same
    ``apply_fv_staggered`` alignment lands the selector on the output
    face its rung serves.

    Parameters
    ----------
    size : int
        The window width in operand cells.

    Returns
    -------
    Callable[[Array, int], Array]
        The ``(mask_storage, axis_index) -> windowed_product`` kernel.
    """
    def kernel(arr: Array, axis_index: int) -> Array:
        out_len = arr.shape[axis_index] - size + 1
        index: list[slice] = [slice(None)] * arr.ndim
        product: Array | None = None
        for offset in range(size):
            index[axis_index] = slice(offset, offset + out_len)
            term = arr[tuple(index)]
            product = term if product is None else product * term
        return product

    return kernel


def apply_graded_mask(
    op: Operator,
    f: FieldLike,
    axis: str,
    interior: FieldLike,
    rungs: Sequence[Rung],
    sel_specs: Sequence[tuple[int, int]],
    shift: int,
    present: FieldLike,
) -> FieldLike:
    r"""
    Select, per output face, the widest wet rung of the ladder (immersed).

    Description
    -----------
    The mask-keyed sibling of :func:`apply_graded_walls` (decisions
    GA-D1/D2 of ``immersed_graded_advection_plan.md``). On an immersed
    grid dry DOFs may sit anywhere, so there is no static two-wall index
    partition; instead every rung is evaluated **full-array** over the
    pre-masked operand ``f`` and the face value is a nested ``jnp.where``
    over per-rung selector fields, widest first.

    - Each reduced ``rung`` is run through the same
      :func:`~fridom.spatial.operators.reconstruct.apply_fv_staggered`
      plumbing as the interior pass (``align = rung.offset + shift``), so
      its window arithmetic is identical to the wall path's
      ``_rung_value``.
    - The selector of the rung at ``sel_specs[i] = (size, offset)`` is the
      windowed all-wet product of ``present`` (the sign-independent
      **union** window; :func:`_window_all_wet`), a static
      trace-time-constant field: ``d >= t`` for that rung is exactly its
      union window being entirely present. ``present`` marks a slot wet
      **or** structurally exempt (the wall-side zero of the ``shift = 1``
      dual direction), so a face-aligned staircase reproduces the wall
      ladder by construction. The bottom rung (``sel_specs[-1]``) carries
      no selector — it is the unconditional fallback (any ``alpha > 0``
      face has two wet neighbours, so it is always legal).

    Because ``present`` is a superset of the pre-mask, the selected rung's
    window reads only wet real values and structural zeros — bitwise the
    wall path where the wet region is face-aligned. The pre-mask (dry ->
    exact zero) reproduces the synthesized Dirichlet zeros, is NaN-safe,
    and seals the reverse VJP at dead slots.

    Parameters
    ----------
    op : Operator
        The (bound) separable kernel (resolves the codomain / alignment).
    f : FieldLike
        The **pre-masked** operand field (dry slots zeroed).
    axis : str
        The resolved coordinate axis.
    interior : FieldLike
        The widest-rung (interior kernel) output field, on the codomain.
    rungs : Sequence[Rung]
        The reduced ladder, widest first and bottom (wall-adjacent) last
        (``K = len(rungs)``); the interior is prepended implicitly.
    sel_specs : Sequence[tuple[int, int]]
        Per-rung union-window ``(size, offset)``, one for the interior
        (index 0) and one per reduced rung (length ``K + 1``); the last
        (bottom) entry is unused.
    shift : int
        The cell-frame shift (0 or 1).
    present : FieldLike
        The selector present-mask on the operand space (wet-or-exempt),
        as a real ``{0, 1}`` field.

    Returns
    -------
    FieldLike
        ``interior`` with every face replaced by its widest wet rung.
    """
    from fridom.spatial.operators.reconstruct import (  # noqa: PLC0415
        apply_fv_staggered,
    )

    values = [interior._data]  # noqa: SLF001 — documented storage seam
    for rung in rungs:
        reduced = apply_fv_staggered(
            op, f, axis, rung.size, rung.kernel, metadata=None,
            align=rung.offset + shift)
        values.append(reduced._data)  # noqa: SLF001 — storage seam
    selectors: list[Array] = []
    for size, offset in sel_specs[:-1]:
        sel = apply_fv_staggered(
            op, present, axis, size, _window_all_wet(size),
            metadata=None, align=offset + shift)
        wet = sel._data > _PRESENT_THRESHOLD  # noqa: SLF001 — storage seam
        selectors.append(wet)
    data = values[-1]
    for i in range(len(selectors) - 1, -1, -1):
        data = jnp.where(selectors[i], values[i], data)
    return type(f)(f.grid, interior.function_space, data,
                   interior.metadata, halo_valid=interior.halo_valid)
