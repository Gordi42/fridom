r"""
``CumulativeIntegral``: the running (partial) integral along one axis.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_products.md``
("Reductions"); design source
``design/plans/active/hydrostatic_model_plan.md`` (stage H1). Where
:class:`~fridom.spatial.operators.integrate.Integral` *reduces* an
axis to its ``ConstantSpace`` total, ``CumulativeIntegral``
(``"cumint"``) keeps the whole axis: it is the staggered running sum
that carries a center-valued integrand to the running integral at
every face (the two hydrostatic DIAGNOSE primitives,

.. math::

    w(z) = -\int_{-H}^{z} \nabla_h\cdot u_h \, dz', \qquad
    p_{hyd}(z) = -\int_{z}^{0} b \, dz').

Discrete structure (the design decisions, per stage H1)
-------------------------------------------------------
Two target lattices, both seeded from the same measure-weighted
increment ``incr[i] = f[i] * grid.measure(domain)[i]`` (the primal
cell width — the exact ``Integral`` weight, and the exact ``diff``
denominator of the matching staggered difference):

- **face** (``Center -> Outer``, ``CellAvg -> Outer``) is the
  fundamental-theorem-exact form. The running integral lives at the
  faces (both boundaries, so ``w`` exists at top and bottom), zero at
  the seeding boundary. Two exactness guarantees hold to machine
  precision: the matching staggered difference recovers the
  integrand — ``FiniteDifference(Outer -> Center)`` on the nodal row,
  ``FluxDifference(Outer -> CellAvg)`` (the exact discrete Gauss
  theorem) on the FV row — and the accumulated value at the far
  boundary telescopes to ``Integral`` of the same field. The
  cell-average cumulative sum is *exact* for averages
  (``avg[i]*dz == \int_{cell} f``), so the FV face form is the true
  running integral, not an approximation.

- **center** (``Center -> Center``, ``CellAvg -> CellAvg``) is the
  co-located form the C-grid pressure wants: ``p_hyd`` lands where
  ``b`` lives, so ``\nabla_h p_hyd`` reaches the ``u``/``v`` faces
  through the ordinary ``FaceDifference``/``FiniteDifference``. It is
  the midpoint of the face accumulation
  (``center[k] = (face[k] + face[k+1]) / 2``) — the pyOM2 section 7.1
  half-cell hydrostatic pressure. On ``CellAvg`` this midpoint is the
  *exact* cell average of the (piecewise-linear) running integral; on
  ``Center`` it is the running integral at the cell centre to
  ``O(dz^2)``. Its discrete vertical balance is
  ``FiniteDifference(center) == LinearInterp(integrand)`` on the inner
  faces (exact), **not** the machine-exact ``diff(cumint) == id`` of
  the face form — a staggered lattice cannot carry both the integral
  and its pointwise derivative at the same nodes. Where a
  machine-exact fundamental theorem at centres is required, compose
  the **face** form with an ``Outer -> Center`` interpolation
  instead; the ``center`` target *is* that composition, pre-baked.

Direction (either bounded end)
------------------------------
``"up"`` seeds the zero at the low-index (``x_min``) boundary face and
accumulates toward ``x_max`` (bottom-up, ``w``); ``"down"`` seeds it
at the high-index (``x_max``) face and accumulates toward ``x_min``
(top-down, ``p_hyd``). The far-boundary value telescopes to
``Integral`` in both directions; the matching difference of the
``"down"`` form returns the *negated* integrand (the derivative of an
integral taken from the upper limit). A periodic axis has no boundary
to seed the zero from: the operator raises a taught
``SpaceMismatchError`` naming the axis.

Weighting and charts
--------------------
The increment weight is ``grid.measure(domain, name=axis)`` (rules
sections 2.7, 3.9), so stretched/mapped meshes enter through the
measure exactly as they do for ``Integral`` — no ``jacobian=`` needed.
The optional ``jacobian=`` family mirrors ``Integral``'s parameter
and names *chart coordinates* (``jacobian_weight``): an embedding
chart's base coordinate adds the ``sqrt_g`` area element once (while
every chart coordinate is still resolved by the operand space), and
an analytic-``maps=`` physical coordinate ``p`` adds the column
Jacobian ``d<p>_d<b>`` on its base axis ``b`` — the terrain-following
running integral :math:`\int f\,dz_p`. Either way the physical
running integral telescopes to the Jacobian-weighted ``Integral``
(rules 3.13); a name off every chart raises a taught error.

Decomposition
-------------
A running sum needs the whole axis in one place, so the operator
declares ``layout="local"`` on the integration axis — the same
axis-local contract ``VerticalDiffusion``'s solve declares. When the
operand arrives with that axis sharded the kernel reshards it onto a
negotiated axis-local layout (``decomposition.layout_for`` + the
``Reshard`` movement operator — no ``.data`` bypass, no hand-built
shardings), accumulates locally, and reshards the result back to the
operand's layout, so the axis carries no cross-shard prefix scan.
"""
# Stage H1: CumulativeIntegral (hydrostatic DIAGNOSE primitive)
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.storage import store
from fridom.spatial.operators.base import (
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
    resolve_codomain,
)
from fridom.spatial.operators.interned import interned
from fridom.spatial.operators.jacobian_weight import jacobian_factor
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.coefficient import CoefficientSpace
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )

#: the two seeding ends (see the module docstring)
_DIRECTIONS = ("up", "down")

#: the two output lattices (see the module docstring)
_TARGETS = ("face", "center")


@final
@interned
class CumulativeIntegral(SeparableOperator):

    r"""
    Running (partial) integral of a center-valued integrand.

    Description
    -----------
    A ``SeparableOperator`` on one **bounded** axis: nodal ``Center``
    and FV ``CellAvg`` integrands only. ``target="face"`` lands on the
    both-boundary face set ``Outer`` (the fundamental-theorem-exact
    form — the matching ``FiniteDifference``/``FluxDifference``
    recovers the integrand and the far value telescopes to
    ``Integral``, both to machine precision); ``target="center"`` lands
    co-located (the pyOM half-cell form, module docstring).
    ``direction`` seeds the zero at either bounded end. With
    ``jacobian=`` set the increment of a chart coordinate additionally
    carries the metric Jacobian — the ``sqrt_g`` area element on an
    embedding chart, or the column Jacobian on an analytic-``maps=``
    terrain column (mirroring ``Integral``). The axis is declared
    ``layout="local"``; a sharded axis is resharded onto a negotiated
    axis-local layout and back (module docstring).

    Parameters
    ----------
    direction : str, optional
        ``"up"`` seeds the zero at the ``x_min`` face and accumulates
        toward ``x_max`` (bottom-up); ``"down"`` seeds it at ``x_max``
        and accumulates toward ``x_min`` (top-down) (default: "up").
    target : str, optional
        ``"face"`` lands on ``Outer`` (both boundary faces);
        ``"center"`` lands co-located with the integrand
        (default: "face").
    jacobian : tuple[str, ...] | None, optional
        Chart coordinate names whose increment picks up the metric
        Jacobian weight — an embedding chart's base coordinates
        (``sqrt_g``) or an analytic-``maps=`` physical coordinate
        (its column Jacobian). None keeps the plain computational
        measure; a name off every chart raises (default: None).
    """

    dispatch_kind: ClassVar[str | None] = "cumint"

    def __init__(
        self,
        direction: str = "up",
        target: str = "face",
        jacobian: tuple[str, ...] | None = None,
    ) -> None:
        """Validate and store the direction, target, and chart family."""
        if direction not in _DIRECTIONS:
            raise ValueError(
                f"direction must be one of {_DIRECTIONS}, got "
                f"{direction!r}")
        if target not in _TARGETS:
            raise ValueError(
                f"target must be one of {_TARGETS}, got {target!r}")
        if jacobian is not None:
            jacobian = tuple(jacobian)
            if not jacobian or not all(
                    isinstance(name, str) for name in jacobian):
                raise TypeError(
                    "jacobian names chart coordinates: a non-empty "
                    f"tuple of strings, got {jacobian!r}")
        self._direction: str = direction
        self._target: str = target
        self._jacobian: tuple[str, ...] | None = jacobian

    def _intern_key(self) -> tuple:
        """Structural key: direction, target, and Jacobian family (D6)."""
        return (self._direction, self._target, self._jacobian)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def direction(self) -> str:
        """The seeding end (``"up"`` = ``x_min``, ``"down"`` = ``x_max``)."""
        return self._direction

    @property
    def target(self) -> str:
        """The output lattice (``"face"`` = Outer, ``"center"``)."""
        return self._target

    @property
    def jacobian(self) -> tuple[str, ...] | None:
        """Chart coordinates carrying the sqrt_g weight, or None."""
        return self._jacobian

    # ================================================================
    #  Signature
    # ================================================================
    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        cumint: Center/CellAvg -> Outer (face) | same set (center).

        Description
        -----------
        The integrand lives at cell centres (nodal BC-free ``Center``
        or FV ``CellAvg``); a periodic axis, a coefficient factor, or
        any face/boundary node set raises. ``target="face"`` lands on
        the both-boundary ``Outer`` face set; ``target="center"``
        keeps the integrand's node set (the co-located midpoint form).

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D integrand factor.

        Returns
        -------
        FunctionSpace
            The running-integral codomain factor (scalars preserved).
        """
        if isinstance(domain, ConstantSpace):
            return domain
        if isinstance(domain, CoefficientSpace):
            raise SpaceMismatchError(
                "cumint has no coefficient-space signature: transform "
                f"back to a nodal/average factor first, got {domain!r}",
                left=domain, operation="cumint")
        center_valued = (
            isinstance(domain, CellAvg)
            or (isinstance(domain, NodalSpace)
                and domain.node_set is NodeSet.CENTER
                and domain.bc.is_free))
        if not center_valued:
            raise SpaceMismatchError(
                f"no cumint signature on {domain!r}: the running "
                "integral takes a center-valued integrand (BC-free "
                "nodal Center, or FV CellAvg) only",
                left=domain, operation="cumint")
        if domain.mesh.periodic:
            raise SpaceMismatchError(
                f"cumint along {domain.names[0]!r} is ill-posed on a "
                "periodic axis: a running integral needs a bounded "
                "axis to seed its zero boundary value (there is no "
                "boundary face on a periodic axis); reduce with "
                "Integral instead, or use a bounded axis",
                left=domain, operation="cumint")
        if self._target == "center":
            return domain
        codomain: FunctionSpace = domain.mesh.outer
        if domain.scalars is Scalars.COMPLEX:
            codomain = codomain.as_complex()
        return codomain

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — axis-local, halo 0
    ) -> OperatorRequirements:
        """
        Declare halo = 0, layout = "local".

        Description
        -----------
        A prefix sum needs the whole axis in one place, so the
        integration axis is kept undistributed (the
        ``VerticalDiffusion`` solve-axis contract). No ghost layers
        are read — the kernel accumulates over the true-shape column.

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=0, layout="local")

    # ================================================================
    #  Kernel
    # ================================================================
    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Accumulate along ``axis``, resharding a sharded axis local.

        Description
        -----------
        On a single-device (or already axis-local) operand the
        accumulation runs directly. When the axis is sharded the
        operand is resharded onto ``decomposition.layout_for((axis,))``
        (a negotiated axis-local layout) through the ``Reshard``
        movement operator, accumulated locally, and the result
        resharded back to the operand's layout — the running sum never
        crosses a shard boundary along ``axis`` (module docstring).

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The running-integral field (default metadata: new quantity).
        """
        layout = f.function_space.layout
        decomposition = getattr(f.grid, "decomposition", None)
        if (decomposition is not None and layout is not None
                and decomposition.device_count > 1
                and not layout.is_local(axis)):
            from fridom.spatial.operators.movement import (  # noqa: PLC0415 — keep movement off the operator import path
                Reshard,
            )
            local = decomposition.layout_for((axis,))
            f_local = Reshard(f.grid, local)(f)
            accumulated = self._accumulate(f_local, axis)
            return Reshard(f.grid, layout)(accumulated)
        return self._accumulate(f, axis)

    def _accumulate(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Run the measure-weighted prefix sum on an axis-local operand.

        Description
        -----------
        Forms the increment ``f * grid.measure(domain)`` (times the
        ``sqrt_g`` chart weight where a Jacobian row resolves it),
        prefix-sums it along the axis with the zero-valued seeding
        boundary (``"up"`` at the low face, ``"down"`` at the high
        face), and — for the ``center`` target — midpoints the face
        accumulation back onto the integrand's node set. The result is
        routed into the codomain storage through ``store``, in the
        operand's (axis-local) layout frame.

        Parameters
        ----------
        f : FieldLike
            The operand field (axis undistributed).
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The running-integral field on the resolved codomain.
        """
        space = f.function_space
        bare = space.bare
        weight = f.grid.measure(bare, name=axis)
        incr = f.data * weight.data
        factor = jacobian_factor(f, axis, self._jacobian)
        if factor is not None:
            incr = incr * factor
        axis_index = bare.names.index(axis)
        running = _running_integral(incr, axis_index, self._direction)
        if self._target == "center":
            running = _midpoint(running, axis_index)
        # keep the operand's (axis-local) layout on the codomain, so
        # the padded storage matches the frame the running sum lives in
        codomain = resolve_codomain(self, space).with_layout(
            space.layout)
        stored = store(f.grid.decomposition, codomain, running)
        return type(f)(f.grid, codomain, stored, None)


# ================================================================
#  Array kernels (pure; slice-based over the true-shape column)
# ================================================================
def _zero_slice(arr: Array, axis: int) -> Array:
    """Return a size-1-along-``axis`` zero slice matching ``arr``."""
    shape = list(arr.shape)
    shape[axis] = 1
    return jnp.zeros(tuple(shape), dtype=arr.dtype)


def _running_integral(
    incr: Array, axis: int, direction: str,
) -> Array:
    """
    Prefix-sum the increments with the zero-valued seeding face.

    Description
    -----------
    ``"up"`` seeds the low (``x_min``) face at zero and accumulates
    upward, so face ``j`` holds ``sum_{i<j} incr[i]``; ``"down"`` seeds
    the high (``x_max``) face and accumulates downward, so face ``j``
    holds ``sum_{i>=j} incr[i]``. The output carries one more DOF than
    the integrand along ``axis`` (the ``n`` cells -> ``n + 1`` faces of
    the ``Outer`` node set).

    Parameters
    ----------
    incr : Array
        The measure-weighted per-cell increments (true shape).
    axis : int
        The accumulation axis index.
    direction : str
        ``"up"`` (seed at ``x_min``) or ``"down"`` (seed at ``x_max``).

    Returns
    -------
    Array
        The face-valued running integral (``+1`` DOF along ``axis``).
    """
    zero = _zero_slice(incr, axis)
    if direction == "up":
        return jnp.concatenate(
            [zero, jnp.cumsum(incr, axis=axis)], axis=axis)
    reverse = jnp.flip(
        jnp.cumsum(jnp.flip(incr, axis=axis), axis=axis), axis=axis)
    return jnp.concatenate([reverse, zero], axis=axis)


def _midpoint(face: Array, axis: int) -> Array:
    """
    Average adjacent faces onto the integrand's node set.

    Description
    -----------
    The ``Outer -> Center`` / ``Outer -> CellAvg`` midpoint of the
    face accumulation: ``center[k] = (face[k] + face[k+1]) / 2``. On a
    ``CellAvg`` integrand this is the exact cell average of the
    piecewise-linear running integral; on ``Center`` it is the running
    integral at the cell centre to ``O(dz^2)`` (module docstring).

    Parameters
    ----------
    face : Array
        The face-valued running integral (``n + 1`` DOFs along ``axis``).
    axis : int
        The accumulation axis index.

    Returns
    -------
    Array
        The co-located running integral (``n`` DOFs along ``axis``).
    """
    lower: list[object] = [slice(None)] * face.ndim
    upper: list[object] = [slice(None)] * face.ndim
    lower[axis] = slice(0, -1)
    upper[axis] = slice(1, None)
    return 0.5 * (face[tuple(lower)] + face[tuple(upper)])
