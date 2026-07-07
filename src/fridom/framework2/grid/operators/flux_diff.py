"""
The finite-volume flux-difference operators and ``FVDerivative``.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_stencils.md``.
``FluxDifference`` is the exact discrete Gauss theorem on the primal
cells (``Outer/Inner -> CellAvg``, periodic ``Right -> CellAvg``);
``DualFluxDifference`` its dual-cell twin (``Center/CellAvg ->
FaceAvg``, the momentum-control-volume derivative);
``FaceDifference`` the FV pressure gradient (``CellAvg -> Right |
Inner``, its own ``"face_diff"`` kind). ``FVDerivative`` is the
factory building the normative ``("diff", CellAvg)`` default
``flux_diff @ Dispatched("reconstruct")``.

Spacing denominators are the mesh's uniform cell width read at trace
time (the same iteration-1 ``grid.measure`` stand-in the Wave-2C
``FiniteDifference`` uses); nonuniform measure fields arrive with
the mapped meshes.
"""
# Wave 3: FluxDifference, DualFluxDifference, FaceDifference,
#    FVDerivative
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.fields.storage import store
from fridom.framework2.grid.operators.base import (
    Dispatched,
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
)
from fridom.framework2.grid.operators.interned import interned
from fridom.framework2.grid.operators.reconstruct import (
    apply_fv_staggered,
    factor_codomain,
)
from fridom.framework2.grid.operators.staggering import (
    uniform_spacing,
)
from fridom.framework2.grid.operators.stencil_kernels import (
    staggered_diff,
)
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.average import CellAvg
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

    from fridom.framework2.grid.operators.base import (
        Operator,
        SeparableComposite,
    )
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )

_DIFF_SIZE = 2


def _mesh_space(
    domain: FunctionSpace, attr: str, operation: str,
) -> FunctionSpace:
    """
    Fetch a codomain space family off the domain's mesh.

    Parameters
    ----------
    domain : FunctionSpace
        The bare 1D domain factor.
    attr : str
        The mesh factory attribute naming the codomain family.
    operation : str
        The dispatch kind, for the error message.

    Returns
    -------
    FunctionSpace
        The codomain factor, with the domain's scalars carried over.
    """
    try:
        codomain: FunctionSpace = getattr(domain.mesh, attr)
    except ValueError as exc:  # mesh lacks the codomain family
        raise SpaceMismatchError(
            f"no {operation} signature on {domain!r}: "
            f"{domain.mesh!r} has no {attr} space", left=domain,
            operation=operation) from exc
    if domain.scalars is Scalars.COMPLEX:
        return codomain.as_complex()
    return codomain


def _windowed_diff(
    op: SeparableOperator, f: FieldLike, axis: str,
) -> FieldLike:
    """Two-point difference via the aligned FV window machinery."""
    spacing = uniform_spacing(f.function_space.bare.factor(axis))

    def kernel(arr: Array, axis_index: int) -> Array:
        return staggered_diff(arr, axis_index, spacing=spacing,
                              order=_DIFF_SIZE)

    return apply_fv_staggered(op, f, axis, _DIFF_SIZE, kernel,
                              metadata=None)


@final
@interned
class FluxDifference(SeparableOperator):

    """
    Exact flux difference: Outer -> CellAvg (discrete Gauss).

    Description
    -----------
    ``(u_{i+1/2} - u_{i-1/2}) / w_i`` with the primal cell width as
    denominator — exactness is the contract: summing the output
    against the cell measure telescopes to the boundary fluxes.
    ``Inner -> CellAvg`` is the homogeneous no-normal-flow variant
    (zero boundary fluxes imposed exactly, never read from the
    BC-free extrapolation ghosts); inhomogeneous boundary fluxes
    occupy the boundary DOFs of an ``Outer``-space flux field. The
    ``Right -> CellAvg`` row is explicitly periodic-only.
    ``eigenvalues`` (``i k sinc(k dx / 2)``) is designed-for until
    the ``Symbol`` cluster lands (Wave 3B).
    """

    dispatch_kind: ClassVar[str | None] = "flux_diff"

    def _intern_key(self) -> tuple:
        """Structural key: no constructor state (D6)."""
        return ()

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Resolve the primal flux-difference codomain.

        Description
        -----------
        flux_diff: Outer(n+1) | Inner(n-1) -> CellAvg(n) (bounded);
        Right(n) -> CellAvg(n) (periodic only).

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D face-space factor holding the fluxes.

        Returns
        -------
        FunctionSpace
            The ``CellAvg`` factor (scalars preserved).
        """
        supported = (
            isinstance(domain, NodalSpace) and domain.bc.is_free
            and (domain.node_set in {NodeSet.OUTER, NodeSet.INNER}
                 or (domain.node_set is NodeSet.RIGHT
                     and domain.mesh.periodic)))
        if not supported:
            if (isinstance(domain, NodalSpace)
                    and domain.node_set is NodeSet.RIGHT):
                raise SpaceMismatchError(
                    "flux_diff: Right -> CellAvg is periodic-only "
                    "(on a bounded mesh Right lacks the left "
                    "boundary face); present fluxes on Outer "
                    "(explicit boundary fluxes) or Inner "
                    f"(homogeneous), got {domain!r}",
                    left=domain, operation="flux_diff")
            raise SpaceMismatchError(
                f"no flux_diff signature on {domain!r}: "
                "Outer/Inner -> CellAvg (bounded), "
                "Right -> CellAvg (periodic); center-domain rows "
                "are DualFluxDifference",
                left=domain, operation="flux_diff")
        return _mesh_space(domain, "cell_avg", "flux_diff")

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — fixed two-point halo
    ) -> OperatorRequirements:
        """
        Declare halo = 1, layout "any".

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=1)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Difference the face fluxes into cell averages.

        Description
        -----------
        ``Outer``/``Right`` domains run the aligned window kernel
        (the periodic wrap ghost supplies the left face). The
        ``Inner`` domain pads the true-shape fluxes with exact zeros
        at both boundary faces (the homogeneous no-normal-flow
        contract) before differencing.

        Parameters
        ----------
        f : FieldLike
            The operand flux field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The flux-difference field (default metadata).
        """
        bare = f.function_space.bare
        factor = bare.factor(axis)
        if factor.node_set is not NodeSet.INNER:
            return _windowed_diff(self, f, axis)
        # homogeneous variant: exact zero boundary fluxes (the
        # BC-free ghost extrapolation must never leak in here)
        codomain = factor_codomain(self, f.function_space, axis)
        axis_index = bare.names.index(axis)
        spacing = uniform_spacing(factor)
        pads = [(0, 0)] * len(bare.shape)
        pads[axis_index] = (1, 1)
        flux = jnp.pad(f.data, pads)
        data = staggered_diff(flux, axis_index, spacing=spacing,
                              order=_DIFF_SIZE)
        stored = store(f.grid.decomposition, codomain, data)
        return type(f)(f.grid, codomain, stored, None)


@final
@interned
class DualFluxDifference(SeparableOperator):

    """
    Exact dual-cell flux difference: Center -> FaceAvg.

    Description
    -----------
    The discrete Gauss theorem on the dual cells: exact FTC when the
    domain holds point values at centers (``Center -> FaceAvg``); the
    ``CellAvg -> FaceAvg`` row carries the declared O(dx^2)
    identification of cell averages with midpoint values. Registered
    under the same ``"flux_diff"`` kind, keyed by the center domains.
    ``eigenvalues`` (``i k sinc(k w / 2)`` on the dual mesh) is
    designed-for until the ``Symbol`` cluster lands (Wave 3B).
    """

    dispatch_kind: ClassVar[str | None] = "flux_diff"

    def _intern_key(self) -> tuple:
        """Structural key: no constructor state (D6)."""
        return ()

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Resolve the dual flux-difference codomain.

        Description
        -----------
        flux_diff: Center -> FaceAvg (exact FTC); CellAvg -> FaceAvg
        (O(dx^2) identification).

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D center-family factor holding the fluxes.

        Returns
        -------
        FunctionSpace
            The ``FaceAvg`` factor (scalars preserved).
        """
        supported = (
            isinstance(domain, CellAvg)
            or (isinstance(domain, NodalSpace) and domain.bc.is_free
                and domain.node_set is NodeSet.CENTER))
        if not supported:
            raise SpaceMismatchError(
                f"no flux_diff signature on {domain!r}: "
                "Center/CellAvg -> FaceAvg (dual cells); face-domain "
                "rows are FluxDifference",
                left=domain, operation="flux_diff")
        return _mesh_space(domain, "face_avg", "flux_diff")

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — fixed two-point halo
    ) -> OperatorRequirements:
        """
        Declare halo = 1, layout "any".

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=1)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Difference the center fluxes into dual-cell averages.

        Parameters
        ----------
        f : FieldLike
            The operand flux field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The dual flux-difference field (default metadata).
        """
        return _windowed_diff(self, f, axis)


@final
@interned
class FaceDifference(SeparableOperator):

    """
    FV pressure gradient: CellAvg -> face point values.

    Description
    -----------
    ``(p_{i+1} - p_i) / d_{i+1/2}`` with the dual center-to-center
    spacing as denominator, landing on the point-value face space
    where the C-grid momentum DOFs live. A dedicated ``"face_diff"``
    kind: ``("diff", CellAvg)`` is the normative ``FVDerivative``
    composition (``CellAvg -> CellAvg``). ``eigenvalues`` is
    designed-for until the ``Symbol`` cluster lands (Wave 3B).
    """

    dispatch_kind: ClassVar[str | None] = "face_diff"

    def _intern_key(self) -> tuple:
        """Structural key: no constructor state (D6)."""
        return ()

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        face_diff: CellAvg -> Right (periodic) | Inner (bounded).

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D ``CellAvg`` factor.

        Returns
        -------
        FunctionSpace
            The face point-value factor (scalars preserved).
        """
        if not isinstance(domain, CellAvg):
            raise SpaceMismatchError(
                f"no face_diff signature on {domain!r}: "
                "CellAvg -> Right (periodic) / Inner (bounded)",
                left=domain, operation="face_diff")
        attr = "right" if domain.mesh.periodic else "inner"
        return _mesh_space(domain, attr, "face_diff")

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — fixed two-point halo
    ) -> OperatorRequirements:
        """
        Declare halo = 1, layout "any".

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=1)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Difference the cell values onto the faces.

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The face-difference field (default metadata).
        """
        return _windowed_diff(self, f, axis)


def FVDerivative(  # noqa: N802 — factory named like the class docs
    reconstruct: Operator | None = None,
) -> SeparableComposite:
    """
    Build ``flux_diff @ (reconstruct or Dispatched("reconstruct"))``.

    Description
    -----------
    The default ``("diff", CellAvg)`` entry: not a class but a
    factory building the algebra chain; the result is an ordinary
    ``SeparableComposite`` (``CellAvg -> CellAvg``). With
    ``reconstruct=None`` the reconstruction is a ``Dispatched``
    placeholder resolved once at registry merge (D4), so a module
    override of ``"reconstruct"`` still propagates into what
    ``f.diff("x")`` does on average spaces; an explicit kernel pins
    the composition. The chain binds with ``["x"]`` and its halo is
    the sum of the factor halos.

    Parameters
    ----------
    reconstruct : Operator | None, optional
        An explicit reconstruction kernel pinning the composition;
        None leaves the dispatched placeholder (default: None).

    Returns
    -------
    SeparableComposite
        The composed FV derivative.
    """
    if reconstruct is None:
        reconstruct = Dispatched("reconstruct")
    return FluxDifference() @ reconstruct
