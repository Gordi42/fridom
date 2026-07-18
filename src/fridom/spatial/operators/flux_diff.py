"""
The finite-volume flux-difference operators and ``FVDerivative``.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_stencils.md``.
``FluxDifference`` is the exact discrete Gauss theorem on the primal
cells (``Outer/Inner -> CellAvg``, periodic ``Right -> CellAvg``);
``DualFluxDifference`` its dual-cell twin (``Center/CellAvg ->
FaceAvg``, the momentum-control-volume derivative);
``FaceDifference`` the FV pressure gradient (``CellAvg -> Right |
Inner``, its own ``"face_diff"`` kind). ``FVDerivative`` is the
factory building the normative ``("diff", CellAvg)`` default
``flux_diff @ Dispatched("reconstruct")``.

Spacing denominators are the codomain's ``grid.measure`` metric
fields, derived at trace time (concepts section 2.7): on uniform
meshes the constant cell width folds into the static weights (the
scalar fast path XLA folds); on mapped meshes the unit-spacing
difference divides by the codomain's measure field — the primal
cell width on ``CellAvg``, the dual center-to-center spacing on the
face family (rules section 3.9).
"""
# Wave 3: FluxDifference, DualFluxDifference, FaceDifference,
#    FVDerivative
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.storage import store
from fridom.spatial.operators.base import (
    Dispatched,
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
    _resolve_axis,
)
from fridom.spatial.operators.interned import interned
from fridom.spatial.operators.reconstruct import (
    apply_fv_staggered,
    factor_codomain,
    wall_slots_addressable,
    wall_zeroed_operand,
)
from fridom.spatial.operators.spectral import (
    finite_difference_symbol,
    fv_fourier_partner,
    fv_trig_diff_codomain,
    in_trig_family,
    trig_finite_difference_symbol,
    trig_partner,
)
from fridom.spatial.operators.staggering import (
    divide_by_codomain_measure,
    mapped_factor,
    uniform_spacing,
)
from fridom.spatial.operators.stencil_kernels import (
    staggered_diff,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.coefficient import (
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

    from fridom.spatial.operators.base import (
        Operator,
        SeparableComposite,
    )
    from fridom.spatial.operators.symbol import Symbol
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike

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
    *, operand: FieldLike | None = None,
) -> FieldLike:
    """
    Two-point difference via the aligned FV window machinery.

    Description
    -----------
    Uniform meshes fold the scalar cell width into the static
    weights (the constant special case); mapped meshes run the
    unit-spacing kernel and divide by the codomain's measure field
    (rules sections 2.7, 3.9).

    ``operand`` overrides the field the kernel reads, while the
    codomain, spacing, and measure divide still come from ``f`` (they
    share a space, grid, and layout): the homogeneous ``Inner`` arm
    feeds a wall-zeroed copy so the ordinary window imposes the exact
    zero wall flux (:meth:`FluxDifference._inner_diff_windowed`).
    ``None`` reads ``f`` directly (the ``Outer``/``Right``/dual/face
    rows).

    Parameters
    ----------
    op : SeparableOperator
        The (bound) FV difference kernel.
    f : FieldLike
        The operand flux field (supplies codomain, spacing, measure).
    axis : str
        The resolved coordinate axis.
    operand : FieldLike | None, optional
        The field the kernel actually reads; None reads ``f``
        (default: None).

    Returns
    -------
    FieldLike
        The flux-difference field (default metadata).
    """
    operand = f if operand is None else operand
    factor = f.function_space.bare.factor(axis)
    mapped = mapped_factor(factor)
    spacing = 1.0 if mapped else uniform_spacing(factor)

    def kernel(arr: Array, axis_index: int) -> Array:
        return staggered_diff(arr, axis_index, spacing=spacing,
                              order=_DIFF_SIZE)

    result = apply_fv_staggered(op, operand, axis, _DIFF_SIZE, kernel,
                                metadata=None)
    if mapped:
        return divide_by_codomain_measure(result, f, axis)
    return result


def _fv_diff_eigenvalues(
    op: SeparableOperator, space: SpaceLike, who: str,
) -> Symbol:
    r"""
    Retagging ``i k_hat`` diagonal of an FV flux/face difference.

    Description
    -----------
    The shared ``eigenvalues`` body of ``FluxDifference`` /
    ``DualFluxDifference`` / ``FaceDifference``: on a periodic,
    uniform mesh a two-point staggered difference diagonalizes in the
    Fourier basis of its origin. :func:`fv_fourier_partner` resolves
    the source Fourier factor and the periodic nodal/average origin
    (raising ``EigenbasisError`` on bounded/mapped/non-Fourier
    factors), and the order-2 staggered-FD symbol builder threads the
    codomain origin through the inter-origin phase — the same
    ``i k_hat = i k sinc(k dx / 2)`` numbers as the nodal
    ``FiniteDifference``, retagged onto the average family.

    Parameters
    ----------
    op : SeparableOperator
        The (bound) FV difference kernel.
    space : SpaceLike
        The coefficient factor (or product) space threaded.
    who : str
        The operator name, for the ``EigenbasisError`` message.

    Returns
    -------
    Symbol
        The retagging ``i k_hat`` diagonal on the coefficient factor.
    """
    bare = space.bare
    axis = _resolve_axis(op, bare)
    factor = bare.factor(axis)
    if in_trig_family(factor):
        # walled (F4): the FV staggering difference diagonalizes in
        # the sine/cosine basis. The average-origin cosine (pressure
        # gradient) and nodal-face sine (flux divergence) pairs cross
        # families through ``op.codomain`` (``fv_trig_diff_codomain``),
        # giving the real +-2 sin(k dz/2)/dz derived shift — the same
        # magnitude as the periodic ``i k_hat``, with no sinc (the
        # 2nd-order FV stencil is bitwise the nodal one).
        coeff, _origin = trig_partner(factor, who)
        return trig_finite_difference_symbol(
            bare, axis, coeff, op.codomain(coeff))
    src, origin = fv_fourier_partner(factor, who)
    return finite_difference_symbol(
        bare, axis, src, origin, op.codomain(origin))


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
    ``eigenvalues`` is the retagging ``i k_hat`` diagonal
    (``i k sinc(k dx / 2)`` with the ``Right``-origin phase),
    diagonalizing ``Fourier(Right) -> Fourier(CellAvg)`` on a
    periodic uniform mesh — bitwise the nodal ``Right -> Center``
    numbers; bounded and mapped meshes raise ``EigenbasisError``.
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
        if isinstance(domain, FourierSpace):
            # layout-faithful eigenvalue threading: retag the Fourier
            # factor through the staggered average origin
            return domain.mesh.fourier(
                origin=self.codomain(domain.origin))
        if isinstance(domain, SineSpace | CosineSpace):
            # walled trig coefficient (F4): the Dirichlet Inner sine of
            # the FV flux divergence pairs with the Neumann CellAvg
            # cosine (average) partner
            return fv_trig_diff_codomain(domain)
        if (isinstance(domain, NodalSpace)
                and domain.node_set is NodeSet.INNER
                and not domain.bc.is_free):
            # walled velocity (F4): a Dirichlet-tagged interior face
            # claims the homogeneous zero wall value -- exactly the
            # exact-zero boundary flux the Inner branch imposes -- so
            # the divergence closes at the walls. A Neumann tag claims
            # no wall value and cannot close (a taught error).
            if not all(c is BC.DIRICHLET for c in domain.bc.components):
                raise SpaceMismatchError(
                    f"no flux_diff signature on {domain!r}: a Neumann "
                    "tag on the interior faces claims no wall value, so "
                    "the flux divergence cannot close at the walls; the "
                    "walled FV divergence reads a Dirichlet "
                    "(no-normal-flow) face", left=domain,
                    operation="flux_diff")
            return _mesh_space(domain, "cell_avg", "flux_diff")
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

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — the factor carries the mesh
        space: SpaceLike,
    ) -> Symbol:
        r"""
        Return the ``i k_hat`` ``Right -> CellAvg`` diagonal.

        Description
        -----------
        The retagging ``Fourier(Right) -> Fourier(CellAvg)`` diagonal
        ``i k_hat = 2i sin(k dx/2)/dx`` (``= i k sinc(k dx/2)``)
        composed with the ``Right``-origin half-cell phase — the
        exact symbol of the periodic ``(u_i - u_{i-1}) / dx``
        divergence, bitwise the nodal ``Right -> Center`` numbers on a
        ``CellAvg`` codomain tag. Bounded (``Outer``/``Inner``) and
        mapped meshes carry no diagonalizing basis and raise
        ``EigenbasisError``.

        Parameters
        ----------
        grid : object
            The grid (unused: the Fourier factor carries the mesh).
        space : SpaceLike
            The coefficient factor (or product) space.

        Returns
        -------
        Symbol
            The retagging ``i k_hat`` diagonal on the coefficient
            factor.
        """
        return _fv_diff_eigenvalues(self, space, "FluxDifference")

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Difference the face fluxes into cell averages.

        Description
        -----------
        ``Outer``/``Right`` domains run the aligned window kernel
        (the periodic wrap ghost supplies the left face). The
        homogeneous ``Inner`` domain imposes exact zeros at both
        boundary faces (the no-normal-flow contract, never read from
        the BC-free extrapolation ghost) before differencing, via one
        of two byte-for-byte equivalent spellings gated by
        :func:`wall_slots_addressable`:

        - the storage-frame windowed fast path
          (:meth:`_inner_diff_windowed`): impose the zero wall flux in
          the storage ghost slots, then run the ordinary window (like
          ``Outer``). It keeps the operand's periodic-axis
          halo-validity claims;
        - the true-frame fallback (:meth:`_inner_diff_true_frame`):
          unpad, ``jnp.pad`` the zero fluxes, difference, and
          ``store``. It works on any layout but drops every axis's
          halo claim.

        The fast path exists because that true-frame excursion is what
        the multi-device step pays for: the SPMD partitioner
        materializes the unpad -> pad -> store tensors in a transposed
        layout and reroutes the periodic-axis halo collective-permutes
        through it, opening the FV-vs-nodal step gap the storage-frame
        spelling closes (``design/research/fv_nodal_step_gap.md``).

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
        factor = f.function_space.bare.factor(axis)
        if factor.node_set is not NodeSet.INNER:
            return _windowed_diff(self, f, axis)
        if wall_slots_addressable(f, axis):
            return self._inner_diff_windowed(f, axis)
        return self._inner_diff_true_frame(f, axis)

    def _inner_diff_windowed(
        self, f: FieldLike, axis: str,
    ) -> FieldLike:
        """
        Storage-frame windowed homogeneous ``Inner -> CellAvg`` divergence.

        Description
        -----------
        Impose the zero wall flux in the two wall ghost slots of the
        operand storage (:func:`wall_zeroed_operand`), then run the
        ordinary :func:`_windowed_diff` window (``m0 = 1``, reach 1).
        On a mapped mesh the unit-spacing difference divides by the
        codomain measure field (storage-frame, VJP-sealed). Bitwise
        the true-frame spelling on every output cell, but the result
        keeps the operand's periodic-axis halo-validity claims, like
        the nodal staggering path — the reason it exists (see
        :meth:`_apply_factor`).

        Parameters
        ----------
        f : FieldLike
            The operand flux field on an ``Inner`` factor with ``axis``
            device-local and halo >= 1.
        axis : str
            The resolved (bounded) coordinate axis.

        Returns
        -------
        FieldLike
            The flux-difference field (default metadata).
        """
        factor = f.function_space.bare.factor(axis)
        operand = wall_zeroed_operand(f, axis, factor.shape[0])
        return _windowed_diff(self, f, axis, operand=operand)

    def _inner_diff_true_frame(
        self, f: FieldLike, axis: str,
    ) -> FieldLike:
        """
        True-frame homogeneous ``Inner -> CellAvg`` divergence (any layout).

        Description
        -----------
        The layout-agnostic fallback of :meth:`_apply_factor` (a
        distributed walled axis or an un-negotiated halo): unpad the
        interior fluxes, ``jnp.pad`` an exact zero at each wall, and
        difference; a mapped mesh then divides by the codomain's primal
        cell width (rules 2.7/3.9, true-shape) before the ``store``
        routing. Correct on any layout, but the ``store``-built result
        claims zero halo validity on every axis.

        Parameters
        ----------
        f : FieldLike
            The operand flux field on an ``Inner`` factor.
        axis : str
            The resolved (bounded) coordinate axis.

        Returns
        -------
        FieldLike
            The flux-difference field (default metadata).
        """
        bare = f.function_space.bare
        factor = bare.factor(axis)
        codomain = factor_codomain(self, f.function_space, axis)
        axis_index = bare.names.index(axis)
        mapped = mapped_factor(factor)
        spacing = 1.0 if mapped else uniform_spacing(factor)
        pads = [(0, 0)] * len(bare.shape)
        pads[axis_index] = (1, 1)
        flux = jnp.pad(f.data, pads)
        data = staggered_diff(flux, axis_index, spacing=spacing,
                              order=_DIFF_SIZE)
        if mapped:
            # true-shape division by the codomain's primal cell
            # width (rules 2.7/3.9), before the storage routing
            query = codomain.with_layout(f.function_space.layout)
            data = data / f.grid.measure(query, name=axis).data
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
    ``eigenvalues`` is the retagging ``i k_hat`` diagonal
    (``i k sinc(k w / 2)`` on the dual mesh, ``w = dx`` uniform),
    diagonalizing ``Fourier(Center) -> Fourier(FaceAvg)`` on a
    periodic uniform mesh — bitwise the nodal ``Center -> Right``
    numbers; bounded and mapped meshes raise ``EigenbasisError``.
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
        if isinstance(domain, FourierSpace):
            # layout-faithful eigenvalue threading: retag the Fourier
            # factor through the dual-cell average origin
            return domain.mesh.fourier(
                origin=self.codomain(domain.origin))
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

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — the factor carries the mesh
        space: SpaceLike,
    ) -> Symbol:
        r"""
        Return the ``i k_hat`` ``Center/CellAvg -> FaceAvg`` diagonal.

        Description
        -----------
        The retagging ``Fourier(Center) -> Fourier(FaceAvg)`` (and
        ``Fourier(CellAvg) -> Fourier(FaceAvg)``) diagonal ``i k_hat``
        composed with the dual-cell half-cell phase — the exact symbol
        of the periodic ``(q_{i+1} - q_i) / dx`` dual difference,
        bitwise the nodal ``Center -> Right`` numbers on a ``FaceAvg``
        codomain tag. Bounded and mapped meshes raise
        ``EigenbasisError``.

        Parameters
        ----------
        grid : object
            The grid (unused: the Fourier factor carries the mesh).
        space : SpaceLike
            The coefficient factor (or product) space.

        Returns
        -------
        Symbol
            The retagging ``i k_hat`` diagonal on the coefficient
            factor.
        """
        return _fv_diff_eigenvalues(self, space, "DualFluxDifference")

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
    composition (``CellAvg -> CellAvg``). ``eigenvalues`` is the
    retagging ``i k_hat`` diagonal (``2i sin(k dx/2)/dx`` with the
    ``CellAvg``-origin phase), diagonalizing
    ``Fourier(CellAvg) -> Fourier(Right)`` on a periodic uniform mesh
    — bitwise the nodal ``Center -> Right`` numbers; composed with
    ``FluxDifference`` it forms the real ``-k_hat^2`` FV pressure
    Laplacian. Bounded and mapped meshes raise ``EigenbasisError``.
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
        if isinstance(domain, FourierSpace):
            # layout-faithful eigenvalue threading: retag the Fourier
            # factor through the staggered face origin
            return domain.mesh.fourier(
                origin=self.codomain(domain.origin))
        if isinstance(domain, SineSpace | CosineSpace):
            # walled trig coefficient (F4): the Neumann CellAvg cosine
            # of the FV pressure gradient pairs with the Dirichlet
            # Inner sine (nodal-face) partner
            return fv_trig_diff_codomain(domain)
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

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — the factor carries the mesh
        space: SpaceLike,
    ) -> Symbol:
        r"""
        Return the ``i k_hat`` ``CellAvg -> Right`` diagonal.

        Description
        -----------
        The retagging ``Fourier(CellAvg) -> Fourier(Right)`` diagonal
        ``i k_hat`` composed with the ``CellAvg``-origin half-cell
        phase — the exact symbol of the periodic
        ``(p_{i+1} - p_i) / dx`` face gradient, bitwise the nodal
        ``Center -> Right`` numbers on a ``CellAvg`` domain tag.
        ``FluxDifference @ FaceDifference`` then composes to the real
        ``-k_hat^2`` FV pressure Laplacian (the phases cancel), which
        ``SpectralSolve`` inverts. Bounded and mapped meshes raise
        ``EigenbasisError``.

        Parameters
        ----------
        grid : object
            The grid (unused: the Fourier factor carries the mesh).
        space : SpaceLike
            The coefficient factor (or product) space.

        Returns
        -------
        Symbol
            The retagging ``i k_hat`` diagonal on the coefficient
            factor.
        """
        return _fv_diff_eigenvalues(self, space, "FaceDifference")

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
