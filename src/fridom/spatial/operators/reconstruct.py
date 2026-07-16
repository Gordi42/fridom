"""
``LinearReconstruction`` / ``LinearDeconvolution``: average <-> point.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_stencils.md``.
Second-order conversions inside the average family. Two members:

- ``LinearReconstruction`` — the *staggering* conversion (the default
  ``("reconstruct", ...)`` entry, also seeded under ``("average", ...)``
  for the nodal -> average direction that ``f.to`` resolves):
  ``CellAvg -> Right`` and its evaluate-to-average inverse, half a cell
  over. At second order both directions (average-to-point Shu mean and
  evaluate-to-average trapezoid mean) collapse to two-point means, but
  they are distinct signatures; higher-order members
  (``WenoReconstruction``, Wave 4) genuinely differ per direction.
- ``LinearDeconvolution`` — the *co-located* conversion (the
  ``("deconvolve", ...)`` entry): ``CellAvg <-> Center`` at the same
  location. At second order this is the identity (the cell mean and the
  midpoint value differ only at ``O(dx^2)``), so it is a one-point
  pass-through retag — a distinct dispatch kind because a registry key
  resolves one codomain and ``"reconstruct"`` is committed to the
  staggering ``Right`` target (rules 3.4, fields.md "one target per
  kind").

This module also owns the FV generalization of the staggered window
alignment (``fv_node_offset`` / ``apply_fv_staggered``): the Wave-2C
``operators.staggering`` module grounds nodal factors only and is
read-only for this cluster, so the average-family offsets (``CellAvg``
at the primal-cell midpoints, ``FaceAvg`` at the dual-cell midpoints)
live here, with the identical alignment calculus.
"""
# Wave 3: LinearReconstruction, FV window alignment --
#    Wave 4: WenoReconstruction
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.base import (
    EigenbasisError,
    FieldLike,
    Operator,
    OperatorRequirements,
    SeparableOperator,
    _resolve_axis,
)
from fridom.spatial.operators.interned import interned
from fridom.spatial.operators.spectral import (
    fv_fourier_partner,
    linear_interp_symbol,
)
from fridom.spatial.operators.staggering import (
    first_node_offset,
)
from fridom.spatial.operators.stencil_kernels import (
    apply_stencil,
    linear_interp,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import CellAvg, FaceAvg
from fridom.spatial.spaces.coefficient import FourierSpace
from fridom.spatial.spaces.function_space import FunctionSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from jax import Array

    from fridom.spatial.fields.metadata import FieldMetadata
    from fridom.spatial.operators.symbol import Symbol
    from fridom.spatial.spaces.tensor_product import SpaceLike

_RECON_SIZE = 2
_DECONV_SIZE = 1


def factor_codomain(
    op: Operator, space: SpaceLike, axis: str,
) -> SpaceLike:
    """
    Lift a per-factor codomain over the product at ``axis``.

    Description
    -----------
    Chain-safe sibling of ``base.resolve_codomain``: it takes the
    axis as an argument instead of re-resolving it from the
    operator's ``bound_axis``, so it stays unambiguous when a
    ``SeparableComposite`` chain applies its (stored-unbound)
    factors on a multi-axis operand.

    Parameters
    ----------
    op : Operator
        The separable kernel (bound or unbound).
    space : SpaceLike
        The operand's (possibly laid-out) space.
    axis : str
        The resolved coordinate axis.

    Returns
    -------
    SpaceLike
        The bare codomain space.
    """
    bare = space.bare
    new_factor = op.codomain(bare.factor(axis))
    if isinstance(bare, FunctionSpace):
        return new_factor
    return bare.replace(**{axis: new_factor})


# ================================================================
#  FV window alignment (average-family generalization of
#  operators.staggering, which is nodal-only and read-only here)
# ================================================================
def fv_node_offset(factor: FunctionSpace) -> float:
    """
    First-node offset in cell widths, average family included.

    Description
    -----------
    For window alignment an average DOF is treated as *positioned* at
    its quadrature point: ``CellAvg`` at the primal-cell midpoints
    (offset 0.5, like ``Center``), ``FaceAvg`` at the dual-cell
    midpoints — the faces (offset 1.0, like ``Right``/``Inner``, on
    periodic and bounded meshes alike). Nodal factors delegate to
    ``staggering.first_node_offset``.

    Parameters
    ----------
    factor : FunctionSpace
        A bare 1D factor space.

    Returns
    -------
    float
        Distance of the first true DOF from ``x_min`` in units of
        the cell width.
    """
    if isinstance(factor, CellAvg):
        return 0.5
    if isinstance(factor, FaceAvg):
        return 1.0
    return first_node_offset(factor)


def apply_fv_staggered(
    op: Operator,
    f: FieldLike,
    axis: str,
    size: int,
    kernel: Callable[[Array, int], Array],
    metadata: FieldMetadata | None,
    align: int | None = None,
) -> FieldLike:
    """
    Run an aligned ``size``-point kernel along ``axis`` (FV family).

    Description
    -----------
    The ``staggering.apply_staggered`` body with ``fv_node_offset``
    in place of the nodal-only ``first_node_offset`` (that module is
    read-only for this cluster): resolves the codomain, aligns the
    kernel over the halo-extended storage, and builds the result on
    the bare codomain via the plumbing constructor. Edge slots the
    kernel cannot compute are zero-filled and repaired by the
    post-application sync.

    The default alignment is the window midpoint (the even-size
    staggered family); a **biased** kernel whose output does not sit
    on the midpoint (odd-size WENO) passes its window cell index
    explicitly via ``align``.

    Parameters
    ----------
    op : Operator
        The (bound) separable kernel being applied.
    f : FieldLike
        The operand field (storage-shaped ``_data``).
    axis : str
        The resolved coordinate axis.
    size : int
        The stencil size (number of input points per output).
    kernel : Callable[[Array, int], Array]
        Array kernel mapping (storage, axis index) to the full
        stencil output (length shrinks by ``size - 1``).
    metadata : FieldMetadata | None
        Metadata of the result (None resets to the default record).
    align : int | None, optional
        Explicit window alignment ``m0``: kernel output ``t`` fills
        output slot ``t + m0``. None derives it from the midpoint
        staggering calculus (default: None).

    Returns
    -------
    FieldLike
        The result field on the bare codomain.
    """
    space = f.function_space
    bare = space.bare
    domain_factor = bare.factor(axis)
    codomain = factor_codomain(op, space, axis)
    codomain_factor = codomain.factor(axis)
    if align is None:
        delta = (fv_node_offset(codomain_factor)
                 - fv_node_offset(domain_factor))
        i0 = delta - (size - 1) / 2
        if i0 != int(i0):  # pragma: no cover — even-size family only
            raise ValueError(
                f"misaligned staggering: offset {delta} with stencil "
                f"size {size} does not land on the lattice")
        m0 = -int(i0)
    else:
        m0 = align

    axis_index = bare.names.index(axis)
    storage = f._data  # noqa: SLF001 — documented storage seam
    decomposition = f.grid.decomposition
    # the codomain storage lives in the operand's layout (the base
    # re-attaches it); resolving the bare codomain would pick the
    # default layout instead of the operand's pencil
    out_shape = decomposition.storage_shape(codomain, space.layout)
    s_out = out_shape[axis_index]
    n_out = codomain_factor.shape[0]
    try:
        width = decomposition.halo[axis]
    except KeyError:
        width = 0

    # per-side stencil reach beyond the true region; the halo must
    # cover it. Frame-independent by construction (never read off
    # the storage extents): on a blocked multi-shard frame the
    # storage-bounds check degenerates (the concatenated blocks are
    # longer than one true axis) and under-negotiated halos would
    # silently read the zeroed stagger slots at block edges.
    reach_right = (n_out - domain_factor.shape[0]) + size - 1 - m0
    if m0 > width or reach_right > width:
        raise ValueError(
            f"the negotiated halo width {width} along {axis!r} is "
            f"too small for the {size}-point stencil of "
            f"{type(op).__name__}; renegotiate with a registry that "
            "declares the wider requirement")
    full = kernel(storage, axis_index)
    length = full.shape[axis_index]
    lo = max(0, m0)
    hi = min(s_out, m0 + length)
    index: list[slice] = [slice(None)] * full.ndim
    index[axis_index] = slice(lo - m0, hi - m0)
    piece = full[tuple(index)]
    pads = [(0, 0)] * full.ndim
    pads[axis_index] = (lo, s_out - hi)
    data = jnp.pad(piece, pads)
    # halo-validity claim (task 1.8, stage B): the kernel computed
    # every output ghost slot its window reaches, so on a *periodic*
    # axis the result keeps the operand's valid layers minus the
    # per-side maximum reach (stencils commute with the wrap fill).
    # On bounded axes the claim is zero: stenciling the input's
    # BC-structured/extrapolated fill is not the BC-consistent fill
    # of the *output* field, so those ghost slots must be refilled
    # at the next consumption.
    if getattr(domain_factor.mesh, "periodic", False):
        valid = f.halo_valid.consume(axis, max(m0, reach_right, 0))
    else:
        valid = f.halo_valid.consume(axis, f.halo_valid[axis])
    return type(f)(f.grid, codomain, data, metadata,
                   halo_valid=valid)


# ================================================================
#  LinearReconstruction
# ================================================================
@final
@interned
class LinearReconstruction(SeparableOperator):

    """
    2nd-order average <-> point-value conversion (FV family).

    Description
    -----------
    Fixed codomain per rules section 3.4: the registered operator
    fixes its own codomain; alternative codomains are per-instance
    via the ``target=`` constructor knob. The bounded ``CellAvg ->
    Outer`` variant is un-grounded under the R1 legality rule
    (boundary_plan.md): its wall faces need exterior values, which
    the (always BC-free) average spaces do not define — a one-sided
    reconstruction variant is designed-for and arrives when a
    concrete model needs it. ``eigenvalues`` is the retagging
    ``one_hat`` averaging diagonal ``cos(k dx/2)`` (with the
    inter-origin phase), diagonalizing ``Fourier(CellAvg) ->
    Fourier(Right)`` — and the other two-point conversions — on a
    periodic uniform mesh; at second order the deconvolution ``sinc``
    correction vanishes, so it is bitwise the nodal ``LinearInterp``
    numbers. The ``target=`` variant, bounded, and mapped meshes
    raise ``EigenbasisError``.

    Parameters
    ----------
    target : NodeSet | None, optional
        Explicit target node set overriding the default table;
        iteration 1 grounds ``NodeSet.OUTER`` only (default: None).
    """

    dispatch_kind: ClassVar[str | None] = "reconstruct"

    def __init__(self, target: NodeSet | None = None) -> None:
        """Create the kernel; ``target`` overrides the codomain."""
        if target is not None and not isinstance(target, NodeSet):
            raise TypeError(
                f"target must be a NodeSet member or None, got "
                f"{target!r}")
        self._target: NodeSet | None = target

    def _intern_key(self) -> tuple:
        """Structural key: the explicit target node set (D6)."""
        return (self._target,)

    @property
    def target(self) -> NodeSet | None:
        """Explicit target node set, or None for the default table."""
        return self._target

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Resolve the reconstruction codomain.

        Description
        -----------
        reconstruct: CellAvg -> Right (periodic) | Inner (bounded);
        Right/Outer/Inner -> CellAvg; FaceAvg <-> Center dual;
        target= selects the bounded CellAvg -> Outer variant.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D factor space (average or BC-free nodal).

        Returns
        -------
        FunctionSpace
            The converted codomain factor (scalars preserved).
        """
        if isinstance(domain, FourierSpace):
            # layout-faithful eigenvalue threading: retag the Fourier
            # factor through the reconstructed average/nodal origin
            return domain.mesh.fourier(
                origin=self.codomain(domain.origin))
        mesh = domain.mesh
        if self._target is not None:
            if (self._target is NodeSet.OUTER and not mesh.periodic
                    and isinstance(domain, CellAvg)):
                # R1 (boundary_plan.md 2c): the Outer wall faces
                # need exterior values, which the (always BC-free)
                # CellAvg does not define; a one-sided
                # reconstruction variant is designed-for
                raise SpaceMismatchError(
                    f"no reconstruct signature on {domain!r}: the "
                    "CellAvg -> Outer wall faces need exterior "
                    "values, which a BC-free bounded space does "
                    "not define (R1, boundary_plan.md); a "
                    "one-sided reconstruction variant arrives when "
                    "a concrete model needs it",
                    left=domain, operation="reconstruct")
            raise SpaceMismatchError(
                "the target= variant grounds bounded "
                "CellAvg -> Outer only in iteration 1; got "
                f"target={self._target} on {domain!r}",
                left=domain, operation="reconstruct")
        if isinstance(domain, CellAvg):
            result = "right" if mesh.periodic else "inner"
        elif isinstance(domain, FaceAvg):
            result = "center"
        elif isinstance(domain, NodalSpace) and domain.bc.is_free:
            result = {NodeSet.RIGHT: "cell_avg",
                      NodeSet.OUTER: "cell_avg",
                      NodeSet.INNER: "cell_avg",
                      NodeSet.CENTER: "face_avg"}.get(domain.node_set)
            if result is None:
                raise SpaceMismatchError(
                    f"no reconstruct signature on {domain!r}: "
                    "Right/Outer/Inner -> CellAvg, Center -> FaceAvg "
                    "(nodal -> nodal conversions are the "
                    "'interpolate' kind)", left=domain,
                    operation="reconstruct")
        else:
            raise SpaceMismatchError(
                "LinearReconstruction covers the average family and "
                f"BC-free nodal spaces, got {domain!r}",
                left=domain, operation="reconstruct")
        try:
            codomain: FunctionSpace = getattr(mesh, result)
        except ValueError as exc:  # mesh lacks the codomain family
            raise SpaceMismatchError(
                f"no reconstruct signature on {domain!r}: {mesh!r} "
                f"has no {result} space", left=domain,
                operation="reconstruct") from exc
        if domain.scalars is Scalars.COMPLEX:
            codomain = codomain.as_complex()
        return codomain

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
        Return the ``one_hat`` averaging diagonal ``cos(k dx/2)``.

        Description
        -----------
        The retagging ``Fourier(CellAvg) -> Fourier(Right)`` (and the
        other two-point conversions the codomain table grounds)
        diagonal ``cos(k dx/2)`` composed with the inter-origin phase
        — the exact symbol of the periodic ``(p_i + p_{i+1}) / 2``
        two-point mean, bitwise the nodal ``LinearInterp`` numbers.
        At second order the exact cell-average deconvolution (which
        would *divide* by ``sinc(k dx/2)``) collapses to the plain
        average, so this row carries no ``sinc`` correction. The
        ``target=`` variant has no diagonalizing symbol, and bounded
        or mapped meshes raise ``EigenbasisError``.

        Parameters
        ----------
        grid : object
            The grid (unused: the Fourier factor carries the mesh).
        space : SpaceLike
            The coefficient factor (or product) space.

        Returns
        -------
        Symbol
            The retagging ``one_hat`` diagonal on the coefficient
            factor.
        """
        if self._target is not None:
            raise EigenbasisError(
                "the target= LinearReconstruction variant has no "
                "diagonalizing symbol in iteration 1")
        bare = space.bare
        axis = _resolve_axis(self, bare)
        factor = bare.factor(axis)
        src, origin = fv_fourier_partner(factor, "LinearReconstruction")
        return linear_interp_symbol(
            bare, axis, src, origin, self.codomain(origin))

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Convert along ``axis`` (window-aligned two-point mean).

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The converted field (metadata kept: same quantity).
        """
        return apply_fv_staggered(self, f, axis, _RECON_SIZE,
                                  linear_interp, metadata=f.metadata)


# ================================================================
#  LinearDeconvolution (co-located CellAvg <-> Center)
# ================================================================
def _identity_kernel(arr: Array, axis: int) -> Array:
    """
    One-point pass-through kernel (the co-located 2nd-order identity).

    Description
    -----------
    A degenerate ``size = 1`` stencil with the single weight ``1`` — the
    slice-based no-op that keeps ``apply_fv_staggered`` on its one
    kernel path (window alignment ``m0 = 0``, no axis shrink), so the
    co-located conversion reuses the same halo/frame accounting as the
    two-point family.

    Parameters
    ----------
    arr : Array
        The input array (halo-extended by the caller as needed).
    axis : int
        The stencil axis; may be negative.

    Returns
    -------
    Array
        ``arr`` unchanged (the axis length is preserved).
    """
    return apply_stencil(arr, axis, (1.0,))


@final
@interned
class LinearDeconvolution(SeparableOperator):

    """
    2nd-order co-located average <-> point-value conversion (FV).

    Description
    -----------
    The same-location member of the FV conversion family: it maps a
    cell average to the point value at the *same* location (the cell
    midpoint) and back — ``CellAvg <-> Center`` — unlike the staggering
    ``LinearReconstruction`` (``CellAvg -> Right``, half a cell over).
    At the family's standing second order the two functionals differ
    only at ``O(dx^2)`` (the cell mean is the midpoint value plus
    ``(dx^2/24) f''``), so the consistent conversion is the **identity**:
    a one-point pass-through that retags the data without moving it
    (spaces.md — "identifying it with the center value is a
    second-order approximation"; rules 3.9; operators_stencils.md's
    "declared ``O(dx^2)`` identification of cell averages with midpoint
    values"). The higher-order member (a genuine ``sinc``-deconvolution)
    is designed-for.

    A distinct dispatch kind (``"deconvolve"``) from ``"reconstruct"``:
    the reconstruct kind is committed to the staggering
    ``CellAvg -> Right/Inner`` (the FV derivative's reconstruction and
    the C-grid Coriolis ``u.to(v)``), and a registry key resolves one
    codomain (rules 3.4, fields.md "one target per kind"). ``f.to``
    reads the ``"deconvolve"`` kind for a co-located average<->nodal
    pair. Grounded on periodic and bounded axes alike: the co-located
    conversion needs no exterior values, so the R1 wall-face
    restriction (boundary_plan.md) does not bite. ``FaceAvg <-> Right``
    (the dual co-located pair) is designed-for and left ungrounded —
    FV-D2 option A never instantiates ``FaceAvg``. ``eigenvalues``
    (the reciprocal-``sinc`` deconvolution symbol) inherits the raising
    base until the ``Symbol`` cluster lands.
    """

    dispatch_kind: ClassVar[str | None] = "deconvolve"

    def _intern_key(self) -> tuple:
        """Structural key: the operator is parameterless (D6)."""
        return ()

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Resolve the co-located conversion codomain.

        Description
        -----------
        deconvolve: CellAvg -> Center, Center -> CellAvg (the
        same-location primal pair). FaceAvg and the dual pair are
        designed-for and raise (the staggering conversions are the
        ``"reconstruct"`` kind).

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D factor space (CellAvg or a BC-free Center).

        Returns
        -------
        FunctionSpace
            The co-located codomain factor (scalars preserved).
        """
        if isinstance(domain, CellAvg):
            result = "center"
        elif (isinstance(domain, NodalSpace) and domain.bc.is_free
                and domain.node_set is NodeSet.CENTER):
            result = "cell_avg"
        else:
            raise SpaceMismatchError(
                "the co-located deconvolution grounds CellAvg <-> "
                f"Center only, got {domain!r} (the staggering "
                "conversions are the 'reconstruct' kind)",
                left=domain, operation="deconvolve")
        # the co-located primal pair always coexists (both live only on
        # a StructuredMesh1D, which carries both), so — unlike the
        # staggering reconstruct, whose Outer domain reaches a mesh
        # without cell_avg — no missing-family guard is reachable here
        codomain: FunctionSpace = getattr(domain.mesh, result)
        if domain.scalars is Scalars.COMPLEX:
            codomain = codomain.as_complex()
        return codomain

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — halo-free identity
    ) -> OperatorRequirements:
        """
        Declare halo = 0, layout "any" (a pass-through identity).

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=0)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Convert along ``axis`` (co-located one-point identity).

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The retagged field (metadata kept: same quantity).
        """
        return apply_fv_staggered(self, f, axis, _DECONV_SIZE,
                                  _identity_kernel, metadata=f.metadata)
