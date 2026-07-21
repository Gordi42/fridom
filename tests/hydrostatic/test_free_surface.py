"""The free surface: energy conservation and barotropic/baroclinic waves."""
import inspect

import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.spaces.constant import ConstantSpace

IM = fr.spatial.meshes.IntervalMesh


def make_grid(nx, nz, depth=1.0):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def _zextent(grid):
    lo, hi = next(m.extent for m in grid.factors if "z" in m.names)
    return float(hi - lo)


def make_model(grid, *, n2, csqr, f0, dt=1e-3):
    """Return a linear hydrostatic model (legacy csqr = g*H folded)."""
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=csqr / _zextent(grid)),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=hy.FPlaneCoriolis(f0=f0),
        stratification=hy.ConstantStratification(n2=n2),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)


def k_disc_sq(n_mode, n_cells, length=1.0):
    """Return the squared discrete wavenumber of the C-grid difference."""
    dx = length / n_cells
    k = 2.0 * np.pi * n_mode / length
    return (2.0 * np.sin(k * dx / 2.0) / dx) ** 2


# ================================================================
#  Constructor validation (the horizontal geometry names)
# ================================================================
@pytest.mark.parametrize(
    "horizontal",
    [
        pytest.param(("x",), id="one-name"),
        pytest.param(("x", "x"), id="duplicate"),
        pytest.param(("x", "y", "z"), id="three-names"),
    ],
)
def test_free_surface_rejects_a_bad_horizontal(horizontal):
    with pytest.raises(TypeError, match="horizontal"):
        hy.ExplicitFreeSurface(horizontal=horizontal)


# ================================================================
#  Machine-exact linear energy conservation (the energy gate)
# ================================================================
def test_linear_energy_is_conserved_to_roundoff():
    n2, csqr, f0 = 2.0, 3.0, 1.3
    model = make_model(make_grid(8, 6, depth=2.0),
                       n2=n2, csqr=csqr, f0=f0)
    rng = np.random.default_rng(3)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        b=rng.standard_normal(model.state["b"].shape),
        ps=rng.standard_normal(model.state["ps"].shape))

    state = model.state
    dX = model.tendency(state)
    p3 = state["b"].function_space  # the collocated 3D volume

    def integral(field):
        return float(field.integrate().data.ravel()[0])

    # ps is weighted 1/c^2 integrated over the full 3D volume
    terms = [
        integral(state["u"] * dX["u"]),
        integral(state["v"] * dX["v"]),
        integral((state["b"] / n2) * dX["b"]),
        integral((state["ps"].to(p3) / csqr) * dX["ps"].to(p3)),
    ]
    skew = sum(terms)
    scale = sum(abs(t) for t in terms)
    # <X, M dX/dt> vanishes to machine round-off (skew-adjoint L)
    assert abs(skew) < 1e-12 * scale


# ================================================================
#  Barotropic Poincaré dispersion (n2 = 0)
# ================================================================
def _mode_field(model, name, kx, ky, m, phase, depth):
    """Return a single Fourier mode (cos(m pi z / H) optional) at nodes."""
    fs = model.state[name].function_space
    var = tuple(n for f in fs.bare.factors
                if not isinstance(f, ConstantSpace) for n in f.names)
    trig = np.cos if phase == "cos" else np.sin

    def init(**c):
        val = trig(2 * np.pi * (kx * c["x"] + ky * c["y"]))
        if m is not None and "z" in c:
            val = val * np.cos(m * np.pi * c["z"] / depth)
        return val + 0.0 * sum(c.values())
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in var])
    return model.grid.create_field(fs, init=init)


def _reduced_omega(model, fields, kx, ky, depth=1.0):
    """Max |Im eig| of the linear operator restricted to one mode."""
    basis = [(f, p) for f in fields for p in ("cos", "sin")]
    modes = {(f, p): _mode_field(model, f, kx, ky, None, p, depth)
             for f, p in basis}
    norms = {k: float((v.data * v.data).sum()) for k, v in modes.items()}
    n = len(basis)
    mat = np.zeros((n, n))
    for i, (f, p) in enumerate(basis):
        st = model.state
        for g in fields:
            zero = model.grid.create_field(
                model.state[g].function_space,
                data=0.0 * model.state[g].data)
            st = st.replace(**{g: zero})
        st = st.replace(**{f: modes[(f, p)]})
        dX = model.tendency(st)
        for j, (g, p2) in enumerate(basis):
            proj = float((np.asarray(dX[g].data)
                          * np.asarray(modes[(g, p2)].data)).sum())
            mat[j, i] = proj / norms[(g, p2)]
    return float(np.max(np.abs(np.linalg.eigvals(mat).imag)))


@pytest.mark.parametrize(
    ("kx", "ky"),
    [
        pytest.param(1, 0, id="k=(1,0)"),
        pytest.param(1, 1, id="k=(1,1)"),
        pytest.param(2, 1, id="k=(2,1)"),
    ],
)
def test_barotropic_poincare_dispersion(kx, ky):
    nx, f0, csqr = 16, 0.8, 4.0
    model = make_model(make_grid(nx, 4), n2=0.0, csqr=csqr, f0=f0)
    om = _reduced_omega(model, ["u", "v", "ps"], kx, ky)
    kd2 = k_disc_sq(kx, nx) + k_disc_sq(ky, nx)
    om_a = np.sqrt(f0**2 + csqr * kd2)
    assert abs(om - om_a) / om_a < 1e-3


# ================================================================
#  Hydrostatic internal-wave dispersion (the discrete relation)
# ================================================================
def _trig_field(model, name, kx, ky, phase):
    """cos/sin(kx x + ky y), depth-uniform, at name's nodes (numpy)."""
    fs = model.state[name].function_space
    var = tuple(n for f in fs.bare.factors
                if not isinstance(f, ConstantSpace) for n in f.names)
    trig = np.cos if phase == "cos" else np.sin

    def init(**c):
        return trig(2 * np.pi * (kx * c["x"] + ky * c["y"])) \
            + 0.0 * sum(c.values())
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in var])
    return np.asarray(model.grid.create_field(fs, init=init).data)


def _full_spectrum(model, kx, ky):
    """All eigenvalues of L restricted to the (kx, ky) horizontal mode."""
    fields3d = ["u", "v", "b"]
    nz = model.state["u"].data.shape[2]
    trig = {(f, p): _trig_field(model, f, kx, ky, p)
            for f in [*fields3d, "ps"] for p in ("cos", "sin")}
    tnorm = {k: float((v * v).sum()) for k, v in trig.items()}

    cols = [(f, j, p) for f in fields3d for j in range(nz)
            for p in ("cos", "sin")]
    cols += [("ps", None, p) for p in ("cos", "sin")]
    zero = {g: 0.0 * np.asarray(model.state[g].data)
            for g in [*fields3d, "ps"]}

    def make_state(col):
        f, j, p = col
        data = {g: model.grid.create_field(
            model.state[g].function_space, data=zero[g])
            for g in [*fields3d, "ps"]}
        arr = np.array(zero[f])
        if j is None:
            arr[:] = trig[(f, p)]
        else:
            arr[:, :, j] = trig[(f, p)][:, :, j]
        data[f] = model.grid.create_field(
            model.state[f].function_space, data=arr)
        return model.state.replace(**data)

    n = len(cols)
    mat = np.zeros((n, n))
    for i, col in enumerate(cols):
        dX = model.tendency(make_state(col))
        for r, (g, jr, pr) in enumerate(cols):
            d = np.asarray(dX[g].data)
            if jr is None:
                proj = float((d * trig[(g, pr)]).sum()) / tnorm[(g, pr)]
            else:
                proj = float((d[:, :, jr] * trig[(g, pr)][:, :, jr]).sum())
                proj /= float((trig[(g, pr)][:, :, jr] ** 2).sum())
            mat[r, i] = proj
    return np.linalg.eigvals(mat)


def _gravest_baroclinic_m_disc_sq(model, kx, n2, f0, csqr, nx):
    """Return the gravest baroclinic mode discrete vertical wavenumber."""
    eig = _full_spectrum(model, kx, 0)
    omega = np.sort(np.unique(np.round(np.abs(eig.imag), 8)))
    omega = omega[omega > f0 + 1e-6]         # above the inertial band
    kh = np.sqrt(k_disc_sq(kx, nx))
    baro = np.sqrt(csqr) * kh                # the stiff barotropic branch
    bc = omega[omega < 0.5 * baro]           # the baroclinic band
    om1 = bc.max()                           # gravest baroclinic: m = 1
    kh2 = k_disc_sq(kx, nx)
    return n2 * kh2 / (om1**2 - f0**2)


def test_internal_wave_discrete_vertical_wavenumber():
    nx, nz, depth = 8, 16, 1.0
    n2, f0, csqr = 4.0, 0.3, 400.0  # stiff barotropic separates the bands
    model = make_model(make_grid(nx, nz, depth=depth),
                       n2=n2, csqr=csqr, f0=f0)
    m1 = _gravest_baroclinic_m_disc_sq(model, 1, n2, f0, csqr, nx)
    m2 = _gravest_baroclinic_m_disc_sq(model, 2, n2, f0, csqr, nx)

    # the discrete hydrostatic relation w^2 = f^2 + N^2 kh^2 / m_disc^2
    # yields the SAME m_disc across the two horizontal wavenumbers
    assert abs(m1 - m2) / m1 < 1e-3
    # and m_disc^2 approaches the continuous (pi / Lz)^2 to the
    # documented vertical-discretization tolerance
    m_cont2 = (np.pi / depth) ** 2
    assert abs(m1 - m_cont2) / m_cont2 < 2e-2
