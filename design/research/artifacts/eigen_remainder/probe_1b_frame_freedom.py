"""1b: frame-freedom evidence (single device, cuda).

Build the SAME single-device nh channel eigenbasis in two coefficient
frames -- the default (half spectrum on the last periodic axis z) and a
forced alternative (half on x, a non-last periodic axis) -- and show the
family projections of the same physical state agree to floating point.

The forced frame is reached by monkeypatching the engine's half-axis
pick to return "x"; this exercises the REAL engine (_probe_block rfftn
ordering), the REAL nh labeler (_constrained_column half-frame
reconstruction), and the REAL projection path (fourier_ops ordering) in
the alternative frame -- exactly the mechanism the prototype uses on a
multi-device grid, but on one device so both frames are directly
comparable.
"""
import numpy as np

import fridom.model.eigen_channel as ec
import fridom as fr  # noqa: F401  (import ordering / backend init)
import fridom.nonhydro2 as nh
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

COMPONENTS = ("u", "v", "w", "b")


def make_model():
    meshes = tuple(
        IntervalMesh(8, (0.0, 1.0 if name == "y" else 2 * np.pi),
                     periodic=(name != "y"), name=name)
        for name in ("x", "y", "z"))
    return nh.Model(
        grid=Grid(meshes, device_ids=(0,)), advection=False, dsqr=2.0,
        coriolis=nh.FPlaneCoriolis(f0=1.5),
        stratification=nh.ConstantStratification(n2=3.0),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


def state(model, fields):
    model.set_fields(**fields)
    return nh.State({c: model.state[c] for c in COMPONENTS})


def absmax(a, b):
    return max(float(np.abs(np.asarray(a[c].data)
                            - np.asarray(b[c].data)).max())
               for c in COMPONENTS)


orig_pick = ec._designate_half_axis

model = make_model()
rng = np.random.default_rng(7)
fields = {c: rng.standard_normal(np.asarray(model.state[c].data).shape)
          for c in COMPONENTS}

# default frame: half = last periodic axis (z)
eb_z = nh.eigenbasis(model)
print("default periodic_axis  :", eb_z.periodic_axis)

# forced frame: half = x (a non-last periodic axis)
ec._designate_half_axis = lambda grid, periodic_names: "x"
try:
    eb_x = nh.eigenbasis(model)
finally:
    ec._designate_half_axis = orig_pick
print("forced  periodic_axis  :", eb_x.periodic_axis)
print("q shape (half=z)       :", tuple(np.asarray(eb_z.q).shape))
print("q shape (half=x)       :", tuple(np.asarray(eb_x.q).shape))
print("labels all labeled z/x :",
      bool((np.asarray(eb_z.labels) != -1).all()),
      bool((np.asarray(eb_x.labels) != -1).all()))

worst = 0.0
for sel in ("vortical", "wave", "kelvin"):
    pz = eb_z.projector(sel)(state(model, fields))
    px = eb_x.projector(sel)(state(model, fields))
    d = absmax(pz, px)
    worst = max(worst, d)
    real_x = not any(np.iscomplexobj(np.asarray(px[c].data))
                     for c in COMPONENTS)
    print(f"  projector({sel:8s})  |half=z - half=x| = {d:.3e}  "
          f"(half=x real: {real_x})")

# idempotency of the forced-frame projector
pv = eb_x.projector("vortical")
out = pv(state(model, fields))
idem = absmax(pv(out), out)
print(f"forced-frame vortical idempotency P(Pz)-Pz = {idem:.3e}")

# f(L) spectral function in both frames
fz = eb_z.function(lambda om: -1.0 / (1j * om), "wave")(
    state(model, fields))
fx = eb_x.function(lambda om: -1.0 / (1j * om), "wave")(
    state(model, fields))
print(f"function(inverse_l, wave) |half=z - half=x| = {absmax(fz, fx):.3e}")

print()
print("WORST projector frame difference:", f"{worst:.3e}",
      "-> PASS" if worst < 1e-12 else "-> FAIL")
