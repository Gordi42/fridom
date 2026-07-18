"""1c: prototype gate on a grid that shards the LAST periodic axis.

Grid: walled-y, x periodic (indivisible cell count -> ranks after z,
stays LOCAL), z periodic (divisible by 4 -> rank 0, the DEFAULT sharded
axis). So the default layout shards z, the engine's default half axis.
The re-designation picks x (local) as the half axis, and the shipped
fused contraction serves the layout with a = z (full spectrum, sharded),
b = x (rfft half, local).

Gates: projection matches a device_ids=(0,) single-device reference
< 1e-11; idempotent < 1e-11; lowered HLO has all-to-all and no
all-gather; lands real.
"""
import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

COMPONENTS = ("u", "v", "w", "b")
NX, NY, NZ = 10, 8, 12  # x local (10%4!=0), z sharded (12%4==0)


def make_model(device_ids=None):
    meshes = tuple(
        IntervalMesh(n, (0.0, 1.0 if name == "y" else 2 * np.pi),
                     periodic=(name != "y"), name=name)
        for name, n in (("x", NX), ("y", NY), ("z", NZ)))
    return nh.Model(
        grid=Grid(meshes, device_ids=device_ids), advection=False,
        dsqr=2.0, coriolis=nh.FPlaneCoriolis(f0=1.5),
        stratification=nh.ConstantStratification(n2=3.0),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


def state(model, fields):
    model.set_fields(**fields)
    return nh.State({c: model.state[c] for c in COMPONENTS})


def absmax(a, b):
    return max(float(np.abs(np.asarray(a[c].data)
                            - np.asarray(b[c].data)).max())
               for c in COMPONENTS)


print("device_count:", jax.device_count())

many = make_model()
axes = many.grid.decomposition.default_layout.device_axes
print("default device_axes    :", axes)
assert axes == (("z", "devices"),), f"expected z sharded, got {axes}"

rng = np.random.default_rng(11)
fields = {c: rng.standard_normal(np.asarray(many.state[c].data).shape)
          for c in COMPONENTS}

eb_many = nh.eigenbasis(many)
print("re-designated half axis:", eb_many.periodic_axis, "(expect x)")
assert eb_many.periodic_axis == "x"
assert bool((np.asarray(eb_many.labels) != -1).all()), "unlabeled cols"

z_many = state(many, fields)
print("u sharding spec        :", z_many["u"]._data.sharding.spec)
assert z_many["u"]._data.sharding.spec[2] == "devices", "z must shard"

proj_many = eb_many.projector("vortical")
out_many = proj_many(z_many)
real = not any(np.iscomplexobj(np.asarray(out_many[c].data))
               for c in COMPONENTS)
print("lands real             :", real)

idem = absmax(proj_many(out_many), out_many)
print(f"idempotency P(Pz)-Pz   : {idem:.3e}  "
      f"({'PASS' if idem < 1e-11 else 'FAIL'})")

# single-device reference
one = make_model(device_ids=(0,))
eb_one = nh.eigenbasis(one)
print("one-device half axis   :", eb_one.periodic_axis, "(expect z)")
z_one = state(one, fields)
out_one = eb_one.projector("vortical")(z_one)
match = absmax(out_many, out_one)
print(f"many vs one-device     : {match:.3e}  "
      f"({'PASS' if match < 1e-11 else 'FAIL'})")

# HLO collectives check
try:
    text = jax.jit(proj_many).lower(z_many).compile().as_text()
except Exception as exc:  # noqa: BLE001
    print("jit(proj) lower failed, trying plan region:", exc)
    from fridom.spatial.operators.distributed_contract import (
        resolve_distributed_contraction)
    plan = resolve_distributed_contraction(
        many.grid, bounded_axis=eb_many.bounded_axis,
        periodic_axis=eb_many.periodic_axis,
        components=eb_many.components, slices=eb_many.slices)
    qf = plan._pad_modes(eb_many.q)
    wf = plan._pad_modes(jnp.where(
        jnp.isin(eb_many.labels, jnp.asarray([0])), 1.0, 0.0
    ).astype(qf.dtype))
    if plan.a_padded:
        dec = many.grid.decomposition
        pieces = {c: dec.unpad_even(z_many[c].storage,
                                    z_many[c].function_space)
                  for c in COMPONENTS}
    else:
        pieces = {c: jnp.asarray(z_many[c].data) for c in COMPONENTS}
    text = plan._region.lower(
        pieces, qf, wf, jnp.asarray(eb_many.metric)).compile().as_text()

has_a2a = "all-to-all" in text
has_ag = "all-gather" in text
print(f"HLO all-to-all present : {has_a2a}")
print(f"HLO all-gather present : {has_ag}")
print("HLO gate               :",
      "PASS" if (has_a2a and not has_ag) else "FAIL")

ok = (real and idem < 1e-11 and match < 1e-11
      and has_a2a and not has_ag
      and eb_many.periodic_axis == "x")
print()
print("1c OVERALL:", "PASS" if ok else "FAIL")
