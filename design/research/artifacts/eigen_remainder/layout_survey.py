"""Layout survey: which axis does the default decomposition shard?"""
import numpy as np
import jax
import fridom as fr  # noqa: F401
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

print("n_devices =", len(jax.devices()), flush=True)


def mk(spec):
    """spec: list of (n, periodic, name)."""
    meshes = tuple(
        IntervalMesh(n, (0.0, 1.0), periodic=p, name=nm)
        for (n, p, nm) in spec)
    return Grid(meshes)


def classify(grid):
    names = tuple(grid.names)
    factors = tuple(grid.factors)
    periodic = tuple(getattr(f, "periodic", None) is True for f in factors)
    walled = tuple(names[i] for i in range(len(names)) if not periodic[i])
    per_names = tuple(names[i] for i in range(len(names)) if periodic[i])
    half = per_names[-1] if per_names else None  # last periodic axis
    dev_axes = grid.decomposition.default_layout.device_axes
    if not dev_axes:
        cls = "UNSHARDED"
        sharded = None
    else:
        sharded = dev_axes[0][0]
        if sharded in walled:
            cls = "BOUNDED-SHARDED"
        elif sharded == half and len(per_names) >= 2:
            cls = "HALF-AXIS-SHARDED"
        elif len(per_names) == 1 and sharded in per_names:
            cls = "2D-PERIODIC-SHARDED"
        elif sharded in per_names:
            cls = "SERVED"
        else:
            cls = "?"
    n_mesh = len(grid.decomposition.device_mesh.axis_names)
    return dict(names=names, walled=walled, half=half, sharded=sharded,
                dev_axes=dev_axes, cls=cls, n_mesh_axes=n_mesh,
                cells={nm: getattr(f, "n_cells", None)
                       for nm, f in zip(names, factors)})


CASES = [
    # label, spec
    ("3D chan wall-z cubic 32^3",
     [(32, True, "x"), (32, True, "y"), (32, False, "z")]),
    ("3D chan wall-x cubic 32^3",
     [(32, False, "x"), (32, True, "y"), (32, True, "z")]),
    ("3D chan wall-y cubic 32^3",
     [(32, True, "x"), (32, False, "y"), (32, True, "z")]),
    ("3D chan wall-z aniso 32x64x128 (half=y largest? no, half=y)",
     [(32, True, "x"), (64, True, "y"), (128, False, "z")]),
    ("3D chan wall-z aniso half-axis largest: x=32 y=128 (half=y=128)",
     [(32, True, "x"), (128, True, "y"), (16, False, "z")]),
    ("3D chan wall-z, x INDIVISIBLE (30) y div (32) -> forces half=y",
     [(30, True, "x"), (32, True, "y"), (32, False, "z")]),
    ("3D chan wall-z, x indiv(30) y indiv(30) z div walled(32)",
     [(30, True, "x"), (30, True, "y"), (32, False, "z")]),
    ("2D chan wall-y (x periodic, y walled) 32x32",
     [(32, True, "x"), (32, False, "y")]),
    ("2D chan wall-x (x walled, y periodic) 32x32",
     [(32, False, "x"), (32, True, "y")]),
    ("3D fully periodic 32^3 (not a channel)",
     [(32, True, "x"), (32, True, "y"), (32, True, "z")]),
    ("3D chan wall-z, x=32 y=30(indiv) -> both periodic, y indiv",
     [(32, True, "x"), (30, True, "y"), (32, False, "z")]),
]

fmt = "{:<52} names={} walled={} half={} sharded={} mesh_ax={} -> {}"
for label, spec in CASES:
    try:
        g = mk(spec)
        r = classify(g)
        print(fmt.format(
            label, r["names"], r["walled"], r["half"], r["sharded"],
            r["n_mesh_axes"], r["cls"]), flush=True)
    except Exception as e:  # noqa: BLE001
        print("{:<52} RAISED: {}: {}".format(
            label, type(e).__name__, str(e)[:120]), flush=True)
