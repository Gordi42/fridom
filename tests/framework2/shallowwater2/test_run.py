"""A small shallow-water run: stability, conservation, sanity."""
import jax
import numpy as np

from fridom.framework2.model.model import chunk_cache_size

from .conftest import gaussian_bump, make_grid, make_model, total_energy


# ================================================================
#  Treedef stability + single compile
# ================================================================
def test_carry_treedef_stable_across_advance():
    model = make_model()
    model.set_fields(h=gaussian_bump())
    before = jax.tree_util.tree_structure(model._carry)
    model.advance(5)
    after = jax.tree_util.tree_structure(model._carry)
    assert before == after


def test_repeated_advance_compiles_nothing(compile_counter):
    model = make_model()
    model.set_fields(h=gaussian_bump())
    model.advance(5)                       # warm every path
    compile_counter.reset()
    reference = chunk_cache_size()
    model.advance(5)
    assert compile_counter.count == 0
    assert chunk_cache_size() == reference


def test_split_advance_matches_one_run():
    grid = make_grid()
    whole = make_model(grid)
    split = make_model(grid)
    whole.set_fields(h=gaussian_bump())
    split.set_fields(h=gaussian_bump())
    whole.advance(6)
    split.advance(2)
    split.advance(4)
    assert np.array_equal(np.asarray(whole.state["h"].data),
                          np.asarray(split.state["h"].data))
    assert int(whole.clock.it) == int(split.clock.it)


# ================================================================
#  Physical sanity
# ================================================================
def test_mass_is_conserved_to_roundoff():
    model = make_model()
    model.set_fields(h=gaussian_bump())
    mass0 = float(model.state["h"].integrate().data.ravel()[0])
    model.advance(60)
    mass1 = float(model.state["h"].integrate().data.ravel()[0])
    # the flux-form thickness equation telescopes: mass to roundoff
    assert abs(mass1 - mass0) < 1e-12


def test_run_stays_finite_and_bounded():
    model = make_model()
    model.set_fields(h=gaussian_bump(amp=0.05))
    model.advance(80)
    h = np.asarray(model.state["h"].data)
    assert np.isfinite(h).all()
    # a gravity wave disperses the bump; the amplitude does not grow
    assert np.abs(h).max() < 0.05


def test_total_energy_is_nearly_conserved():
    model = make_model(dt=2e-3)
    model.set_fields(h=gaussian_bump(amp=0.05))
    e0 = total_energy(model)
    energies = []
    for _ in range(20):
        model.advance(5)
        energies.append(total_energy(model))
    energies = np.asarray(energies)
    # bounded drift over the run (AB3 + Sadourny: no secular growth)
    assert np.all(np.abs(energies - e0) / e0 < 5e-3)
