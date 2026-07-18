"""Memory-hygiene tests for the Model carry (model/model.py).

Covers the two GPU-memory-ceiling fixes:

- **Donating ``_canonicalize``** (``_prepare_for_donation`` +
  ``_canonicalize_donating``): the whole-carry copy donates its big
  buffers instead of transiently doubling them. The guard copies the
  leaves that cannot be donated as-is — the aliased ``AdamBashforth``
  history ring (``(zeros,) * (order - 1)``) and small caller-held
  parameter scalars — so donation never errors and never eats a
  user buffer, while staying bitwise value-preserving.
- **One-time pre-chunk defragmentation** (``_defragment_carry`` gated
  by ``_should_defragment`` / ``_defrag_enabled``): the first
  ``advance`` sequentially repacks the carry's big buffers so BFC can
  place the chunk's single contiguous temp arena; armed at
  construction and every ``_commit``, GPU-only, kill-switchable.

Self-contained (the shard convention): the small builders are
duplicated rather than imported across test files.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.model.declarations import FieldDeclaration
from fridom.model.model import (
    _DONATE_MIN_BYTES,
    Model,
    _canonicalize,
    _canonicalize_donating,
    _defrag_enabled,
    _prepare_for_donation,
)
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.terms import term
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

DT = 1e-3
N = 8


# ================================================================
#  Self-contained builders
# ================================================================
@partial(jaxify, dynamic=("nu",))
class Core(Module):

    """Two-field co-located PROGNOSTIC core with a scalar param."""

    def __init__(self, nu=1e-3):
        self.nu = jnp.asarray(nu, dtype=dtype_real())

    field_declarations = (
        FieldDeclaration("u", space=Collocated()),
        FieldDeclaration("b", space=Collocated()),
    )
    parameter_declarations = (
        ParameterDeclaration("core.nu", attr="nu", units="1"),)

    @term(advances=("u",))
    def force(self, state, _ctx):
        return {"u": state["b"]}

    @term(advances=("b",))
    def restore(self, state, _ctx):
        return {"b": state["u"] * (-1.0)}


def make_grid():
    return Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),))


def make_model(*, order=3, **kwargs):
    # order 3 -> a 2-slot history ring: (zeros,) * 2 aliases its
    # buffers (the donation hazard). This is also the nonhydro2
    # default order.
    return Model(grid=make_grid(), modules=(Core(),),
                 time_stepper=AdamBashforth(DT, order=order), **kwargs)


def ic(shift=0.0):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    return np.sin(2 * np.pi * x) + shift


# ================================================================
#  Donation — the aliased ring hazard and its guard
# ================================================================
def test_fresh_stepper_ring_is_aliased():
    # the premise: AdamBashforth.init shares one zeros buffer across
    # the ring slots, so a fresh order-3 carry holds aliased leaves
    model = make_model(order=3)
    leaves = jax.tree_util.tree_leaves(model._fresh_stepper_state())
    ids = [id(x) for x in leaves if isinstance(x, jax.Array)]
    assert len(ids) != len(set(ids))  # duplicates present


def test_construction_survives_aliased_ring():
    # the donating _canonicalize must not raise on the aliased fresh
    # ring; the guard copies the duplicate slot's buffer
    model = make_model(order=3)
    assert len(model.carry.stepper_state.history) == 2
    # the committed carry is de-aliased (each leaf a distinct buffer)
    leaves = jax.tree_util.tree_leaves(model._carry)
    ids = [id(x) for x in leaves if isinstance(x, jax.Array)]
    assert len(ids) == len(set(ids))


def test_raw_aliased_carry_would_error_without_guard():
    # pins WHY the guard exists: the donating jit alone raises the
    # "donate the same buffer twice" error on an aliased tree
    model = make_model(order=3)
    raw = model._carry.replace(
        stepper_state=model._fresh_stepper_state())
    with pytest.raises(jax.errors.JaxRuntimeError, match="twice"):
        _canonicalize_donating(raw)
    # the guarded path handles the same tree
    assert _canonicalize(raw) is not None


def test_prepare_for_donation_dedups_and_copies_small():
    a = jnp.arange(6.0)
    big = jnp.ones(_DONATE_MIN_BYTES // 8 + 16, dtype=jnp.float64)
    big_alias = jnp.ones(_DONATE_MIN_BYTES // 8 + 16, dtype=jnp.float64)
    tree = {"alias0": a, "alias1": a, "small": jnp.asarray(2.0),
            "big": big, "big_alias0": big_alias,
            "big_alias1": big_alias, "host": 3.0, "none": None}
    out = _prepare_for_donation(tree)
    # small aliased leaf: both occurrences copied -> distinct buffers
    assert id(out["alias0"]) != id(out["alias1"])
    # small leaf: copied (a fresh buffer, not the original)
    assert out["small"] is not tree["small"]
    # big unique leaf: passed straight through (to be donated)
    assert out["big"] is big
    # big aliased leaf: 1st occurrence passes through, 2nd is copied
    # (the id-seen dedup branch) so donation gets distinct buffers
    assert out["big_alias0"] is big_alias
    assert out["big_alias1"] is not big_alias
    # non-array leaves untouched
    assert out["host"] == 3.0
    assert out["none"] is None


def test_canonicalize_donates_big_copies_small():
    big = jnp.arange(float(_DONATE_MIN_BYTES // 8 + 16))
    small = jnp.asarray(1.25)
    out = _canonicalize({"big": big, "small": small})
    jax.block_until_ready(jax.tree_util.tree_leaves(out))
    # the big unique leaf was DONATED (the input buffer is consumed)
    assert big.is_deleted()
    # the small leaf was copied first, so the caller's handle survives
    assert not small.is_deleted()
    assert float(small) == 1.25
    # values preserved
    assert np.array_equal(np.asarray(out["big"]),
                          np.arange(float(_DONATE_MIN_BYTES // 8 + 16)))
    assert float(out["small"]) == 1.25


def test_canonicalize_preserves_scalars_and_none():
    out = _canonicalize({"f": 2.5, "i": 3, "none": None,
                         "arr": jnp.arange(4.0)})
    jax.block_until_ready(jax.tree_util.tree_leaves(out))
    assert float(out["f"]) == 2.5
    assert int(out["i"]) == 3
    assert out["none"] is None
    assert np.array_equal(np.asarray(out["arr"]), np.arange(4.0))


def test_module_handle_survives_construction():
    # a caller-held module handle's scalar param is NOT eaten by the
    # construction donation (probe: jax donates 0-d scalars, so the
    # guard's copy-small is load-bearing here)
    core = Core(nu=7e-3)
    model = Model(grid=make_grid(), modules=(core,),
                  time_stepper=AdamBashforth(DT, order=3))
    assert not core.nu.is_deleted()
    assert float(core.nu) == pytest.approx(7e-3)
    # and the live model reads the same value
    assert float(model.parameters["core.nu"]) == pytest.approx(7e-3)


# ================================================================
#  Donation — user buffers and value round-trips
# ================================================================
def test_set_fields_read_roundtrip_is_bitwise():
    model = make_model()
    u_in, b_in = ic(0.1), ic(-0.2)
    model.set_fields(u=u_in, b=b_in)
    u_out = np.asarray(model.state["u"].data)
    b_out = np.asarray(model.state["b"].data)
    assert np.array_equal(u_out, u_in)
    assert np.array_equal(b_out, b_in)


def test_user_buffers_valid_after_set_fields():
    model = make_model()
    grid = model.grid
    # a user ScalarField handle (device-resident) and an np.ndarray
    user_field = grid.create_field(init=lambda x: np.cos(2 * np.pi * x),
                                   name="u")
    user_array = ic(0.3)
    model.set_fields(u=user_field, b=user_array)
    # donation re-pads through decomposition.pad, so neither user
    # buffer becomes a carry leaf: both stay readable
    assert not user_field.data.is_deleted()
    assert np.isfinite(np.asarray(user_field.data)).all()
    assert np.array_equal(user_array, ic(0.3))


def test_double_set_fields_and_advance_after_advance():
    model = make_model()
    model.set_fields(u=ic(), b=ic(0.1))
    model.set_fields(u=ic(0.2), b=ic(0.3))  # no donated-buffer reuse
    r1 = model.advance(2)
    r2 = model.advance(2)  # advance after advance, no stale carry
    assert r1.steps_done == 2
    assert r2.steps_done == 2
    assert np.isfinite(np.asarray(model.state["u"].data)).all()


# ================================================================
#  Defragmentation — gating predicates
# ================================================================
def test_defrag_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("FRIDOM_DISABLE_DEFRAG", raising=False)
    assert _defrag_enabled() is True
    monkeypatch.setenv("FRIDOM_DISABLE_DEFRAG", "1")
    assert _defrag_enabled() is False
    monkeypatch.setenv("FRIDOM_DISABLE_DEFRAG", "0")
    assert _defrag_enabled() is True


def test_should_defragment_gating(monkeypatch):
    model = make_model()
    # force a cpu backend -> off regardless of the env (the gate keys
    # on jax.default_backend(), so pin it rather than trust the ambient
    # backend, which is gpu on a gpu host and would flip the assertion)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.delenv("FRIDOM_DISABLE_DEFRAG", raising=False)
    assert model._should_defragment() is False
    # force a gpu backend: now the env decides
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert model._should_defragment() is True
    monkeypatch.setenv("FRIDOM_DISABLE_DEFRAG", "1")
    assert model._should_defragment() is False


def test_defrag_skipped_on_cpu_backend(monkeypatch):
    # a cpu advance neither defrags nor clears the pending flag; pin the
    # cpu backend so the test holds on a gpu host too (the gate reads
    # jax.default_backend(), which is gpu on a gpu host)
    model = make_model()
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    calls = []
    monkeypatch.setattr(model, "_defragment_carry",
                        lambda: calls.append(1))
    model.set_fields(u=ic(), b=ic())
    model.advance(1)
    assert calls == []
    assert model._defrag_pending is True


# ================================================================
#  Defragmentation — arming / one-time firing
# ================================================================
def test_first_advance_defrags_once_and_rearms(monkeypatch):
    model = make_model()
    model.set_fields(u=ic(), b=ic(0.1))
    calls = []
    monkeypatch.setattr(model, "_should_defragment", lambda: True)
    monkeypatch.setattr(model, "_defragment_carry",
                        lambda: calls.append(1))
    model.advance(1)
    assert len(calls) == 1                 # fired once
    assert model._defrag_pending is False
    model.advance(1)
    assert len(calls) == 1                 # not re-triggered
    model.set_fields(u=ic(0.2))            # a _commit re-arms
    assert model._defrag_pending is True
    model.advance(1)
    assert len(calls) == 2                 # armed again -> fired again


def test_defrag_not_run_for_zero_steps(monkeypatch):
    model = make_model()
    calls = []
    monkeypatch.setattr(model, "_should_defragment", lambda: True)
    monkeypatch.setattr(model, "_defragment_carry",
                        lambda: calls.append(1))
    model.advance(0)                       # no chunk -> no defrag
    assert calls == []
    assert model._defrag_pending is True   # stays armed


def test_defrag_disabled_by_env(monkeypatch):
    model = make_model()
    model.set_fields(u=ic(), b=ic())
    calls = []
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("FRIDOM_DISABLE_DEFRAG", "1")
    monkeypatch.setattr(model, "_defragment_carry",
                        lambda: calls.append(1))
    model.advance(1)
    assert calls == []                     # kill switch suppresses it


# ================================================================
#  Defragmentation — the repack itself (value-preserving)
# ================================================================
def test_forced_cpu_defrag_is_bitwise_identical(monkeypatch):
    # baseline: no defrag
    base = make_model()
    base.set_fields(u=ic(), b=ic(0.1))
    base.advance(3)
    ref = [np.asarray(x)
           for x in jax.tree_util.tree_leaves(base._carry.state)]

    # repacked: force the real copy+delete path on cpu by dropping the
    # size threshold so the tiny leaves qualify, then defrag + advance
    monkeypatch.setattr("fridom.model.model._DEFRAG_MIN_BYTES", 0)
    rep = make_model()
    rep.set_fields(u=ic(), b=ic(0.1))
    rep._defragment_carry()
    jax.block_until_ready(jax.tree_util.tree_leaves(rep._carry))
    rep.advance(3)
    got = [np.asarray(x)
           for x in jax.tree_util.tree_leaves(rep._carry.state)]

    assert len(got) == len(ref)
    for a, b in zip(ref, got, strict=True):
        assert np.array_equal(a, b)


def test_defragment_carry_repacks_big_leaves(monkeypatch):
    model = make_model()
    model.set_fields(u=ic(), b=ic())
    monkeypatch.setattr("fridom.model.model._DEFRAG_MIN_BYTES", 0)
    old = {id(x) for x in jax.tree_util.tree_leaves(model._carry)
           if isinstance(x, jax.Array)}
    model._defragment_carry()
    jax.block_until_ready(jax.tree_util.tree_leaves(model._carry))
    new = {id(x) for x in jax.tree_util.tree_leaves(model._carry)
           if isinstance(x, jax.Array)}
    # every big (threshold 0 => all) array leaf migrated to a fresh
    # buffer; the model still advances on the repacked carry
    assert old.isdisjoint(new)
    model.advance(1)
    assert np.isfinite(np.asarray(model.state["u"].data)).all()


def test_defragment_carry_leaves_small_untouched():
    # at the real 1 MiB threshold every leaf of a tiny model is small
    # -> the repack skips them all and rebinds an identical carry
    model = make_model()
    model.set_fields(u=ic(), b=ic())
    ids_before = [id(x)
                  for x in jax.tree_util.tree_leaves(model._carry)]
    model._defragment_carry()
    ids_after = [id(x)
                 for x in jax.tree_util.tree_leaves(model._carry)]
    assert ids_before == ids_after
    assert np.isfinite(np.asarray(model.state["u"].data)).all()


def test_defragment_carry_handles_aliased_leaf(monkeypatch):
    # a committed carry never aliases, but the fresh order-3 ring does
    # -> drive the id-guard (copy once, free once) with the real thing
    model = make_model(order=3)
    model.set_fields(u=ic(), b=ic())
    # bypass _commit so the aliased ring reaches _defragment_carry
    model._carry = model._carry.replace(
        stepper_state=model._fresh_stepper_state())
    ring = [x for x in jax.tree_util.tree_leaves(
        model._carry.stepper_state) if isinstance(x, jax.Array)]
    ids = [id(x) for x in ring]
    assert len(ids) != len(set(ids))  # aliasing present
    monkeypatch.setattr("fridom.model.model._DEFRAG_MIN_BYTES", 0)
    model._defragment_carry()  # must not raise (no double delete)
    jax.block_until_ready(jax.tree_util.tree_leaves(model._carry))
    ring2 = [x for x in jax.tree_util.tree_leaves(
        model._carry.stepper_state) if isinstance(x, jax.Array)]
    # the ring zeros are preserved bitwise through the repack
    assert all(float(jnp.asarray(x).sum()) == 0.0 for x in ring2)
