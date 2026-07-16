"""Tests for the persistent compilation cache default (_compile_cache).

Covers configure(): the FRIDOM_DISABLE_COMPILE_CACHE opt-out, the
never-clobber guard, the default XDG path, the FRIDOM_JAX_CACHE_DIR
override, the per-rank subdirectory under a real multi-process launch,
and the best-effort fallback when rank detection raises.
"""
from pathlib import Path

import jax
import pytest
from jax._src import distributed

import fridom._compile_cache as cache_mod

CACHE_DIR = "jax_compilation_cache_dir"
MIN_SECS = "jax_persistent_cache_min_compile_time_secs"
MIN_BYTES = "jax_persistent_cache_min_entry_size_bytes"


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(autouse=True)
def restore_cache_config():
    """Save and restore the three persistent-cache config values.

    The session-wide cache configuration (set by conftest) is shared by
    every other xdist test, so each test here MUST leave it as it found
    it — the teardown reinstates the saved values.
    """
    saved = {key: getattr(jax.config, key)
             for key in (CACHE_DIR, MIN_SECS, MIN_BYTES)}
    yield
    for key, value in saved.items():
        jax.config.update(key, value)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every case from a known, override-free environment."""
    monkeypatch.delenv("FRIDOM_DISABLE_COMPILE_CACHE", raising=False)
    monkeypatch.delenv("FRIDOM_JAX_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)


# ================================================================
#  configure(): opt-out and never-clobber
# ================================================================
def test_disabled_leaves_all_config_untouched(monkeypatch):
    monkeypatch.setenv("FRIDOM_DISABLE_COMPILE_CACHE", "1")
    jax.config.update(CACHE_DIR, None)
    jax.config.update(MIN_SECS, 7.0)
    jax.config.update(MIN_BYTES, 42)
    cache_mod.configure()
    assert jax.config.jax_compilation_cache_dir is None
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 7.0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == 42


def test_preexisting_cache_dir_is_not_clobbered(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/should/not/be/used")
    jax.config.update(CACHE_DIR, "/sentinel/cache")
    jax.config.update(MIN_SECS, 7.0)
    jax.config.update(MIN_BYTES, 42)
    cache_mod.configure()
    assert jax.config.jax_compilation_cache_dir == "/sentinel/cache"
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 7.0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == 42


# ================================================================
#  configure(): the enabled path
# ================================================================
def test_default_path_is_xdg_fridom_jax(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    jax.config.update(CACHE_DIR, None)
    cache_mod.configure()
    got = Path(jax.config.jax_compilation_cache_dir)
    assert got == tmp_path / "fridom" / "jax"
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == 0


def test_default_path_falls_back_to_home_cache(monkeypatch, tmp_path):
    # XDG_CACHE_HOME unset -> ~/.cache/fridom/jax
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    jax.config.update(CACHE_DIR, None)
    cache_mod.configure()
    got = Path(jax.config.jax_compilation_cache_dir)
    assert got == tmp_path / ".cache" / "fridom" / "jax"


def test_fridom_jax_cache_dir_override_is_respected(monkeypatch,
                                                    tmp_path):
    override = tmp_path / "my_cache"
    monkeypatch.setenv("FRIDOM_JAX_CACHE_DIR", str(override))
    monkeypatch.setenv("XDG_CACHE_HOME", "/should/not/be/used")
    jax.config.update(CACHE_DIR, None)
    cache_mod.configure()
    assert Path(jax.config.jax_compilation_cache_dir) == override


# ================================================================
#  configure(): multi-process rank isolation
# ================================================================
def test_distributed_run_gets_a_per_rank_subdir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(jax.distributed, "is_initialized",
                        lambda: True)
    monkeypatch.setattr(distributed.global_state, "process_id", 3)
    jax.config.update(CACHE_DIR, None)
    cache_mod.configure()
    got = Path(jax.config.jax_compilation_cache_dir)
    assert got == tmp_path / "fridom" / "jax" / "proc3"


def test_rank_detection_failure_falls_back_to_the_plain_dir(
        monkeypatch, tmp_path):
    def _boom():
        raise RuntimeError("no coordinator")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(jax.distributed, "is_initialized", _boom)
    jax.config.update(CACHE_DIR, None)
    cache_mod.configure()
    got = Path(jax.config.jax_compilation_cache_dir)
    assert got == tmp_path / "fridom" / "jax"
