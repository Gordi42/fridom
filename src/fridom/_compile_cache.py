"""Persistent JAX compilation cache, enabled by default at import.

Description
-----------
``configure()`` turns on JAX's on-disk persistent compilation cache so
that the many small compilations a fridom run performs are paid once and
reused on every subsequent run of the same configuration. It is called
from ``fridom/__init__.py`` at import, before any lazy submodule can
trigger a compile.

What is cached
    The compiled XLA executables (keyed on the lowered HLO). A cold run
    of the canonical nonhydro2 model performs ~113 sub-second
    construction compiles plus the chunk executable; on a warm rerun
    these become cheap disk reads (measured time-to-first-step roughly
    halves).

Keying and safety
    JAX keys each entry on the HLO, the jaxlib version, the backend, and
    the relevant compilation flags, so the cache stays correct across
    fridom code changes: a change that alters the program produces a
    different HLO and hence a different key. Entries are therefore
    per-grid-size too — array shapes and grid constants are baked into
    the HLO, so each grid size pays its own one-time cold write and there
    is no cross-size reuse. Writes fall back silently on cache errors.

Footprint
    ~1-2 MB and ~120 entries per configuration. The cache is left
    unbounded on purpose: ``jax_compilation_cache_max_size`` keeps JAX's
    default (no cap) so nothing is evicted mid-session; users with quota
    concerns can set that flag themselves.

The min-compile-time threshold
    ``jax_persistent_cache_min_compile_time_secs`` is set to ``0.0``.
    This is load-bearing: JAX's default (``1.0``) would skip the ~113
    sub-second construction compiles, which are exactly the ones this
    cache exists to serve.

Environment knobs
    ``FRIDOM_DISABLE_COMPILE_CACHE=1``
        Disable the default entirely; ``configure()`` returns without
        importing jax or touching any configuration.
    ``FRIDOM_JAX_CACHE_DIR``
        Override the cache directory. When unset the cache lives under
        ``$XDG_CACHE_HOME/fridom/jax`` (falling back to
        ``~/.cache/fridom/jax``).

``configure()`` never clobbers an explicit configuration: if
``jax_compilation_cache_dir`` is already set — by the user, the test
suite's conftest, or a prior call — it returns untouched.
"""
from __future__ import annotations

import os
from pathlib import Path


# ================================================================
#  Public API
# ================================================================
def configure() -> None:
    """
    Enable JAX's persistent compilation cache (fridom default).

    Description
    -----------
    A no-op when ``FRIDOM_DISABLE_COMPILE_CACHE == "1"`` (returns
    before importing jax) or when a compilation cache directory is
    already configured (never clobbers an explicit setting). Otherwise
    points ``jax_compilation_cache_dir`` at the fridom cache location
    (``FRIDOM_JAX_CACHE_DIR`` or the XDG default) and lowers the
    persistent-cache thresholds to zero so sub-second compilations are
    cached too. Under a real multi-process launch each rank writes to
    its own subdirectory (jax's on-disk writes are not atomic). See the
    module docstring for the full contract.
    """
    if os.environ.get("FRIDOM_DISABLE_COMPILE_CACHE") == "1":
        return
    import jax  # noqa: PLC0415 — deferred so the disable path skips it

    # never clobber an explicit configuration (user, the test suite's
    # conftest, or a prior configure() call)
    if jax.config.jax_compilation_cache_dir is not None:
        return

    cache_dir = _base_cache_dir()
    subdir = _process_subdir()
    if subdir is not None:
        cache_dir = cache_dir / subdir

    # jax's LRUCache creates the directory on first write; nothing to
    # mkdir here. The 0-second threshold is load-bearing (see the
    # module docstring); max_size is left at jax's unbounded default.
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


# ================================================================
#  Internals
# ================================================================
def _base_cache_dir() -> Path:
    """Resolve the cache directory (override, else the XDG default)."""
    override = os.environ.get("FRIDOM_JAX_CACHE_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "fridom" / "jax"


def _process_subdir() -> str | None:
    """
    Per-rank subdirectory under a real multi-host launch, else None.

    Description
    -----------
    jax's on-disk cache writes are not atomic, so a directory shared
    across processes lets one rank read a half-written entry. The
    documented fridom multi-host flow calls
    ``jax.distributed.initialize()`` before importing fridom, so at
    ``configure()`` time a genuine multi-process run is already
    detectable: give each rank its own ``proc<id>`` subdirectory. The
    rank id is read off the distributed global state rather than
    ``jax.process_index()`` (which would initialize the backend — import
    fridom must not). Any failure of the detection falls back to the
    single-process default (no subdirectory).
    """
    try:
        import jax  # noqa: PLC0415 — only needed on the enabled path
        if not jax.distributed.is_initialized():
            return None
        from jax._src import distributed  # noqa: PLC0415
        process_id = distributed.global_state.process_id
    except Exception:  # noqa: BLE001 — detection is strictly best-effort
        return None
    return f"proc{process_id}"
