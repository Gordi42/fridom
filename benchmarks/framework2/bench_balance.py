"""The propagate+rebalance balance benchmark (T5 of the NNMD rewrite).

The classic diagnosed-imbalance protocol (Machenhauer-style balance
comparisons; Chouksey, Eden, Masur & Oliver 2023 JFM 971 A2), run
once per release of the method — this is a manual accuracy benchmark,
NOT a timing case and NOT a test (``tests/`` carries the cheap
epsilon-slope ladder instead):

1. build a small periodic f-plane shallow-water model
   (``sw.Model``, 32 x 32 on the doubly 2-pi domain, ``c = f = 1``
   so the deformation radius is one and the geostrophic spectrum
   peak ``k0 = 2`` is resolved);
2. take a geostrophically dominated IC — ``random_vortical``
   (Masur & Oliver spectrum, fixed seed), scaled to amplitude Ro.
   The **Rossby number lives in the state amplitude**, not in
   ``rossby_number=`` (the model keeps ``scaling.rossby = 1``; for
   the quadratic term the two spellings are dynamically
   equivalent). Historical note: this started as a workaround for
   ``OptimalBalance`` hard-coding a 0 -> 1 rossby ramp; OB now ramps
   to the model's own nominal value, so ``rossby_number=Ro`` with an
   O(1) amplitude works equally well;
3. for each balance method M: ``z_b = M(z)``; integrate the FULL
   nonlinear model for one eddy turnover ``T = turnover / Ro``;
   rebalance with the SAME method and report the diagnosed
   imbalance ``|| (I - P_div)(z_f - M(z_f)) ||_M / || z_f ||_M``;
4. record ``BalanceExpansion.residual_fast(z)`` per order at t = 0
   and report whether the cheap differential diagnostic ranks the
   orders the same way the expensive protocol does — the
   ranking-agreement check is this benchmark's key deliverable (it
   justifies never running the propagate+rebalance protocol in CI).

``P_div`` is the package ``DivergenceProjection`` — the structural
complement of the eigenmode families (the interpolation-Nyquist
planes where the staggered geostrophic column is a structural
zero). The aliased Sadourny advection pumps a dt-independent,
method-independent residual (~5e-3 relative over one turnover at
32^2) into exactly those planes; every balance method annihilates
that content identically (BalanceExpansion by construction,
OptimalBalance through its vortical base projection), so it is a
discretization artifact of the diagnosis, not wave imbalance, and
is excluded from the reported number (raw floors are printed too).

Methods: ``fr.transforms.BalanceExpansion`` orders 0-3 and
``fr.transforms.OptimalBalance`` (vortical base projection, exp
ramp over 2 inertial periods per leg, ``max_it = 2``). Optionally
one walled-channel f-plane column (orders 0-2, labeled channel
eigenbasis) at the middle Rossby number.

Usage (CPU, ~1 minute at the defaults)::

    JAX_PLATFORMS=cpu uv run python \
        benchmarks/framework2/bench_balance.py

Knobs: ``--size``, ``--rossby``, ``--dt``, ``--turnover``,
``--orders``, ``--ob-ramp``, ``--ob-max-it``, ``--seed``,
``--skip-channel``. Deterministic at fixed knobs (fixed seeds, one
jax cpu device). Results and the acceptance-target comparison:
``benchmarks/framework2/RESULTS.md``.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import fridom.framework2 as fr
import fridom.shallowwater2 as sw


def echo(text: str = "") -> None:
    """Report through stdout (the benchmark's interface)."""
    print(text)  # noqa: T201 — script output by design


def fmt(value: float) -> str:
    """Three-significant-digit scientific notation."""
    return f"{value:.2e}"


def table(header: list[str], rows: list[list[str]]) -> str:
    """Render a simple markdown table."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# ================================================================
#  Model, state and protocol building blocks
# ================================================================
def make_model(args: argparse.Namespace, *, periodic_y: bool = True):
    """Doubly (or zonally) periodic f-plane shallow-water model."""
    length = 2.0 * np.pi
    mx = fr.grid.meshes.IntervalMesh(
        args.size, (0.0, length), periodic=True, name="x")
    my = fr.grid.meshes.IntervalMesh(
        args.size, (0.0, length), periodic=periodic_y, name="y")
    return sw.Model(
        grid=fr.grid.Grid((mx, my)), csqr=1.0, rossby_number=1.0,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0), advection=True,
        time_stepper=fr.time_steppers.AdamBashforth(
            args.dt, order=3))


def diagnosed_imbalance(
    balance, propagator, structural, metric, z,
) -> tuple[float, float]:
    """Run balance -> propagate -> rebalance on one state.

    Returns ``(imbalance, raw)``: the M-norm residual of the final
    state against its own rebalance, relative to the final state —
    with and without the structural (eigenmode-complement) content
    excluded.
    """
    z_final = propagator(balance(z))
    residual = z_final - balance(z_final)
    scale = float(metric.norm(z_final))
    raw = float(metric.norm(residual)) / scale
    clean = residual - structural(residual)
    return float(metric.norm(clean)) / scale, raw


def vorticity_rossby(state) -> float:
    """Effective Rossby number ``max|zeta| / f0`` (f0 = 1)."""
    zeta = state["v"].diff("x") - state["u"].diff("y")
    return float(np.abs(np.asarray(zeta.data)).max())


def ranking(values: dict[int, float]) -> list[int]:
    """Order the keys by ascending value."""
    return sorted(values, key=values.__getitem__)


# ================================================================
#  The sweeps
# ================================================================
def run_periodic(args: argparse.Namespace) -> None:
    """Run the main sweep: methods x Rossby numbers, two tables."""
    model = make_model(args)
    metric = fr.EnergyMetric.from_model(model)
    structural = sw.transforms.DivergenceProjection.from_model(model)
    unit = sw.initial_conditions.random_vortical(model, seed=args.seed)
    echo(f"periodic 2pi domain, {args.size}^2, dt={args.dt}, "
         f"seed={args.seed}")
    echo(f"IC vorticity scale: max|zeta|/f0 = "
         f"{vorticity_rossby(unit):.2f} x amplitude")
    echo()

    balances = {
        order: fr.transforms.BalanceExpansion(
            model, order=order, lint=False)
        for order in args.orders}
    optimal = fr.transforms.OptimalBalance(
        model,
        sw.transforms.VorticalProjection.from_model(model),
        ramp_period=args.ob_ramp * 2.0 * np.pi,
        max_it=args.ob_max_it)

    imbalance: dict[str, dict[float, float]] = {}
    raw_floor: dict[float, float] = {}
    fast: dict[int, dict[float, float]] = {
        order: {} for order in args.orders}
    for ro in args.rossby:
        z = ro * unit
        propagator = fr.transforms.Propagator(
            model, runlen=args.turnover / ro)
        for order, balance in balances.items():
            imb, raw = diagnosed_imbalance(
                balance, propagator, structural, metric, z)
            imbalance.setdefault(f"BalanceExpansion({order})",
                                 {})[ro] = imb
            raw_floor[ro] = raw
            fast[order][ro] = balance.residual_fast(z)
        imb, _ = diagnosed_imbalance(
            optimal, propagator, structural, metric, z)
        imbalance.setdefault(
            f"OptimalBalance(ramp={args.ob_ramp:g}IP, "
            f"max_it={args.ob_max_it})", {})[ro] = imb

    ro_header = [f"Ro={ro:g}" for ro in args.rossby]
    echo("## Propagate+rebalance imbalance "
         f"(T = {args.turnover:g}/Ro, structural complement "
         "excluded)")
    echo()
    echo(table(
        ["method", *ro_header],
        [[name, *(fmt(row[ro]) for ro in args.rossby)]
         for name, row in imbalance.items()]))
    echo()
    echo("raw structural floor (order-independent, included by no "
         "method): "
         + ", ".join(f"Ro={ro:g}: {fmt(raw_floor[ro])}"
                     for ro in args.rossby))
    echo()
    echo("## residual_fast at t = 0 (the cheap differential "
         "diagnostic)")
    echo()
    echo(table(
        ["order", *ro_header],
        [[str(order), *(fmt(fast[order][ro]) for ro in args.rossby)]
         for order in args.orders]))
    echo()

    agree = True
    for ro in args.rossby:
        expensive = ranking({
            order: imbalance[f"BalanceExpansion({order})"][ro]
            for order in args.orders})
        cheap = ranking({order: fast[order][ro]
                         for order in args.orders})
        match = "AGREE" if expensive == cheap else "DISAGREE"
        agree &= expensive == cheap
        echo(f"ranking at Ro={ro:g}: propagate+rebalance "
             f"{expensive} vs residual_fast {cheap} -> {match}")
    echo()
    echo("ranking-agreement verdict: "
         + ("the cheap residual_fast diagnostic ranks the orders "
            "exactly like the expensive protocol at every Ro"
            if agree else
            "MISMATCH — see the tables above"))


#: highest BalanceExpansion order of the walled-channel column.
_CHANNEL_MAX_ORDER = 2


def run_channel(args: argparse.Namespace) -> None:
    """Run one walled-channel column (orders 0-2, middle Rossby)."""
    ro = args.rossby[len(args.rossby) // 2]
    model = make_model(args, periodic_y=False)
    metric = fr.EnergyMetric.from_model(model)
    structural = sw.transforms.DivergenceProjection.from_model(model)
    unit = sw.initial_conditions.random_vortical(model, seed=args.seed)
    z = ro * unit
    propagator = fr.transforms.Propagator(
        model, runlen=args.turnover / ro)
    orders = [order for order in args.orders
              if order <= _CHANNEL_MAX_ORDER]
    rows = []
    for order in orders:
        balance = fr.transforms.BalanceExpansion(
            model, order=order, lint=False)
        imb, raw = diagnosed_imbalance(
            balance, propagator, structural, metric, z)
        rows.append([f"BalanceExpansion({order})", fmt(imb),
                     fmt(raw), fmt(balance.residual_fast(z))])
    echo()
    echo(f"## Walled channel (f-plane, walls in y) at Ro={ro:g}")
    echo()
    echo(table(["method", "imbalance", "raw", "residual_fast(t=0)"],
               rows))


def main() -> None:
    """Parse the knobs and run the sweeps."""
    parser = argparse.ArgumentParser(
        description="propagate+rebalance balance benchmark (T5)")
    parser.add_argument("--size", type=int, default=32,
                        help="grid points per axis (default: 32)")
    parser.add_argument("--rossby", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2],
                        help="Rossby numbers (state amplitudes)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="AB3 time step (default: 0.02)")
    parser.add_argument("--turnover", type=float, default=1.0,
                        help="integration time in units of 1/Ro "
                             "(default: 1.0)")
    parser.add_argument("--orders", type=int, nargs="+",
                        default=[0, 1, 2, 3],
                        help="BalanceExpansion orders")
    parser.add_argument("--ob-ramp", type=float, default=2.0,
                        help="OptimalBalance ramp period in "
                             "inertial periods (default: 2)")
    parser.add_argument("--ob-max-it", type=int, default=2,
                        help="OptimalBalance iterations (default: 2)")
    parser.add_argument("--seed", type=int, default=123,
                        help="random_vortical seed (default: 123)")
    parser.add_argument("--skip-channel", action="store_true",
                        help="skip the walled-channel column")
    args = parser.parse_args()

    start = time.perf_counter()
    run_periodic(args)
    if not args.skip_channel:
        run_channel(args)
    echo()
    echo(f"total wall time: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
