"""Final runtime-delta decomposition from XLA cost_analysis (cs1 = one
clean step) + observed cs50 per-step times."""
from __future__ import annotations

# XLA cost_analysis, chunk_size=1 (one step, no scan/unroll)
COST = {  # (flops_G, bytes_accessed_MB)
    ("centered", 128): (6.610, 2605.8),
    ("u5_base", 128):  (7.374, 3286.9),
    ("u5_sel", 128):   (7.193, 3286.9),
    ("centered", 192): (22.268, 8562.3),
    ("u5_base", 192):  (24.684, 11008.6),
}
# observed cs50 per-step ms (benchmark table)
OBS = {
    ("centered", 128): 12.17, ("u5_base", 128): 22.70,
    ("u5_sel", 128): 22.14,
    ("centered", 192): 42.70, ("u5_base", 192): 77.85,
}
# temp write bytes per step (cs1 buffer-assignment sum) MB
TEMP_WR = {("centered", 128): 478.3, ("u5_base", 128): 781.7}


def main():
    for n in (128, 192):
        cf, cb = COST[("centered", n)]
        uf, ub = COST[("u5_base", n)]
        ct = OBS[("centered", n)]
        ut = OBS[("u5_base", n)]
        eff_c = cb / ct            # GB/s (MB/ms)
        eff_u = ub / ut
        dobs = ut - ct
        dbytes = ub - cb
        # extra bytes cost at centered efficiency
        extra_bytes_ms = dbytes / eff_c
        # efficiency-loss cost: u5 bytes at u5 vs centered efficiency
        eff_loss_ms = ub / eff_u - ub / eff_c
        print(f"===== n{n} =====")
        print(f"  centered: {cb:.0f}MB/step, {ct}ms -> {eff_c:.0f} GB/s "
              f"eff | flops {cf}G")
        print(f"  u5_base : {ub:.0f}MB/step, {ut}ms -> {eff_u:.0f} GB/s "
              f"eff | flops {uf}G")
        print(f"  observed delta = +{dobs:.2f} ms")
        print(f"  bytes_accessed delta = +{dbytes:.0f} MB/step "
              f"(+{dbytes/cb*100:.0f}%)")
        print(f"  DECOMPOSITION of +{dobs:.2f} ms:")
        print(f"    (a) extra HBM bytes @ centered eff = "
              f"+{extra_bytes_ms:.2f} ms ({extra_bytes_ms/dobs*100:.0f}%)")
        print(f"    (b) lower kernel efficiency (u5 {eff_u:.0f} vs "
              f"{eff_c:.0f} GB/s) = +{eff_loss_ms:.2f} ms "
              f"({eff_loss_ms/dobs*100:.0f}%)")
        print(f"    (c) arithmetic (f64 FMA) ~0 (u5_selected: "
              f"{COST[('u5_sel',128)][0] if n==128 else '—'} flops, "
              f"same bytes, not faster)")
        print(f"    check: (a)+(b) = {extra_bytes_ms+eff_loss_ms:.2f} "
              f"vs observed {dobs:.2f}")
        if n == 128:
            tc, tu = TEMP_WR[("centered", n)], TEMP_WR[("u5_base", n)]
            print(f"  temp-write delta = +{tu-tc:.0f} MB/step "
                  f"(= {(tu-tc)/dbytes*100:.0f}% of bytes delta; rest "
                  f"= extra reads/wide-halo)")
        print()


if __name__ == "__main__":
    main()
