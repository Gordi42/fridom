"""Analyze an nsys cuda_gpu_trace CSV: GPU-active span vs idle, gap
distribution, per-kernel launch config. Localises whether the wall gap
is on-device (kernel time) or off-device (idle bubbles)."""
from __future__ import annotations

import csv
import sys


def load(path):
    rows = []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = None
        for r in rd:
            if not r:
                continue
            if header is None:
                if r and r[0].strip() == "Start (ns)":
                    header = [c.strip() for c in r]
                continue
            rows.append(r)
    idx = {h: i for i, h in enumerate(header)}
    out = []
    for r in rows:
        try:
            start = int(r[idx["Start (ns)"]])
            dur = int(r[idx["Duration (ns)"]])
        except (ValueError, IndexError):
            continue
        name = r[idx["Name"]].strip()
        gx = r[idx["GrdX"]]; gy = r[idx["GrdY"]]; gz = r[idx["GrdZ"]]
        bx = r[idx["BlkX"]]; by = r[idx["BlkY"]]; bz = r[idx["BlkZ"]]
        reg = r[idx.get("Reg/Trd", -1)] if "Reg/Trd" in idx else ""
        out.append((start, dur, name, gx, gy, gz, bx, by, bz, reg))
    out.sort()
    return out


def norm(name):
    # collapse XLA numeric suffixes / template args to a short tag
    n = name
    if n.startswith("void "):
        n = n[5:]
    n = n.split("<")[0].split("(")[0]
    return n[:44]


def main():
    path, label = sys.argv[1], sys.argv[2]
    rows = load(path)
    first = rows[0][0]
    last = max(s + d for s, d, *_ in rows)
    span = last - first
    active = sum(d for _, d, *_ in rows)
    idle = span - active
    print(f"\n===== {label} ({len(rows)} kernels) =====")
    print(f"span={span/1e6:.2f} ms  active={active/1e6:.2f} ms "
          f"({active/span*100:.1f}%)  idle={idle/1e6:.2f} ms "
          f"({idle/span*100:.1f}%)")
    # gaps between consecutive kernels (single stream assumption)
    gaps = []
    prev_end = None
    for s, d, *_ in rows:
        if prev_end is not None and s > prev_end:
            gaps.append(s - prev_end)
        prev_end = max(prev_end, s + d) if prev_end else s + d
    gaps.sort(reverse=True)
    tot_gap = sum(gaps)
    print(f"inter-kernel gap total={tot_gap/1e6:.2f} ms over {len(gaps)} "
          f"gaps; mean={tot_gap/max(1,len(gaps))/1e3:.2f} us; "
          f"top gaps us: {[round(g/1e3,1) for g in gaps[:8]]}")
    # per-kernel-name aggregation
    agg = {}
    for s, d, name, gx, gy, gz, bx, by, bz, reg in rows:
        t = norm(name)
        a = agg.setdefault(t, [0, 0, (gx, gy, gz, bx, by, bz, reg)])
        a[0] += 1
        a[1] += d
    print(f"{'kernel':46s} {'inst':>5s} {'tot us':>9s} {'avg us':>8s} "
          f"grid/block/reg")
    for t in sorted(agg, key=lambda k: -agg[k][1]):
        c, tot, cfg = agg[t]
        print(f"{t:46s} {c:>5d} {tot/1e3:>9.1f} {tot/c/1e3:>8.2f} "
              f"g={cfg[0]}x{cfg[1]}x{cfg[2]} b={cfg[3]}x{cfg[4]}x{cfg[5]} "
              f"r={cfg[6]}")


if __name__ == "__main__":
    main()
