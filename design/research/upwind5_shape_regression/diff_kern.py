"""Diff two nsys cuda_gpu_kern_sum CSVs (w3 vs w4).

Reports total on-device time/step, per-category buckets, and top kernels
for each, so the shape-sensitive kernel(s) surface. Kernel identity is
not stable across the two executables, so we compare (a) total, (b)
coarse buckets by name pattern, (c) the ranked top lists side by side.

Usage: python diff_kern.py <csvA> <stepsA> <labelA> <csvB> <stepsB> <labelB>
"""
from __future__ import annotations

import csv
import re
import sys

BUCKETS = [
    ("fft", re.compile(r"fft|cufft", re.I)),
    ("transpose", re.compile(r"transpose", re.I)),
    ("copy/memset", re.compile(r"memcpy|memset|copy", re.I)),
    ("reduce", re.compile(r"reduc", re.I)),
    ("gemm/cublas", re.compile(r"gemm|cublas|cutlass", re.I)),
    ("fusion", re.compile(r"fusion|loop|emitter", re.I)),
]


def load(path):
    rows = []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = None
        for r in rd:
            if not r:
                continue
            if header is None:
                # find header row (contains "Name")
                if any(c.strip() == "Name" for c in r):
                    header = [c.strip() for c in r]
                continue
            rows.append(r)
    idx = {h: i for i, h in enumerate(header)}

    def col(names):
        for n in names:
            if n in idx:
                return idx[n]
        raise KeyError(names)

    ci_tot = col(["Total Time (ns)", "Total Time(ns)"])
    ci_inst = col(["Instances", "Num Calls", "Count"])
    ci_name = col(["Name"])
    out = []
    for r in rows:
        tot = float(r[ci_tot].replace(",", ""))
        inst = int(float(r[ci_inst].replace(",", "")))
        out.append((r[ci_name].strip(), tot, inst))
    return out


def bucket_of(name):
    for b, rx in BUCKETS:
        if rx.search(name):
            return b
    return "other"


def summarize(path, steps, label):
    rows = load(path)
    total = sum(t for _, t, _ in rows)
    buckets = {}
    for name, t, inst in rows:
        b = bucket_of(name)
        d = buckets.setdefault(b, [0.0, 0])
        d[0] += t
        d[1] += inst
    print(f"\n===== {label}  (file={path.split('/')[-1]}, steps={steps}) =====")
    print(f"total GPU time = {total/1e6:.2f} ms over {steps} steps "
          f"= {total/1e6/steps*1e3:.4f} us/step... "
          f"= {total/1e9/steps*1e3:.4f} ms/step")
    print(f"{'bucket':16s} {'us/step':>10s} {'inst/step':>10s} {'%':>6s}")
    for b in sorted(buckets, key=lambda k: -buckets[k][0]):
        t, inst = buckets[b]
        print(f"{b:16s} {t/1e3/steps:>10.2f} {inst/steps:>10.2f} "
              f"{t/total*100:>6.1f}")
    print("-- top 18 kernels (us/step, inst/step, name) --")
    for name, t, inst in sorted(rows, key=lambda x: -x[1])[:18]:
        print(f"  {t/1e3/steps:>9.2f} {inst/steps:>8.2f}  {name[:90]}")
    return total / 1e9 / steps * 1e3, {b: buckets[b][0]/1e9/steps*1e3
                                       for b in buckets}


def main():
    a, sa, la, b, sb, lb = sys.argv[1:7]
    ma, ba = summarize(a, int(sa), la)
    mb, bb = summarize(b, int(sb), lb)
    print(f"\n===== DELTA {la} vs {lb} =====")
    print(f"total ms/step: {la}={ma:.4f}  {lb}={mb:.4f}  "
          f"delta={ma-mb:+.4f} ({(ma/mb-1)*100:+.1f}%)")
    print(f"{'bucket':16s} {la+' ms':>12s} {lb+' ms':>12s} {'delta ms':>12s}")
    for bk in sorted(set(ba) | set(bb),
                     key=lambda k: -(ba.get(k, 0) - bb.get(k, 0))):
        print(f"{bk:16s} {ba.get(bk,0):>12.4f} {bb.get(bk,0):>12.4f} "
              f"{ba.get(bk,0)-bb.get(bk,0):>+12.4f}")


if __name__ == "__main__":
    main()
