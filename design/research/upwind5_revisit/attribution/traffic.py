"""Per-step temp traffic model.

Sum of all distinct temp-buffer value sizes = bytes written to scratch
per scan-body iteration (= one step). Each such array is also read back
>= once, so traffic ~ 2x write bytes (gathers/DUS move more). Bucketed
by fridom source module, and converted to ms at candidate bandwidths.
"""
from __future__ import annotations

import os

from parse_ba import (DUMP, VARIANTS, _files, classify, op_leaf,
                      parse_ba, parse_hlo, source_bucket)

BW = {"measured_107": 107e9, "hw_250": 250e9, "hw_330": 330e9}


def analyze(vdir):
    ba, hlo = _files(vdir)
    allocs = parse_ba(ba)
    instr = parse_hlo(hlo)
    temp = max((a for a in allocs if classify(a["kind_raw"]) == "temp"),
               key=lambda a: a["size"])
    write = {}
    nfield = {}
    total = 0
    for x in temp["values"]:
        base = x["name"].split("{")[0]
        buck = source_bucket(instr.get(base, {}).get("op_name", ""))
        write[buck] = write.get(buck, 0) + x["size"]
        total += x["size"]
        if x["size"] >= 4 * 1024 * 1024:
            nfield[buck] = nfield.get(buck, 0) + 1
    return {"total": total, "write": write, "nfield": nfield}


def main():
    res = {v: analyze(os.path.join(DUMP, v)) for v in VARIANTS}
    for v in VARIANTS:
        r = res[v]
        print(f"\n===== {v}  total temp write/step = "
              f"{r['total']/1e6:.1f} MB  (>=4MB fields: "
              f"{sum(r['nfield'].values())}) =====")
        for b, sz in sorted(r["write"].items(), key=lambda x: -x[1]):
            if sz < 1e6:
                continue
            print(f"    {b:22s} {sz/1e6:8.1f} MB  "
                  f"({r['nfield'].get(b,0)} fields>=4MB)")

    # delta drivers centered vs u5 at both sizes
    for n in ("128", "192"):
        c = res[f"centered_n{n}"]
        u = res[f"u5_baseline_n{n}"]
        dtot = (u["total"] - c["total"]) / 1e6
        print(f"\n### n{n}: total temp write/step "
              f"centered={c['total']/1e6:.1f} u5={u['total']/1e6:.1f} "
              f"delta=+{dtot:.1f} MB")
        buckets = sorted(set(c["write"]) | set(u["write"]))
        for b in buckets:
            d = (u["write"].get(b, 0) - c["write"].get(b, 0)) / 1e6
            if abs(d) < 1:
                continue
            print(f"    {b:22s} delta {d:+8.1f} MB")
        # traffic ~ 2x (write+read); delta ms at each BW
        wr = (u["total"] - c["total"])
        print(f"    => extra WRITE {wr/1e6:.0f}MB, "
              f"2x(w+r) {2*wr/1e6:.0f}MB. ms at BW:")
        for name, bw in BW.items():
            print(f"       {name:12s} 1x={wr/bw*1e3:.2f}ms  "
                  f"2x={2*wr/bw*1e3:.2f}ms  3x={3*wr/bw*1e3:.2f}ms")


if __name__ == "__main__":
    main()
