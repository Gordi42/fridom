"""Offset-aware peak-liveness decomposition of the temp buffer.

Each temp value carries an exact (name{idx}, size, offset). Its live
range is read per-name{idx} from BufferLiveRange. At each logical time
the live footprint = sum over distinct offsets of the max live size
there (offset = physical slot; values sharing an offset are time-
disjoint). At the peak time, attribute each live slot to a fridom
source op via HLO op_name metadata.
"""
from __future__ import annotations

import json
import os
import re

from parse_ba import (DUMP, VARIANTS, _files, classify, op_leaf,
                      parse_ba, parse_hlo, source_bucket)
from live_peak import find_lr

RANGE_RE = re.compile(r"^\s+(\S+):(\d+)-(\d+)\s*$")


def parse_ranges_full(path):
    """full value name (incl {idx}) -> (start, end)."""
    ranges = {}
    in_lr = False
    with open(path) as fh:
        for line in fh:
            if line.startswith("  BufferLiveRange:"):
                in_lr = True
                continue
            if not in_lr:
                continue
            m = RANGE_RE.match(line)
            if m:
                ranges[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return ranges


def analyze(vdir):
    ba, hlo = _files(vdir)
    allocs = parse_ba(ba)
    instr = parse_hlo(hlo)
    ranges = parse_ranges_full(find_lr(vdir))
    temp = max((a for a in allocs if classify(a["kind_raw"]) == "temp"),
               key=lambda a: a["size"])

    # each value: (offset, size, base, range)
    vals = []
    miss = 0
    for x in temp["values"]:
        name = x["name"]
        r = ranges.get(name)
        if r is None:
            # BA sometimes writes name without braces; try base{}
            r = ranges.get(name + "{}")
        if r is None:
            miss += 1
            continue
        vals.append((x["offset"], x["size"], name.split("{")[0],
                     x["shape"], r[0], r[1]))
    max_t = max((e for *_, e in vals), default=0)

    # sweep: at each t, live footprint = sum over offsets of max size
    def footprint(t):
        slots = {}
        for off, sz, *_rest, s, e in vals:
            if s <= t <= e:
                slots[off] = max(slots.get(off, 0), sz)
        return sum(slots.values()), slots

    best_t, best_b = 0, 0
    for t in range(max_t + 1):
        b, _ = footprint(t)
        if b > best_b:
            best_b, best_t = b, t

    _, peak_slots = footprint(best_t)
    # attribute each live slot (>=4MB) at peak
    FIELD_MIN = 4 * 1024 * 1024
    live = []
    for off, sz in peak_slots.items():
        if sz < FIELD_MIN:
            continue
        # find the value(s) at this offset live at best_t
        occ = [(base, shape) for o, s2, base, shape, s, e in vals
               if o == off and s <= best_t <= e]
        base, shape = occ[0]
        meta = instr.get(base, {})
        live.append({"offset": off, "MB": round(sz / 1e6, 2),
                     "shape": shape.split("{")[0],
                     "bucket": source_bucket(meta.get("op_name", "")),
                     "leaf": op_leaf(meta.get("op_name", "")),
                     "op_name": meta.get("op_name", ""),
                     "name": base})
    live.sort(key=lambda x: -x["MB"])
    bucket = {}
    for x in live:
        bucket[x["bucket"]] = bucket.get(x["bucket"], 0) + x["MB"]
    return {"temp_MB": round(temp["size"] / 1e6, 1),
            "peak_MB": round(best_b / 1e6, 1), "peak_t": best_t,
            "missed_values": miss, "n_live_ge4MB": len(live),
            "bucket_MB": dict(sorted(bucket.items(),
                                     key=lambda x: -x[1])),
            "live": live}


def main():
    out = {}
    for v in VARIANTS:
        out[v] = analyze(os.path.join(DUMP, v))
        r = out[v]
        print(f"\n===== {v}  temp={r['temp_MB']}MB "
              f"peak_liveness={r['peak_MB']}MB (t={r['peak_t']}) "
              f"missed={r['missed_values']} =====")
        print(f"  live full-field slots (>=4MB): {r['n_live_ge4MB']}")
        for k, mb in r["bucket_MB"].items():
            print(f"    {k:20s} {mb:8.1f} MB")
    with open("peak2.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote peak2.json")


if __name__ == "__main__":
    main()
