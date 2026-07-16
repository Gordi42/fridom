"""True peak-liveness composition of the temp buffer.

Uses the after_optimizations live-range dump (BufferLiveRange gives
each value's [start,end] logical time) restricted to the values that
live in the single preallocated temp allocation. At the logical time of
maximum live temp bytes it reports the live set, sized and attributed to
fridom source ops via HLO op_name metadata. This is the authoritative
peak decomposition (offset-region sums overcount because of reuse).
"""
from __future__ import annotations

import json
import os
import re

from parse_ba import (DUMP, VARIANTS, _files, classify, op_leaf,
                      parse_ba, parse_hlo, source_bucket)


def shape_bytes(shape):
    m = re.match(r"(\w+)\[([0-9,]*)\]", shape)
    if not m:
        return 0
    dt, dims = m.groups()
    n = 1
    if dims:
        for d in dims.split(","):
            n *= int(d)
    return n * {"f64": 8, "f32": 4, "c128": 16, "c64": 8, "pred": 1,
               "s64": 8, "s32": 4, "u32": 4}.get(dt, 8)


def find_lr(vdir):
    for f in os.listdir(vdir):
        if f.endswith("-live-range.txt"):
            return os.path.join(vdir, f)
    return None


RANGE_RE = re.compile(r"^\s+(\S+?)\{[^}]*\}:(\d+)-(\d+)\s*$")


def parse_ranges(path):
    """base_name -> (min_start, max_end) merged over shape indices."""
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
            if not m:
                continue
            name, s, e = m.group(1), int(m.group(2)), int(m.group(3))
            s0, e0 = ranges.get(name, (s, e))
            ranges[name] = (min(s0, s), max(e0, e))
    return ranges


def analyze(vdir):
    ba, hlo = _files(vdir)
    allocs = parse_ba(ba)
    instr = parse_hlo(hlo)
    ranges = parse_ranges(find_lr(vdir))
    temp = max((a for a in allocs if classify(a["kind_raw"]) == "temp"),
               key=lambda a: a["size"])

    # temp values: base name -> (size, shape)  (largest per base name)
    tvals = {}
    for v in temp["values"]:
        base = v["name"].split("{")[0]
        if base not in tvals or v["size"] > tvals[base][0]:
            tvals[base] = (v["size"], v["shape"])

    # build interval list for temp values that have a live range
    intervals = []
    for base, (sz, shape) in tvals.items():
        if base in ranges:
            s, e = ranges[base]
            intervals.append((s, e, base, sz, shape))
    max_t = max((e for _, e, _, _, _ in intervals), default=0)

    # sweep to find peak live temp bytes
    best_t, best_bytes = 0, 0
    for t in range(max_t + 1):
        b = sum(sz for s, e, _, _, _ in intervals if s <= t <= e)
        if b > best_bytes:
            best_bytes, best_t = b, t

    live = [(base, sz, shape) for s, e, base, sz, shape in intervals
            if s <= best_t <= e]
    live.sort(key=lambda x: -x[1])

    # attribute each live buffer
    FIELD_MIN = 4 * 1024 * 1024
    attributed = []
    bucket_bytes = {}
    for base, sz, shape in live:
        if sz < FIELD_MIN:
            continue
        meta = instr.get(base, {})
        b = source_bucket(meta.get("op_name", ""))
        leaf = op_leaf(meta.get("op_name", ""))
        attributed.append({"name": base, "MB": round(sz / 1e6, 2),
                           "shape": shape.split("{")[0],
                           "bucket": b, "leaf": leaf,
                           "opcode": meta.get("opcode", "?"),
                           "op_name": meta.get("op_name", "")})
        key = f"{b}"
        bucket_bytes[key] = bucket_bytes.get(key, 0) + sz

    return {"temp_size": temp["size"], "peak_t": best_t,
            "peak_live_bytes": best_bytes,
            "n_live_ge4MB": len(attributed),
            "bucket_bytes": {k: round(v / 1e6, 1)
                             for k, v in sorted(bucket_bytes.items(),
                                                key=lambda x: -x[1])},
            "live": attributed}


def main():
    out = {}
    for v in VARIANTS:
        out[v] = analyze(os.path.join(DUMP, v))
    with open("live_peak.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for v in VARIANTS:
        r = out[v]
        print(f"\n===== {v} =====")
        print(f"  temp_size      = {r['temp_size']/1e6:.1f} MB")
        print(f"  peak live temp = {r['peak_live_bytes']/1e6:.1f} MB "
              f"(t={r['peak_t']})  n(>=4MB)={r['n_live_ge4MB']}")
        print("  peak live temp bytes by source bucket:")
        for k, mb in r["bucket_bytes"].items():
            print(f"    {k:22s} {mb:8.1f} MB")
    print("\nwrote live_peak.json")


if __name__ == "__main__":
    main()
