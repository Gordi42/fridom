"""Parse XLA chunk buffer-assignment + optimized HLO into an
allocation/value table with source-op attribution.

For each variant dump dir it reads:
  *-buffer-assignment.txt   -> allocations (size, kind, values+offsets)
  *_after_optimizations.txt -> instruction -> (opcode, shape, op_name)

and emits, per variant, allocations >= THRESH bytes with the defining
instruction and the fridom source op the metadata attributes it to.
"""
from __future__ import annotations

import json
import os
import re
import sys

THRESH = 4 * 1024 * 1024  # 4 MB
DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dump")

VARIANTS = ["centered_n128", "centered_n192",
            "u5_baseline_n128", "u5_baseline_n192", "u5_selected_n128"]


def _files(vdir):
    ba = hlo = None
    for f in os.listdir(vdir):
        if f.endswith("-buffer-assignment.txt"):
            ba = os.path.join(vdir, f)
        elif f.endswith("_after_optimizations.txt"):
            hlo = os.path.join(vdir, f)
    return ba, hlo


ALLOC_RE = re.compile(
    r"^allocation (\d+): size (\d+), (.*?)(?:, shape|:)")
VALUE_RE = re.compile(
    r"^ value: <\d+ (\S+?) @\d+> \(size=(\d+),offset=(\d+)\): (.+)$")


def parse_ba(path):
    allocs = []
    cur = None
    with open(path) as fh:
        for line in fh:
            m = ALLOC_RE.match(line)
            if m:
                idx, size, kind = m.groups()
                cur = {"idx": int(idx), "size": int(size),
                       "kind_raw": kind.strip(), "values": []}
                allocs.append(cur)
                continue
            v = VALUE_RE.match(line)
            if v and cur is not None:
                name, sz, off, shape = v.groups()
                cur["values"].append(
                    {"name": name, "size": int(sz),
                     "offset": int(off), "shape": shape.strip()})
    return allocs


def classify(kind_raw):
    if kind_raw.startswith("parameter"):
        return "parameter"
    if kind_raw.startswith("constant"):
        return "constant"
    if kind_raw.startswith("thread-local"):
        return "thread-local"
    if "maybe-live-out" in kind_raw or kind_raw.startswith("output"):
        return "output"
    if kind_raw.startswith("preallocated") or kind_raw == "":
        return "temp"
    return kind_raw.split()[0]


# instruction line: "  %name = <shape> opcode(...), metadata={op_name=..}"
# shape may be a tuple "(f64[..], f64[..])" so opcode = first lowercase
# token immediately followed by '(' on the RHS (before the metadata).
NAME_RE = re.compile(r"^\s*%?([\w.\-]+) = (.*)$")
OPCODE_RE = re.compile(r"\b([a-z][a-z0-9-]*)\(")
SHAPE0_RE = re.compile(r"(\w+\[[0-9,]*\]|\(\s*\w+\[)")
OPNAME_RE = re.compile(r'op_name="([^"]*)"')


def parse_hlo(path):
    instr = {}
    with open(path) as fh:
        for line in fh:
            m = NAME_RE.match(line)
            if not m:
                continue
            name, rhs = m.groups()
            oc = OPCODE_RE.search(rhs)
            opcode = oc.group(1) if oc else "?"
            sh = SHAPE0_RE.match(rhs)
            shape = sh.group(1).rstrip("(").strip() if sh else "?"
            op = OPNAME_RE.search(rhs)
            instr[name] = {"opcode": opcode, "shape": shape,
                           "op_name": op.group(1) if op else ""}
    return instr


def source_bucket(op_name):
    """Collapse a jax op_name path to a coarse fridom source bucket."""
    s = op_name
    if not s:
        return "?"
    for key, tag in (
        ("CenteredAdvection", "advection"),
        ("UpwindAdvection", "advection"),
        ("WENOAdvection", "advection"),
        ("_SelectedFaceReconstruction", "advection"),
        ("FPlaneCoriolis", "coriolis"),
        ("ConstantStratification", "stratification"),
        ("DynamicalCore/projection", "projection/FFT"),
        ("DynamicalCore", "dynamical_core"),
        ("AdamBashforth", "timestepper"),
        ("stepper", "timestepper"),
    ):
        if key in s:
            return tag
    # fall back to the last informative path segment
    segs = [p for p in s.split("/") if p not in
            ("jit(_chunk_body)", "while", "body", "closed_call")]
    return "/".join(segs[:2]) if segs else "?"


def op_leaf(op_name):
    """Trailing op kind (gather, dynamic_update_slice, mul, ...)."""
    segs = op_name.split("/")
    return segs[-1] if segs else "?"


def analyze(vdir):
    ba, hlo = _files(vdir)
    allocs = parse_ba(ba)
    instr = parse_hlo(hlo)
    total = sum(a["size"] for a in allocs)
    by_kind = {}
    for a in allocs:
        k = classify(a["kind_raw"])
        by_kind[k] = by_kind.get(k, 0) + a["size"]
    big = []
    for a in allocs:
        if a["size"] < THRESH:
            continue
        k = classify(a["kind_raw"])
        # attribute via the values: collect (bucket, leaf, opcode) with
        # a representative defining instruction
        buckets = {}
        for v in a["values"]:
            meta = instr.get(v["name"], {})
            b = source_bucket(meta.get("op_name", ""))
            leaf = op_leaf(meta.get("op_name", ""))
            buckets.setdefault((b, meta.get("opcode", "?")),
                               []).append((v["name"], leaf))
        big.append({
            "idx": a["idx"], "size": a["size"], "kind": k,
            "kind_raw": a["kind_raw"], "n_values": len(a["values"]),
            "buckets": {f"{b}|{oc}": [n for n, _ in vs][:3]
                        for (b, oc), vs in buckets.items()},
        })
    return {"total": total, "by_kind": by_kind, "n_alloc": len(allocs),
            "big": big}


def main():
    out = {}
    for v in VARIANTS:
        vdir = os.path.join(DUMP, v)
        out[v] = analyze(vdir)
        r = out[v]
        print(f"\n===== {v} =====")
        print(f"n_alloc={r['n_alloc']} total={r['total']/1e6:.1f}MB")
        for k, sz in sorted(r["by_kind"].items(), key=lambda x: -x[1]):
            print(f"  {k:14s} {sz/1e6:8.1f} MB")
        print(f"  big allocations (>= {THRESH/1e6:.0f}MB): "
              f"{len(r['big'])}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "parsed_ba.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote parsed_ba.json")


if __name__ == "__main__":
    main()
