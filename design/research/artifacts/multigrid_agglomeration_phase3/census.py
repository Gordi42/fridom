"""Collective census over an optimized-HLO step module.

Parses computations, builds the call graph (while/conditional/fusion/
call/async), and attributes every collective op definition to the
computation it lives in, then to structural role (outside the CG while,
inside the while body unconditionally, inside the conditional's real
branch, inside the skip branch). Counts op DEFINITIONS (not operand
references). Emits per-kind, per-computation, per-level tables.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

PATH = sys.argv[1]
with open(PATH) as fh:
    LINES = fh.readlines()

DTYPE_BYTES = {"f64": 8, "f32": 4, "c128": 16, "c64": 8,
               "s64": 8, "s32": 4, "u32": 4, "s8": 1, "u8": 1,
               "pred": 1, "f16": 2, "bf16": 2, "u64": 8, "s16": 2}

COLLECTIVE_OPS = {
    "all-gather", "all-gather-start",
    "all-reduce", "all-reduce-start",
    "all-to-all", "all-to-all-start",
    "collective-permute", "collective-permute-start",
    "reduce-scatter", "collective-broadcast",
}
# -done ops are sync points, not data movement; we count the -start /
# sync op as the launch. We still tally -done for cross-checking.
DONE_OPS = {"all-gather-done", "all-reduce-done", "all-to-all-done",
            "collective-permute-done"}


def _opcode(rhs: str) -> str | None:
    """Extract the opcode from an op RHS, skipping a leading tuple
    result shape ``(...)`` and the result shape token."""
    r = rhs.lstrip()
    if r.startswith("("):          # balanced tuple result shape
        d = 0
        for i, ch in enumerate(r):
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
                if d == 0:
                    r = r[i + 1:].lstrip()
                    break
    # opcode = first  identifier(  at this level (result shape token,
    # if present, is followed by '[' not '(')
    m = re.search(r"([a-z][a-zA-Z0-9_\-]*)\(", r)
    return m.group(1) if m else None


def shape_bytes(shape: str) -> int:
    """Bytes of a single array shape like f64[35,130,130]{2,1,0}."""
    m = re.match(r"([a-z0-9]+)\[([0-9,]*)\]", shape)
    if not m:
        return 0
    dt, dims = m.group(1), m.group(2)
    nb = DTYPE_BYTES.get(dt, 8)
    if dims == "":
        return nb
    n = 1
    for d in dims.split(","):
        n *= int(d)
    return n * nb


# ---- split into computations -------------------------------------
# A computation header ends with " {" at brace depth 0; ops live at
# depth 1; "}" at depth 0 closes it.
comps = {}          # name -> list of (op_lhs, opcode, rhs, full_line)
comp_order = []
cur = None
depth = 0
header_re = re.compile(r"^\s*(ENTRY\s+)?(%?[\w.\-]+)\s*\(")
for raw in LINES:
    line = raw.rstrip("\n")
    stripped = line.strip()
    if depth == 0:
        m = header_re.match(line)
        if m and stripped.endswith("{"):
            name = m.group(2)
            if not name.startswith("%"):
                name = "%" + name
            cur = name
            comps[cur] = []
            comp_order.append(cur)
            depth = 1
            continue
    if depth >= 1:
        # op line inside a computation
        opm = re.match(r"\s*(ROOT\s+)?(%[\w.\-]+)\s*=\s*(.*)", line)
        if opm and cur is not None:
            rhs = opm.group(3)
            opcode = _opcode(rhs)
            comps[cur].append((opm.group(2), opcode, rhs, line))
        # track brace depth
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            cur = None
            depth = 0

# ---- build call graph --------------------------------------------
# edges: caller -> list of (callee, kind)
edges = defaultdict(list)
ref_res = {
    "body": re.compile(r"body=(%[\w.\-]+)"),
    "condition": re.compile(r"condition=(%[\w.\-]+)"),
    "calls": re.compile(r"calls=(%[\w.\-]+)"),
    "to_apply": re.compile(r"to_apply=(%[\w.\-]+)"),
}
branch_re = re.compile(r"(?:branch_computations|called_computations)="
                       r"\{([^}]*)\}")
truefalse_re = re.compile(r"(?:true|false)_computation=(%[\w.\-]+)")
select_re = re.compile(r"select=(%[\w.\-]+)")

for cname, ops in comps.items():
    for lhs, opcode, rhs, line in ops:
        for kind, rx in ref_res.items():
            for t in rx.findall(rhs):
                edges[cname].append((t, kind, lhs, opcode))
        bm = branch_re.search(rhs)
        if bm:
            for t in re.findall(r"%[\w.\-]+", bm.group(1)):
                edges[cname].append((t, "branch", lhs, opcode))
        for t in truefalse_re.findall(rhs):
            edges[cname].append((t, "branch", lhs, opcode))

# ---- locate the CG while + its condition/body --------------------
entry = None
for raw in LINES:
    if raw.lstrip().startswith("ENTRY"):
        m = header_re.match(raw)
        nm = m.group(2)
        entry = nm if nm.startswith("%") else "%" + nm
        break

while_body = while_cond = None
cond_branches = []          # (op_lhs, [branch comps]) for the conditional
for cname, ops in comps.items():
    for lhs, opcode, rhs, line in ops:
        if opcode == "while":
            bm = ref_res["body"].search(rhs)
            cm = ref_res["condition"].search(rhs)
            while_body = bm.group(1) if bm else while_body
            while_cond = cm.group(1) if cm else while_cond
        if opcode == "conditional":
            br = branch_re.search(rhs)
            comps_list = (re.findall(r"%[\w.\-]+", br.group(1))
                          if br else truefalse_re.findall(rhs))
            cond_branches.append((cname, lhs, comps_list))

print(f"entry={entry}")
print(f"while_body={while_body} while_cond={while_cond}")
print(f"conditionals={cond_branches}")


def reachable(roots, blockers=()):
    """Computations reachable from roots without traversing INTO any
    blocker node (blockers themselves excluded)."""
    seen = set()
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n in seen or n in blockers:
            continue
        seen.add(n)
        for (callee, kind, _l, _o) in edges.get(n, ()):
            if callee not in seen:
                stack.append(callee)
    return seen


# structural regions:
# 1) OUTSIDE the CG while: reachable from entry, not descending into
#    while_body / while_cond.
outside = reachable([entry], blockers={while_body, while_cond})
# 2) the conditional branches. Identify real vs skip by collective load.
branch_sets = {}
for (host, lhs, blist) in cond_branches:
    for b in blist:
        branch_sets[b] = reachable([b])
# 3) INSIDE while body but OUTSIDE both conditional branches
all_branch_roots = set(branch_sets)
body_all = reachable([while_body]) if while_body else set()
cond_all = reachable([while_cond]) if while_cond else set()
body_unconditional = reachable(
    [while_body], blockers=all_branch_roots) if while_body else set()


def classify(cname):
    """Return a structural label for a computation."""
    for b, s in branch_sets.items():
        if cname in s:
            return f"cond-branch:{b}"
    if while_cond and cname in cond_all:
        return "while-condition"
    if while_body and cname in body_all:
        return "while-body-uncond"
    if cname in outside:
        return "outside-while"
    return "unreached"


# ---- collect collective op definitions ---------------------------
records = []           # dict per collective op
done_counts = defaultdict(int)
for cname, ops in comps.items():
    for lhs, opcode, rhs, line in ops:
        if opcode in DONE_OPS:
            done_counts[opcode] += 1
            continue
        if opcode not in COLLECTIVE_OPS:
            continue
        # result shape(s): the leading shape of rhs
        # for -start the result is often a tuple ((in),(out),ctx)
        # operand shapes: from the operand list inside the opcode(...)
        opnd_shapes = re.findall(r"[a-z0-9]+\[[0-9,]*\]", rhs)
        # the FIRST operand shape approximates the moved payload
        payload = opnd_shapes[0] if opnd_shapes else ""
        # bytes: for all-reduce/permute payload; for all-gather use
        # result (gathered) but we report operand (per-shard) payload
        b = shape_bytes(payload)
        # source/target pairs for collective-permute
        m_pairs = re.search(r"source_target_pairs=\{([^}]*)\}", rhs)
        pairs = m_pairs.group(1) if m_pairs else ""
        rid = re.search(r"replica_groups=\{?\[?[^,\}]*", rhs)
        records.append({
            "comp": cname, "op": lhs, "opcode": opcode,
            "payload": payload, "bytes": b, "pairs": pairs,
            "region": classify(cname),
        })

# ---- summaries ---------------------------------------------------
print("\n==== collective op DEFINITIONS by opcode ====")
by_op = defaultdict(int)
for r in records:
    by_op[r["opcode"]] += 1
for k in sorted(by_op):
    print(f"  {k:28s} {by_op[k]}")
print("  -- done ops (sync, cross-check) --")
for k in sorted(done_counts):
    print(f"  {k:28s} {done_counts[k]}")

print("\n==== by region x opcode ====")
reg_op = defaultdict(int)
for r in records:
    reg_op[(r["region"], r["opcode"])] += 1
for (reg, op) in sorted(reg_op):
    print(f"  {reg:24s} {op:26s} {reg_op[(reg, op)]}")

print("\n==== region totals ====")
reg_tot = defaultdict(int)
reg_bytes = defaultdict(int)
for r in records:
    reg_tot[r["region"]] += 1
    reg_bytes[r["region"]] += r["bytes"]
for reg in sorted(reg_tot):
    print(f"  {reg:24s} count={reg_tot[reg]:5d} "
          f"payload_bytes_sum={reg_bytes[reg]}")

# ---- per-level attribution of collective-permutes ----------------
# level Y,Z from a payload like f64[1,Y,Z] or f64[2,Y,Z] (x-halo of
# 1-2 planes) OR f64[dx,Y,1]/f64[dx,1,Z] etc. We bucket by (payload)
print("\n==== collective-permute payload shapes (in-body real) ====")
cp_shapes = defaultdict(int)
cp_bytes = {}
for r in records:
    if "collective-permute" not in r["opcode"]:
        continue
    cp_shapes[(r["region"], r["payload"])] += 1
    cp_bytes[r["payload"]] = r["bytes"]
for (reg, pay) in sorted(cp_shapes, key=lambda t: (t[0], t[1])):
    print(f"  {reg:24s} {pay:22s} x{cp_shapes[(reg, pay)]:4d} "
          f"({cp_bytes.get(pay,0)} B each)")

print("\n==== all-reduce payload shapes ====")
ar_shapes = defaultdict(int)
ar_bytes = {}
for r in records:
    if r["opcode"] not in ("all-reduce", "all-reduce-start"):
        continue
    ar_shapes[(r["region"], r["payload"])] += 1
    ar_bytes[r["payload"]] = r["bytes"]
for (reg, pay) in sorted(ar_shapes):
    print(f"  {reg:24s} {pay:18s} x{ar_shapes[(reg, pay)]:4d} "
          f"({ar_bytes.get(pay,0)} B each)")

print("\n==== all-gather payload shapes ====")
for r in records:
    if "all-gather" in r["opcode"]:
        print(f"  {r['region']:24s} {r['op']:24s} payload={r['payload']}"
              f" ({r['bytes']} B)  full_opcode={r['opcode']}")

print("\n==== all-to-all payload shapes ====")
a2a = defaultdict(int)
a2a_b = {}
for r in records:
    if "all-to-all" in r["opcode"]:
        a2a[(r["region"], r["payload"], r["opcode"])] += 1
        a2a_b[r["payload"]] = r["bytes"]
for key in sorted(a2a):
    reg, pay, op = key
    print(f"  {reg:24s} {op:22s} {pay:18s} x{a2a[key]:3d} "
          f"({a2a_b.get(pay,0)} B)")


# ================================================================
#  Per-level (by Y-extent) aggregation + executed-per-step totals
# ================================================================
def level_of(payload):
    """Map a CP payload f64[hx,Y,Z] to a multigrid level by Y-extent."""
    m = re.search(r"\[[0-9]+,([0-9]+),", payload)
    if not m:
        m2 = re.search(r"\[([0-9]+)\]", payload)
        return f"1d[{m2.group(1)}]" if m2 else payload
    y = int(m.group(1))
    return {130: "L128", 66: "L64", 34: "L32", 18: "L16",
            10: "L8", 6: "L4", 128: "L128*", 64: "L64*",
            32: "L32*", 16: "L16*", 8: "L8*", 4: "L4*"}.get(y, f"Y{y}")


OUT = "outside-while"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # total CG iters
# REAL = the conditional branch region carrying collectives (the CG
# real_step); the skip branch is a pure tuple pass-through (0 collectives)
_branch_load = defaultdict(int)
for r in records:
    if r["region"].startswith("cond-branch:"):
        _branch_load[r["region"]] += 1
REAL = max(_branch_load, key=_branch_load.get) if _branch_load else "none"
print(f"\nREAL(collective-bearing) branch region = {REAL}")

print("\n==== CP per-level: REAL branch (=1 CG iter/V-cycle) vs OUTSIDE ====")
lvl = defaultdict(lambda: defaultdict(int))
lvl_bytes = {}
for r in records:
    if "collective-permute" not in r["opcode"]:
        continue
    L = level_of(r["payload"])
    lvl[L][r["region"]] += 1
    lvl_bytes[L] = r["bytes"]
order = ["L128", "L128*", "L64", "L64*", "L32", "L32*", "L16", "L16*",
         "L8", "L8*", "L4", "L4*"]
keys = order + [k for k in lvl if k not in order]
print(f"  {'level':7s} {'real/Vcyc':>10s} {'outside':>8s} "
      f"{'bytes_ea':>10s}")
tot_real = tot_out = 0
for L in keys:
    if L not in lvl:
        continue
    rr = lvl[L].get(REAL, 0)
    oo = lvl[L].get(OUT, 0)
    tot_real += rr
    tot_out += oo
    print(f"  {L:7s} {rr:10d} {oo:8d} {lvl_bytes.get(L,0):10d}")
print(f"  {'TOTAL':7s} {tot_real:10d} {tot_out:8d}")

print(f"\n==== EXECUTED per STEP (ITERS={ITERS}: 1 peel + "
      f"{ITERS-1} real trips + {100-ITERS} skip trips) ====")
# per-region static counts
cnt = defaultdict(lambda: defaultdict(int))
byt = defaultdict(lambda: defaultdict(int))
for r in records:
    kind = ("CP" if "collective-permute" in r["opcode"]
            else "AR" if "all-reduce" in r["opcode"]
            else "AG" if "all-gather" in r["opcode"]
            else "A2A" if "all-to-all" in r["opcode"] else "OTH")
    cnt[r["region"]][kind] += 1
    byt[r["region"]][kind] += r["bytes"]
real_trips = ITERS - 1
for kind in ("CP", "AR", "AG", "A2A"):
    out_c = cnt[OUT][kind]
    real_c = cnt[REAL][kind]
    ex = out_c + real_trips * real_c
    out_b = byt[OUT][kind]
    real_b = byt[REAL][kind]
    ex_b = out_b + real_trips * real_b
    print(f"  {kind:4s} outside={out_c:4d} real/trip={real_c:4d} "
          f"-> executed={ex:5d}  bytes~={ex_b:,}")
tot = sum(cnt[OUT][k] + real_trips * cnt[REAL][k]
          for k in ("CP", "AR", "AG", "A2A"))
tot_b = sum(byt[OUT][k] + real_trips * byt[REAL][k]
            for k in ("CP", "AR", "AG", "A2A"))
print(f"  TOTAL collectives executed/step = {tot}  "
      f"total payload bytes ~= {tot_b:,}")
