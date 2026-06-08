import os
import re
import ast
import glob
import math
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────
# Directory of this script (final_codes/Evaluation/)
BASE = os.path.dirname(os.path.abspath(__file__))

# Repository root (two levels up from final_codes/Evaluation/)
REPO_ROOT = os.path.join(BASE, "..", "..")

# Shorthand for the lcl_infrq_glbl_frq data directory
LCL_DIR = os.path.join(REPO_ROOT, "lcl_infrq_glbl_frq")

EXPERIMENTS = [
    {
        "name":        "Accidents — Spread-Thin",
        "centralized": os.path.join(REPO_ROOT, "data_acc", "cnt_acci.txt"),
        "locals_dir":  os.path.join(LCL_DIR, "locals_res", "accidents_lcls_thnSprd"),
        "locals_glob": "cl*.txt",
        "feds_dir":    os.path.join(LCL_DIR, "feds_res", "accidents_thnSpred"),
        "shards_dir":  os.path.join(LCL_DIR, "data", "accidents_thnSprd"),
        "usage_low":   300,
        "usage_high":  3000,
        "skip_feds":   [],
    },
    {
        "name":        "Chess-KK — IID",
        "centralized": os.path.join(LCL_DIR, "datasets", "cnt_chess_kk.txt"),
        "locals_dir":  os.path.join(LCL_DIR, "locals_res", "chesskk_lcls_iid"),
        "locals_glob": "cnt_chesskk_cl*_iid.txt",
        "feds_dir":    os.path.join(LCL_DIR, "feds_res", "chesskk_iid"),
        "shards_dir":  None,   # no raw shard .dat files exist for IID
        "usage_low":   10,
        "usage_high":  200,
        # secAgg file is byte-for-byte the accidents IID result (wrong data:
        # max usage 37,690 >> Chess-KK's 3,196 total transactions)
        "skip_feds":   ["secAgg_chesskk_srvr_iid.txt"],
    },
    {
        "name":        "Chess-KK — Spread-Thin",
        "centralized": os.path.join(LCL_DIR, "datasets", "cnt_chess_kk.txt"),
        "locals_dir":  os.path.join(LCL_DIR, "locals_res", "chesskk_lcls_thnSprd"),
        "locals_glob": "cnt_chesskk_cl*_thnSprd.txt",
        "feds_dir":    os.path.join(LCL_DIR, "feds_res", "chesskk_thnSprd"),
        "shards_dir":  os.path.join(LCL_DIR, "data", "chess_kk_thnSprd"),
        "usage_low":   10,
        "usage_high":  200,
        "skip_feds":   [],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def parse_codetable(path):
    result = {}
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\d+\s*-\s*(\[.*?\])\s*:\s*(\d+)", line.strip())
            if m:
                items = frozenset(ast.literal_eval(m.group(1)))
                result[items] = int(m.group(2))
    return result


def load_transactions(path):
    txs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                txs.append(set(line.split()))
    return txs


def build_inverted_index(txs):
    idx = defaultdict(set)
    for i, tx in enumerate(txs):
        for item in tx:
            idx[item].add(i)
    return idx


def count_pattern_in_shard(pattern, inv_idx):
    items = list(pattern)
    result = inv_idx.get(items[0], set()).copy()
    for item in items[1:]:
        result &= inv_idx.get(item, set())
        if not result:
            return 0
    return len(result)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────
def _rank(lst):
    sorted_vals = sorted(enumerate(lst), key=lambda x: x[1])
    ranks = [0.0] * len(lst)
    i, n = 0, len(sorted_vals)
    while i < n:
        j = i
        while j < n - 1 and sorted_vals[j + 1][1] == sorted_vals[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mx) ** 2 for i in range(n))
        * sum((ry[i] - my) ** 2 for i in range(n))
    )
    return num / den if den != 0 else float("nan")


def weighted_recall_at_k(A, B, k):
    topk  = sorted(A.items(), key=lambda x: x[1], reverse=True)[:k]
    total   = sum(u for _, u in topk)
    matched = sum(u for x, u in topk if x in B)
    return matched / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────────────────────
def compute_gap_metrics(A, local_cts, usage_low, usage_high):
    set_A    = set(A.keys())
    ct_union = set()
    for ct in local_cts:
        ct_union |= set(ct.keys())
    inter_A    = set_A & ct_union
    gap        = set_A - ct_union
    target_A   = {x for x, u in A.items() if usage_low <= u <= usage_high}
    gap_target = target_A & gap
    return gap, gap_target, ct_union, inter_A, target_A


def compute_spread_factors(gap, A, shard_inv_indices):
    results = []
    for X in gap:
        per_shard = [count_pattern_in_shard(X, idx) for idx in shard_inv_indices]
        max_local = max(per_shard) if per_shard else 0
        sX = A[X] / max_local if max_local > 0 else float("inf")
        results.append((X, sX, per_shard))
    results.sort(key=lambda t: t[1], reverse=True)
    return results


def compute_gap_recovery(gap, gap_target, B):
    set_B      = set(B.keys())
    rec        = gap & set_B
    rec_target = gap_target & set_B
    grr        = len(rec)        / len(gap)        if gap        else 1.0
    grr_t      = len(rec_target) / len(gap_target) if gap_target else 1.0
    return {
        "|B|":         len(set_B),
        "|G∩B|":       len(rec),
        "GRR":         grr,
        "|Gt∩B|":      len(rec_target),
        "GRR_target":  grr_t,
    }


def compute_fidelity(A, B):
    set_A = set(A.keys())
    set_B = set(B.keys())
    inter = set_A & set_B

    recall    = len(inter) / len(set_A) if set_A else 1.0
    precision = len(inter) / len(set_B) if set_B else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    rho = spearman([A[x] for x in inter], [B[x] for x in inter]) \
          if len(inter) >= 2 else float("nan")

    return {
        "|A|":       len(set_A),
        "|B|":       len(set_B),
        "|A∩B|":     len(inter),
        "Recall":    recall,
        "Precision": precision,
        "F1":        f1,
        "WR@50":     weighted_recall_at_k(A, B, 50),
        "WR@100":    weighted_recall_at_k(A, B, 100),
        "Spearman":  rho,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(v, decimals=4):
    if isinstance(v, float):
        return f"{v:.{decimals}f}" if math.isfinite(v) else "  nan"
    return str(v)


def print_gap_table(gap, gap_target, ct_union, inter_A, A, out):
    lines = [
        "",
        "  ── Discovery Gap ────────────────────────────────────────────────",
        f"  |A|             = {len(A)}",
        f"  |∪CT_i|         = {len(ct_union):,}",
        f"  |∪CT_i ∩ A|     = {len(inter_A)}",
        f"  |G|             = {len(gap)}",
        f"  |G_target|      = {len(gap_target)}",
    ]
    for ln in lines:
        print(ln)
        out.append(ln)


def print_spread_table(spread_results, out):
    lines = [
        "",
        "  ── Spread Factors s(X) for Gap Itemsets (sorted desc) ──────────",
        f"  {'|X|':>4}  {'usage_glb':>10}  {'s(X)':>7}  {'max_local':>10}  {'per_shard_counts'}",
        "  " + "-" * 72,
    ]
    for X, sX, per_shard in spread_results:
        glb_usage = sum(per_shard)   # raw occurrence total across shards
        max_loc   = max(per_shard) if per_shard else 0
        sX_str    = f"{sX:.3f}" if math.isfinite(sX) else "  inf"
        counts    = " ".join(str(c) for c in per_shard)
        ln = f"  {len(X):>4}  {glb_usage:>10}  {sX_str:>7}  {max_loc:>10}  [{counts}]"
        lines.append(ln)
    for ln in lines:
        print(ln)
        out.append(ln)


def print_gap_recovery_table(gap, gap_target, fed_results, ct_union, out):
    hdr = (f"\n  ── Gap Recovery (GRR) ──────────────────────────────────────────\n"
           f"  {'Method':<40} {'|B|':>7}  {'|G∩B|':>6}/{len(gap):<3}  "
           f"{'GRR':>6}  {'|Gt∩B|':>7}/{len(gap_target):<3}  {'GRR_t':>6}")
    div = "  " + "-" * 85
    baseline = (f"  {'Local ∪CT_i  (baseline)':<40} {len(ct_union):>7}  "
                f"{'0':>6}/{len(gap):<3}  {'0.0000':>6}  {'0':>7}/{len(gap_target):<3}  {'0.0000':>6}")
    lines = [hdr, div, baseline]
    for fname, m in fed_results:
        ln = (f"  {fname:<40} {m['|B|']:>7}  "
              f"{m['|G∩B|']:>6}/{len(gap):<3}  {m['GRR']:>6.4f}  "
              f"{m['|Gt∩B|']:>7}/{len(gap_target):<3}  {m['GRR_target']:>6.4f}")
        lines.append(ln)
    for ln in lines:
        print(ln)
        out.append(ln)

    # LaTeX rows
    latex_hdr = "\n  ── LaTeX rows (gap-recovery table) ────────────────────────────"
    print(latex_hdr); out.append(latex_hdr)
    baseline_latex = (f"  Local $\\cup CT_i$ & {len(ct_union):,} & 0 & 0.00 & 0 & 0.00 \\\\")
    print(baseline_latex); out.append(baseline_latex)
    for fname, m in fed_results:
        row = (f"  {fname} & {m['|B|']:,} & {m['|G∩B|']} & {m['GRR']:.2f} & "
               f"{m['|Gt∩B|']} & {m['GRR_target']:.2f} \\\\")
        print(row); out.append(row)


def print_fidelity_table(fed_results, out):
    hdr = (f"\n  ── Fidelity (vs. Centralised A) ────────────────────────────────\n"
           f"  {'Method':<40} {'|A|':>5} {'|B|':>6} {'|A∩B|':>6} "
           f"{'Rec':>6} {'Prec':>6} {'F1':>6} "
           f"{'WR@50':>7} {'WR@100':>7} {'Spear':>7}")
    div = "  " + "-" * 98
    lines = [hdr, div]
    for fname, m in fed_results:
        sp = f"{m['Spearman']:.4f}" if math.isfinite(m["Spearman"]) else "   nan"
        ln = (f"  {fname:<40} {m['|A|']:>5} {m['|B|']:>6} {m['|A∩B|']:>6} "
              f"{m['Recall']:>6.4f} {m['Precision']:>6.4f} {m['F1']:>6.4f} "
              f"{m['WR@50']:>7.4f} {m['WR@100']:>7.4f} {sp:>7}")
        lines.append(ln)
    for ln in lines:
        print(ln)
        out.append(ln)

    # LaTeX rows
    latex_hdr = "\n  ── LaTeX rows (fidelity table) ────────────────────────────────"
    print(latex_hdr); out.append(latex_hdr)
    for fname, m in fed_results:
        sp = f"{m['Spearman']:.3f}" if math.isfinite(m["Spearman"]) else "--"
        row = (f"  {fname} & {m['|A|']:,} & {m['|B|']:,} & {m['|A∩B|']:,} & "
               f"{m['Recall']:.3f} & {m['Precision']:.3f} & {m['F1']:.3f} & "
               f"{m['WR@50']:.3f} & {m['WR@100']:.3f} & {sp} \\\\")
        print(row); out.append(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(cfg, out):
    sep = "=" * 80
    hdr = f"  EXPERIMENT: {cfg['name']}"
    print(f"\n{sep}\n{hdr}\n{sep}")
    out += [f"\n{sep}", hdr, sep]

    # 1. Centralised codetable
    print(f"\n[1] Loading centralised codetable: {cfg['centralized']}")
    A = parse_codetable(cfg["centralized"])
    print(f"    |A| = {len(A)}")

    # 2. Local codetables
    print(f"\n[2] Loading local codetables from: {cfg['locals_dir']}")
    local_paths = sorted(glob.glob(os.path.join(cfg["locals_dir"], cfg["locals_glob"])))
    if not local_paths:
        print("    [WARN] No local codetable files found — skipping experiment.")
        out.append("    [WARN] No local codetable files found.")
        return
    local_cts = []
    for p in local_paths:
        ct = parse_codetable(p)
        local_cts.append(ct)
        print(f"    {os.path.basename(p)}: {len(ct):,} itemsets")

    # 3. Gap metrics
    print("\n[3] Computing gap metrics ...")
    gap, gap_target, ct_union, inter_A, target_A = compute_gap_metrics(
        A, local_cts, cfg["usage_low"], cfg["usage_high"]
    )
    print_gap_table(gap, gap_target, ct_union, inter_A, A, out)
    ln = f"  |target A| (usage {cfg['usage_low']}–{cfg['usage_high']}) = {len(target_A)}"
    print(ln); out.append(ln)

    # 4. Spread factors (only when raw shard data is available)
    if cfg["shards_dir"] is not None and gap:
        print(f"\n[4] Computing spread factors from shards: {cfg['shards_dir']}")
        shard_paths = sorted(glob.glob(os.path.join(cfg["shards_dir"], "cl*.dat")))
        if not shard_paths:
            ln = "    [WARN] No shard .dat files found — skipping spread factor."
            print(ln); out.append(ln)
        else:
            shard_inv = []
            for sp in shard_paths:
                txs = load_transactions(sp)
                shard_inv.append(build_inverted_index(txs))
                print(f"    {os.path.basename(sp)}: {len(txs):,} transactions")
            spread_results = compute_spread_factors(gap, A, shard_inv)
            print_spread_table(spread_results, out)
    elif cfg["shards_dir"] is None:
        ln = "\n[4] Spread factor: SKIPPED (no raw shard .dat files for this partition)"
        print(ln); out.append(ln)

    # 5. Per-federated-file metrics
    print(f"\n[5] Processing federated results from: {cfg['feds_dir']}")
    fed_paths = sorted(glob.glob(os.path.join(cfg["feds_dir"], "*.txt")))
    if not fed_paths:
        ln = "    [WARN] No federated result files found."
        print(ln); out.append(ln)
        return

    gap_rows     = []   # (label, gap_recovery_dict)
    fidelity_rows = []  # (label, fidelity_dict)

    for fp in fed_paths:
        fname = os.path.basename(fp)
        if fname in cfg["skip_feds"]:
            ln = f"    [SKIP] {fname}  (marked as invalid data)"
            print(ln); out.append(ln)
            continue
        print(f"    Loading {fname} ...")
        B  = parse_codetable(fp)
        gr = compute_gap_recovery(gap, gap_target, B)
        fi = compute_fidelity(A, B)
        label = os.path.splitext(fname)[0]
        gap_rows.append((label, gr))
        fidelity_rows.append((label, fi))

    if gap_rows:
        print_gap_recovery_table(gap, gap_target, gap_rows, ct_union, out)
    if fidelity_rows:
        print_fidelity_table(fidelity_rows, out)


def main():
    all_lines = []
    for cfg in EXPERIMENTS:
        run_experiment(cfg, all_lines)

    # Write results to file
    out_path = os.path.join(BASE, "all_metrics_results.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(all_lines) + "\n")
    print(f"\n{'='*80}")
    print(f"Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
