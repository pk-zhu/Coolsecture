#!/usr/bin/env python3
import argparse
import math
from collections import defaultdict

import numpy as np
import cooler
import matplotlib.pyplot as plt

from .post_common import build_chrom_alias_map

def _pearson_from_stats(n, sx, sy, sxx, syy, sxy):
    if n < 2:
        return float("nan")
    vx = sxx - (sx * sx) / n
    vy = syy - (sy * sy) / n
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sxy - (sx * sy) / n
    return cov / math.sqrt(vx * vy)

def _accumulate_stats(stats, d, x, y):
    s = stats[d]
    s[0] += 1
    s[1] += x
    s[2] += y
    s[3] += x * x
    s[4] += y * y
    s[5] += x * y

def _std_from_weighted_corr(rows):
    finite = [(r, w) for _, _, r, w in rows if np.isfinite(r) and np.isfinite(w) and w > 0]
    if len(finite) < 2:
        return float("nan")
    corr = np.array([x[0] for x in finite], dtype=float)
    wei = np.array([x[1] for x in finite], dtype=float)
    var_corr = float(np.var(corr, ddof=1))
    denom = float(np.sum(wei)) ** 2
    if denom <= 0:
        return float("nan")
    return math.sqrt(float(np.sum((wei ** 2) * var_corr)) / denom)

def _scc_from_coolers(clr_a, clr_b, max_dist_bins, min_dist_bins):
    stats = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0])
    alias_a = build_chrom_alias_map(clr_a.chromnames)
    alias_b = build_chrom_alias_map(clr_b.chromnames)
    shared = []
    for alias, chrom_a in alias_a.items():
        chrom_b = alias_b.get(alias)
        if chrom_b is None:
            continue
        if (chrom_a, chrom_b) not in shared:
            shared.append((chrom_a, chrom_b))
    for chrom_a, chrom_b in shared:
        mat_a = clr_a.matrix(balance=False, sparse=True).fetch(chrom_a).tocoo()
        mat_b = clr_b.matrix(balance=False, sparse=True).fetch(chrom_b).tocoo()
        if mat_a.shape != mat_b.shape:
            n = min(mat_a.shape[0], mat_b.shape[0])
            if n <= 1:
                continue
            mat_a = mat_a.tocsr()[:n, :n].tocoo()
            mat_b = mat_b.tocsr()[:n, :n].tocoo()
        dict_b = {}
        for i, j, v in zip(mat_b.row, mat_b.col, mat_b.data):
            if j < i:
                i, j = j, i
            d = j - i
            if d < min_dist_bins or d > max_dist_bins:
                continue
            dict_b[(i, j)] = float(v)
        seen = set()
        for i, j, v in zip(mat_a.row, mat_a.col, mat_a.data):
            if j < i:
                i, j = j, i
            d = j - i
            if d < min_dist_bins or d > max_dist_bins:
                continue
            key = (i, j)
            y = dict_b.get(key, 0.0)
            _accumulate_stats(stats, d, float(v), y)
            seen.add(key)
        for (i, j), v in dict_b.items():
            if (i, j) in seen:
                continue
            d = j - i
            if d < min_dist_bins or d > max_dist_bins:
                continue
            _accumulate_stats(stats, d, 0.0, float(v))
    # compute per-stratum r and SCC
    rows = []
    for d, (n, sx, sy, sxx, syy, sxy) in stats.items():
        r = _pearson_from_stats(n, sx, sy, sxx, syy, sxy)
        w = float(n) if np.isfinite(r) else float("nan")
        rows.append((d, n, r, w))
    rows.sort(key=lambda x: x[0])
    weights = [w for _, _, r, w in rows if np.isfinite(r) and np.isfinite(w) and w > 0]
    rvals = [r for _, _, r, w in rows if np.isfinite(r) and np.isfinite(w) and w > 0]
    if weights:
        scc = float(np.average(rvals, weights=weights))
    else:
        scc = float("nan")
    std = _std_from_weighted_corr(rows)
    return rows, scc, std

def main():
    p = argparse.ArgumentParser(
        prog="cross-validate",
        description="Compute stratum-adjusted correlation coefficient (SCC) between matched source-coordinate matrices (typically Observed vs Target from lift2matrix)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--matrix-a", required=True, help="Observed matrix (lift2matrix outputs)")
    p.add_argument("--matrix-b", required=True, help="Target matrix (lift2matrix outputs)")
    p.add_argument("--max-dist-mb", type=float, default=10.0, help="Max genomic distance (Mb)")
    p.add_argument("--min-dist-bins", type=int, default=1, help="Min distance in bins to include")
    p.add_argument("--format", default="pdf", choices=["png","pdf","svg"], help="Plot format")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs")
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    args = p.parse_args()

    clr_a = cooler.Cooler(args.matrix_a)
    clr_b = cooler.Cooler(args.matrix_b)
    if clr_a.binsize is None or clr_b.binsize is None:
        raise SystemExit("Input matrices must have fixed bin size.")
    if int(clr_a.binsize) != int(clr_b.binsize):
        raise SystemExit("Resolution mismatch: matrix-a and matrix-b must have the same bin size.")
    res = int(clr_a.binsize)
    max_dist_bins = int(args.max_dist_mb * 1e6 / res)

    rows, scc, std = _scc_from_coolers(clr_a, clr_b, max_dist_bins, args.min_dist_bins)

    out_tsv = f"{args.out_prefix}.scc.tsv"
    with open(out_tsv, "w") as f:
        f.write("dist_bins\tcount\tpearson_r\tweight\n")
        for d, n, r, w in rows:
            r_str = f"{r:.6f}" if np.isfinite(r) else "nan"
            w_str = f"{w:.6f}" if np.isfinite(w) else "nan"
            f.write(f"{d}\t{n}\t{r_str}\t{w_str}\n")

    out_summary = f"{args.out_prefix}.scc.summary.tsv"
    valid_rows = [(d, n, r, w) for d, n, r, w in rows if np.isfinite(r)]
    with open(out_summary, "w") as f:
        f.write("metric\tvalue\n")
        f.write(f"scc\t{scc:.6f}\n" if np.isfinite(scc) else "scc\tnan\n")
        f.write(f"std\t{std:.6f}\n" if np.isfinite(std) else "std\tnan\n")
        f.write(f"n_strata\t{len(valid_rows)}\n")
        f.write(f"total_weight\t{sum(w for _, _, _, w in valid_rows):.6f}\n")
        f.write(f"resolution_bp\t{res}\n")
        f.write(f"min_dist_bins\t{args.min_dist_bins}\n")
        f.write(f"max_dist_bins\t{max_dist_bins}\n")
        f.write(f"max_dist_mb\t{args.max_dist_mb:.6f}\n")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ds = [d for d, _, r, _ in rows if np.isfinite(r)]
    rs = [r for _, _, r, _ in rows if np.isfinite(r)]
    if rs:
        ax.plot(np.array(ds) * res / 1e6, rs, lw=1.2, color="#f1594f", alpha=0.85)
    ax.set_xlabel("genomic distance (Mb)")
    ax.set_ylabel("stratum Pearson r")
    ax.set_title(f"SCC = {scc:.4f}")
    ax.grid(True, alpha=0.3)
    fig_path = f"{args.out_prefix}.scc.{args.format}"
    if args.format == "pdf":
        fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    else:
        fig.savefig(fig_path, format=args.format, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_tsv}")
    print(f"[OK] {out_summary}")
    print(f"[OK] {fig_path}")

if __name__ == "__main__":
    main()


