#!/usr/bin/env python3
import argparse
import math
from collections import defaultdict

import numpy as np
import cooler
from .post_common import build_chrom_alias_map, _save_fig
import matplotlib.pyplot as plt

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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = stats[d]
    if x.ndim:
        s[0] += int(x.size)
        s[1] += float(x.sum())
        s[2] += float(y.sum())
        s[3] += float((x * x).sum())
        s[4] += float((y * y).sum())
        s[5] += float((x * y).sum())
    else:
        s[0] += 1
        s[1] += float(x)
        s[2] += float(y)
        s[3] += float(x * x)
        s[4] += float(y * y)
        s[5] += float(x * y)

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

def _box_sum_sparse(rows, cols, data, n, h, dtype=np.float64):
    """2-D box sum (window 2h+1) on a sparse upper-triangle matrix, without
    densifying. Implemented as two separable 1-D shifts; only offsets present
    in the sparse data are materialized, so it stays safe for large chromosomes.
    """
    from scipy import sparse
    if h == 0:
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def shift_dim(r, c, v, axis, delta):
        if axis == 0:
            r = r + delta
            ok = (r >= 0) & (r < n)
        else:
            c = c + delta
            ok = (c >= 0) & (c < n)
        return sparse.coo_matrix((v[ok], (r[ok], c[ok])), shape=(n, n))

    # horizontal pass (along columns)
    parts = [shift_dim(rows, cols, data, 1, db) for db in range(-h, h + 1)]
    H = sum(parts[1:], parts[0]).tocsr()
    hc = H.tocoo()
    # vertical pass (along rows)
    if hc.nnz == 0:
        return sparse.csr_matrix((n, n))
    parts = [shift_dim(hc.row, hc.col, hc.data, 0, da) for da in range(-h, h + 1)]
    return sum(parts[1:], parts[0]).tocsr()


def _upper_triangle(mat, n):
    """Canonical (row, col, data) arrays on the upper triangle (j >= i)."""
    mat = mat.tocoo()
    r, c, v = mat.row.astype(np.int64), mat.col.astype(np.int64), mat.data.astype(np.float64)
    flip = c < r
    if flip.any():
        r0, c0 = r[flip], c[flip]
        r[flip], c[flip] = c0, r0
    return r, c, v


def _diag_map(csr):
    # {offset k>=0: diagonal vector}; for k>=0 the first k slots of the dia
    # data row are padding, so the vector is row[k:n] (length n-k).
    dia = csr.todia()
    out = {}
    for j, k in enumerate(dia.offsets.tolist()):
        if k >= 0:
            row = dia.data[j]
            out[int(k)] = row[k:]
    return out


def _shared_chrom_data(clr_a, clr_b):
    """Read each shared chromosome's upper-triangle matrices once."""
    alias_a = build_chrom_alias_map(clr_a.chromnames)
    alias_b = build_chrom_alias_map(clr_b.chromnames)
    seen = set()
    shared = []
    for alias, chrom_a in alias_a.items():
        chrom_b = alias_b.get(alias)
        if chrom_b is None or (chrom_a, chrom_b) in seen:
            continue
        seen.add((chrom_a, chrom_b))
        shared.append((chrom_a, chrom_b))
    data = []
    for chrom_a, chrom_b in shared:
        mat_a = clr_a.matrix(balance=False, sparse=True).fetch(chrom_a)
        mat_b = clr_b.matrix(balance=False, sparse=True).fetch(chrom_b)
        n = min(mat_a.shape[0], mat_b.shape[0])
        if n <= 1:
            continue
        mat_a = mat_a.tocsr()[:n, :n]
        mat_b = mat_b.tocsr()[:n, :n]
        ra, ca, va = _upper_triangle(mat_a, n)
        rb, cb, vb = _upper_triangle(mat_b, n)
        data.append((n, ra, ca, va, rb, cb, vb))
    return data


def _stats_for_h(chrom_data, max_dist_bins, min_dist_bins, h):
    stats = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for n, ra, ca, va, rb, cb, vb in chrom_data:
        if h and h > 0:
            # HiCRep-style 2-D stratum smoothing. Each pixel is replaced by the
            # mean over its (2h+1)x(2h+1) neighbourhood, dividing by the number
            # of observed pixels in that window (per-matrix valid mask), so
            # unmapped positions are not filled with zeros.
            ma = np.ones(ra.size, dtype=np.float64)
            mb = np.ones(rb.size, dtype=np.float64)
            sx = _diag_map(_box_sum_sparse(ra, ca, va, n, h))
            cx = _diag_map(_box_sum_sparse(ra, ca, ma, n, h))
            sy = _diag_map(_box_sum_sparse(rb, cb, vb, n, h))
            cy = _diag_map(_box_sum_sparse(rb, cb, mb, n, h))
            for d in range(min_dist_bins, max_dist_bins + 1):
                if d not in sx or d not in sy:
                    continue
                xa, ca_d = sx[d], cx[d]
                yb, cb_d = sy[d], cy[d]
                L = min(xa.size, yb.size)
                xa, ca_d, yb, cb_d = xa[:L], ca_d[:L], yb[:L], cb_d[:L]
                ok = (ca_d > 0) & (cb_d > 0)
                x = xa[ok] / ca_d[ok]
                y = yb[ok] / cb_d[ok]
                finite = np.isfinite(x) & np.isfinite(y)
                x, y = x[finite], y[finite]
                if x.size:
                    _accumulate_stats(stats, d, x, y)
        else:
            dict_b = {}
            for i, j, v in zip(rb, cb, vb):
                d = int(j - i)
                if d < min_dist_bins or d > max_dist_bins:
                    continue
                dict_b[(int(i), int(j))] = float(v)
            seen_keys = set()
            for i, j, v in zip(ra, ca, va):
                d = int(j - i)
                if d < min_dist_bins or d > max_dist_bins:
                    continue
                key = (int(i), int(j))
                _accumulate_stats(stats, d, float(v), dict_b.get(key, 0.0))
                seen_keys.add(key)
            for (i, j), v in dict_b.items():
                if (i, j) in seen_keys:
                    continue
                d = j - i
                if d < min_dist_bins or d > max_dist_bins:
                    continue
                _accumulate_stats(stats, d, 0.0, float(v))
    return stats


def _aggregate_stats(stats):
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


def _scc_from_coolers(clr_a, clr_b, max_dist_bins, min_dist_bins, smooth_h=0):
    chrom_data = _shared_chrom_data(clr_a, clr_b)
    stats = _stats_for_h(chrom_data, max_dist_bins, min_dist_bins, smooth_h)
    return _aggregate_stats(stats)


def choose_smooth_h(chrom_data, max_dist_bins, min_dist_bins,
                    h_max=10, tol=0.01):
    """Pick h by the HiCRep plateau heuristic: scan SCC(h) for h=0..h_max and
    return the smallest h after which a further smoothing step changes the
    score by less than `tol` (the plateau onset). If no plateau is reached the
    largest h is used. Returns (h_star, curve, rows_by_h, std_by_h) where
    curve is [(h, scc), ...] and rows_by_h holds the per-stratum rows for each
    scanned h so the caller can reuse the chosen one.
    """
    curve, rows_by_h, std_by_h = [], {}, {}
    for h in range(0, int(h_max) + 1):
        rows, scc, std = _aggregate_stats(_stats_for_h(chrom_data, max_dist_bins, min_dist_bins, h))
        if np.isfinite(scc):
            curve.append((h, scc))
            rows_by_h[h] = rows
            std_by_h[h] = std
    if not curve:
        return 0, [], {}, {}
    if len(curve) == 1:
        h_star = curve[0][0]
    else:
        h_star = curve[-1][0]
        for k in range(1, len(curve)):
            if abs(curve[k][1] - curve[k - 1][1]) < tol:
                h_star = curve[k - 1][0]
                break
    return int(h_star), curve, rows_by_h, std_by_h

def main():
    p = argparse.ArgumentParser(
        prog="similarity",
        description="Compute a HiCRep-inspired SCC-like score between matched source-coordinate matrices "
                    "(typically Observed vs Target from lift2matrix): per-distance-stratum Pearson correlation, "
                    "aggregated weighted by stratum pixel count. By default no 2-D stratum smoothing is applied "
                    "(a simplified SCC-like statistic). Use --smooth-h h to enable HiCRep-style (2h+1)x(2h+1) "
                    "smoothing before the per-stratum correlations (bin units; the result depends on h, which is "
                    "recorded in the summary), or --smooth-h auto to pick h at the SCC plateau. "
                    "Inverse-variance weights are never used (weights = pixel counts).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--matrix-a", required=True, help="Observed matrix (lift2matrix outputs)")
    p.add_argument("--matrix-b", required=True, help="Target matrix (lift2matrix outputs)")
    p.add_argument("--max-dist-mb", type=float, default=10.0, help="Max genomic distance (Mb)")
    p.add_argument("--min-dist-bins", type=int, default=1, help="Min distance in bins to include")
    p.add_argument("--smooth-h", default="0",
        help="HiCRep-style 2-D stratum smoothing half-window in bins: an integer "
             "(0 = no smoothing, the simplified SCC-like default), or 'auto' to "
             "scan h and pick the SCC plateau onset (see --auto-h-max/--auto-h-tol)")
    p.add_argument("--auto-h-max", type=int, default=25,
        help="Maximum h scanned when --smooth-h auto")
    p.add_argument("--auto-h-tol", type=float, default=0.01,
        help="Plateau tolerance (|SCC(h)-SCC(h-1)| < tol) when --smooth-h auto")
    p.add_argument("--format", default="pdf", choices=["png","pdf","svg"], help="Plot format")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs")
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    args = p.parse_args()

    auto_mode = str(args.smooth_h).strip().lower() == "auto"
    if not auto_mode:
        try:
            smooth_h = int(args.smooth_h)
        except ValueError:
            raise SystemExit("--smooth-h must be a non-negative integer or 'auto'")
        if smooth_h < 0:
            raise SystemExit("--smooth-h must be >= 0")
    if args.auto_h_max < 1:
        raise SystemExit("--auto-h-max must be >= 1")

    clr_a = cooler.Cooler(args.matrix_a)
    clr_b = cooler.Cooler(args.matrix_b)
    if clr_a.binsize is None or clr_b.binsize is None:
        raise SystemExit("Input matrices must have fixed bin size.")
    if int(clr_a.binsize) != int(clr_b.binsize):
        raise SystemExit("Resolution mismatch: matrix-a and matrix-b must have the same bin size.")
    res = int(clr_a.binsize)
    max_dist_bins = int(args.max_dist_mb * 1e6 / res)

    chrom_data = _shared_chrom_data(clr_a, clr_b)
    curve = []
    if auto_mode:
        smooth_h, curve, rows_by_h, std_by_h = choose_smooth_h(
            chrom_data, max_dist_bins, args.min_dist_bins,
            h_max=args.auto_h_max, tol=args.auto_h_tol)
        rows = rows_by_h[smooth_h]
        std = std_by_h[smooth_h]
        scc = dict(curve)[smooth_h]
        print(f"[INFO] --smooth-h auto selected h={smooth_h} (plateau tol={args.auto_h_tol}, h_max={args.auto_h_max})")
    else:
        stats = _stats_for_h(chrom_data, max_dist_bins, args.min_dist_bins, smooth_h)
        rows, scc, std = _aggregate_stats(stats)

    out_tsv = f"{args.out_prefix}.scc-like.tsv"
    with open(out_tsv, "w") as f:
        f.write("dist_bins\tcount\tpearson_r\tweight\n")
        for d, n, r, w in rows:
            r_str = f"{r:.6f}" if np.isfinite(r) else "nan"
            w_str = f"{w:.6f}" if np.isfinite(w) else "nan"
            f.write(f"{d}\t{n}\t{r_str}\t{w_str}\n")

    out_summary = f"{args.out_prefix}.scc-like.summary.tsv"
    valid_rows = [(d, n, r, w) for d, n, r, w in rows if np.isfinite(r)]
    with open(out_summary, "w") as f:
        f.write("metric\tvalue\n")
        f.write(f"scc_like\t{scc:.6f}\n" if np.isfinite(scc) else "scc_like\tnan\n")
        f.write(f"std\t{std:.6f}\n" if np.isfinite(std) else "std\tnan\n")
        f.write(f"n_strata\t{len(valid_rows)}\n")
        f.write(f"total_weight\t{sum(w for _, _, _, w in valid_rows):.6f}\n")
        f.write(f"resolution_bp\t{res}\n")
        f.write(f"smooth_h_bins\t{smooth_h}\n")
        f.write(f"smooth_window_bins\t{2 * smooth_h + 1}\n")
        if auto_mode:
            f.write(f"smooth_h_selected\tauto\n")
            f.write(f"smooth_h_scan_max\t{args.auto_h_max}\n")
            f.write(f"smooth_h_plateau_tol\t{args.auto_h_tol}\n")
        f.write(f"min_dist_bins\t{args.min_dist_bins}\n")
        f.write(f"max_dist_bins\t{max_dist_bins}\n")
        f.write(f"max_dist_mb\t{args.max_dist_mb:.6f}\n")

    if auto_mode and curve:
        curve_tsv = f"{args.out_prefix}.scc-h.tsv"
        with open(curve_tsv, "w") as f:
            f.write("smooth_h\tscc_like\tselected\n")
            for h, val in curve:
                f.write(f"{h}\t{val:.6f}\t{1 if h == smooth_h else 0}\n")
        print(f"[OK] {curve_tsv}")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ds = [d for d, _, r, _ in rows if np.isfinite(r)]
    rs = [r for _, _, r, _ in rows if np.isfinite(r)]
    if rs:
        ax.plot(np.array(ds) * res / 1e6, rs, lw=1.2, color="#f1594f", alpha=0.85)
    # Axis titles one step larger than the tick labels (default 10 pt).
    label_fs = plt.rcParams["font.size"] + 2
    ax.set_xlabel("genomic distance (Mb)", fontsize=label_fs)
    ax.set_ylabel("stratum Pearson r", fontsize=label_fs)
    title = f"SCC-like = {scc:.4f}"
    if smooth_h > 0:
        title += f"  (smoothing h={smooth_h}{', auto' if auto_mode else ''})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig_path = f"{args.out_prefix}.scc-like.{args.format}"
    _save_fig(fig, fig_path, fmt=args.format, dpi=args.dpi)
    plt.close(fig)
    print(f"[OK] {out_tsv}")
    print(f"[OK] {out_summary}")
    print(f"[OK] {fig_path}")

if __name__ == "__main__":
    main()


