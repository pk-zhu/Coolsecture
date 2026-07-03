#!/usr/bin/env python3
import argparse, os
import numpy as np
from .post_common import (
    infer_resolution_from_liftover, ChromIndexingFAI, read_contacts,
    extract_arrays_for_contact_stat, _log2_ratio_safe,
    make_pinkblue_cmap, make_diverging_norm, _save_fig, resolve_liftover_inputs
)
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(
        prog="contact-stat",
        description=("Generate three diagnostics from a .liftContacts file:\n"
            "  1) Percentile heatmap (observed vs. target percentile scores)\n"
            "  2) Distance heatmap (source vs. target genomic distances for lifted-over contacts)\n"
            "  3) Ratio scatter (genomic distance vs. percentile score)"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--liftover", help="Path to .liftContacts file")
    p.add_argument("--liftover-prefix", help="Prefix to batch process .liftContacts (e.g., out/A_B)")
    p.add_argument("--fadix", required=True, help="Path to FASTA index (.fai) file")
    p.add_argument("--stats-a", help="Optional species-A step1 stats.tsv (distance-stratified percentile->score map)")
    p.add_argument("--stats-b", help="Optional species-B step1 stats.tsv (distance-stratified percentile->score map)")
    p.add_argument("--min-bins", type=int, default=0,
        help="Lower bound on intra-chromosomal bin distance d (open): keep pairs with d > min-bins")
    p.add_argument("--max-bins", type=int, default=20, 
        help="Upper bound on intra-chromosomal bin distance d (closed): keep pairs with d ≤ max-bins")
    p.add_argument("--repeats", type=int, default=5, help="Number of randomization repeats for null/target estimation")
    p.add_argument("--bins", type=int, default=100, help="Number of bins for 2D histograms")
    p.add_argument("--cmap", default="pinkblue", choices=["pinkblue","RdBu_r","coolwarm","PuOr_r","viridis"], 
        help="Colormap for heatmaps")
    p.add_argument("--vmax", type=float, default=None,
        help="Symmetric color limit for diverging maps (|v| ≤ vmax); if None, use 99th percentile")
    p.add_argument("--max-dist-mb", type=float, default=5.0, help="Axis limit (Mb) for the distance heatmap")
    p.add_argument("--format", default="pdf", choices=["pdf","png","svg"], help="Figure format (default: pdf)")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs")
    p.add_argument("--xy-range", type=float, default=20.0, help="Axis range for distance/percentile ratio scatter")
    p.add_argument("--ratio-scatter-standard", default="c-intersecture", choices=["c-intersecture", "modern"],
        help="Ratio-scatter algorithm standard: C-InterSecture-compatible legacy mode or modern mode")
    p.add_argument("--ratio-y-mode", default="auto", choices=["auto", "percentile", "score"],
        help="Ratio-scatter y-axis source: percentile ratio or score ratio (requires --stats-a/--stats-b)")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible randomization")
    p.add_argument("--label-a", default="species A", help="Y-axis label (A)")
    p.add_argument("--label-b", default="species B", help="X-axis label (B)")
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    args = p.parse_args()
    if args.seed is not None:
        np.random.seed(int(args.seed))

    def _adaptive_bins(base_bins: int, n_points: int, lower: int = 40, upper: int = None) -> int:
        base = max(int(base_bins), 1)
        n = max(int(n_points), 1)
        tgt = int(np.sqrt(float(n)) * 1.2)
        bins = max(lower, min(base, tgt))
        if upper is not None:
            bins = min(int(upper), bins)
        return max(1, int(bins))

    def _adaptive_cap(values: np.ndarray, user_cap: float, floor: float = 5.0) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr) & (arr >= 0)]
        if arr.size == 0:
            cap = float(user_cap) if (user_cap is not None and np.isfinite(user_cap) and user_cap > 0) else floor
            return max(float(floor), cap)
        q995 = float(np.nanpercentile(arr, 99.5))
        if (not np.isfinite(q995)) or q995 <= 0:
            q995 = float(np.nanmax(arr))
        data_cap = max(float(floor), q995 * 1.15)
        if user_cap is None or (not np.isfinite(user_cap)) or user_cap <= 0:
            return data_cap
        return min(float(user_cap), data_cap)

    def mean_hist2d_random(x_obs, y_obs, bins, hist_range, repeats, pair_y=None, independent_pair=False):
        if x_obs.size == 0 or y_obs.size == 0:
            return np.zeros((bins, bins), dtype=float)
        repeats = max(int(repeats), 1)
        acc = np.zeros((bins, bins), dtype=float)
        x_obs = np.asarray(x_obs, dtype=float)
        y_obs = np.asarray(y_obs, dtype=float)
        if pair_y is not None:
            pair_y = np.asarray(pair_y, dtype=float)
        for _ in range(repeats):
            perm = np.random.permutation(x_obs.size)
            x_rnd = x_obs[perm]
            if pair_y is None:
                y_rnd = y_obs
            else:
                if independent_pair:
                    y_rnd = pair_y[np.random.permutation(pair_y.size)]
                else:
                    y_rnd = pair_y[perm]
            H, _, _ = np.histogram2d(x_rnd, y_rnd, bins=(bins, bins), range=hist_range, density=True)
            acc += H
        return acc / repeats

    def _ratio_arrays_legacy(short_list, resolution):
        # C-InterSecture-compatible path: use short-list distances and independent randomization of
        # target percentile and target distance in null model.
        pA = []
        pB = []
        dA = []
        dB = []
        for c, q, lr, lq in short_list:
            try:
                # Keep legacy ratio-scatter distance axis in bin units (C-InterSecture behavior).
                da = float(lr)
                if float(resolution) <= 0:
                    continue
                db = float(lq) / float(resolution)
                ca = float(c)
                cb = float(q)
            except Exception:
                continue
            if (not np.isfinite(da)) or (not np.isfinite(db)) or da <= 0 or db <= 0:
                continue
            if (not np.isfinite(ca)) or (not np.isfinite(cb)) or ca < 0 or cb < 0:
                continue
            pA.append(ca); pB.append(cb); dA.append(da); dB.append(db)
        return (
            np.asarray(pA, dtype=float),
            np.asarray(pB, dtype=float),
            np.asarray(dA, dtype=float),
            np.asarray(dB, dtype=float),
        )

    def _ratio_arrays_legacy_from_extracted(pA_arr, pB_arr, dA_mb_arr, dB_mb_arr, resolution):
        if float(resolution) <= 0:
            return (
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
            )
        dA_bins = np.asarray(dA_mb_arr, dtype=float) * 1e6 / float(resolution)
        dB_bins = np.asarray(dB_mb_arr, dtype=float) * 1e6 / float(resolution)
        pA_arr = np.asarray(pA_arr, dtype=float)
        pB_arr = np.asarray(pB_arr, dtype=float)
        mask = (
            np.isfinite(pA_arr) & np.isfinite(pB_arr) &
            np.isfinite(dA_bins) & np.isfinite(dB_bins) &
            (pA_arr >= 0) & (pB_arr >= 0) &
            (dA_bins > 0) & (dB_bins > 0)
        )
        return (
            pA_arr[mask],
            pB_arr[mask],
            dA_bins[mask],
            dB_bins[mask],
        )

    def _ratio_arrays_modern(pA, pB, dA, dB, ok_dist, max_dist_mb):
        mask = ok_dist & (dA >= 0) & (dB >= 0)
        return (
            np.asarray(pA[mask], dtype=float),
            np.asarray(pB[mask], dtype=float),
            np.asarray(np.clip(dA[mask], 0, max_dist_mb), dtype=float),
            np.asarray(np.clip(dB[mask], 0, max_dist_mb), dtype=float),
        )

    def _load_score_lookup(path):
        if (not path) or (not os.path.exists(path)):
            return None
        table = {}
        with open(path) as f:
            _ = f.readline()
            for line in f:
                a = line.strip().split()
                if len(a) < 6:
                    continue
                try:
                    dist_bin = int(float(a[0]))
                except Exception:
                    continue
                vals = []
                if len(a) >= 105:
                    cand = a[5:105]
                elif ';' in a[5]:
                    cand = [x for x in a[5].split(';') if x]
                else:
                    cand = a[5:]
                for tok in cand:
                    try:
                        vals.append(float(tok))
                    except Exception:
                        continue
                    if len(vals) >= 100:
                        break
                if len(vals) < 100:
                    continue
                arr = np.asarray(vals[:100], dtype=float)
                if np.any(np.isfinite(arr)):
                    table[dist_bin] = arr
        if not table:
            return None
        keys = np.asarray(sorted(table.keys()), dtype=int)
        return {"table": table, "keys": keys}

    def _score_from_percentile(lookup, dist_bins, pct):
        if lookup is None:
            return np.nan
        if (not np.isfinite(dist_bins)) or dist_bins <= 0:
            return np.nan
        keys = lookup["keys"]
        if keys.size == 0:
            return np.nan
        d = int(np.floor(float(dist_bins)))
        idx = int(np.searchsorted(keys, d, side='right') - 1)
        if idx < 0:
            idx = 0
        key = int(keys[idx])
        row = lookup["table"].get(key)
        if row is None:
            return np.nan
        try:
            p = int(round(float(pct)))
        except Exception:
            return np.nan
        p = max(1, min(100, p))
        v = float(row[p - 1])
        if (not np.isfinite(v)) or v <= 0:
            return np.nan
        return v

    def _ratio_arrays_legacy_score(short_list, resolution, score_lookup_a, score_lookup_b):
        sA = []
        sB = []
        dA = []
        dB = []
        for c, q, lr, lq in short_list:
            try:
                da = float(lr)
                if float(resolution) <= 0:
                    continue
                db = float(lq) / float(resolution)
                ca = float(c)
                cb = float(q)
            except Exception:
                continue
            if (not np.isfinite(da)) or (not np.isfinite(db)) or da <= 0 or db <= 0:
                continue
            if (not np.isfinite(ca)) or (not np.isfinite(cb)):
                continue
            va = _score_from_percentile(score_lookup_a, da, ca)
            vb = _score_from_percentile(score_lookup_b, db, cb)
            if (not np.isfinite(va)) or (not np.isfinite(vb)) or va <= 0 or vb <= 0:
                continue
            sA.append(va)
            sB.append(vb)
            dA.append(da)
            dB.append(db)
        return (
            np.asarray(sA, dtype=float),
            np.asarray(sB, dtype=float),
            np.asarray(dA, dtype=float),
            np.asarray(dB, dtype=float),
        )

    def _ratio_arrays_legacy_score_from_extracted(
        pA_arr, pB_arr, dA_mb_arr, dB_mb_arr, resolution, score_lookup_a, score_lookup_b
    ):
        if float(resolution) <= 0:
            return (
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
            )
        pA_arr = np.asarray(pA_arr, dtype=float)
        pB_arr = np.asarray(pB_arr, dtype=float)
        dA_bins = np.asarray(dA_mb_arr, dtype=float) * 1e6 / float(resolution)
        dB_bins = np.asarray(dB_mb_arr, dtype=float) * 1e6 / float(resolution)

        sA = []
        sB = []
        oA = []
        oB = []
        for ca, cb, da, db in zip(pA_arr, pB_arr, dA_bins, dB_bins):
            if (not np.isfinite(ca)) or (not np.isfinite(cb)):
                continue
            if (not np.isfinite(da)) or (not np.isfinite(db)) or da <= 0 or db <= 0:
                continue
            va = _score_from_percentile(score_lookup_a, da, ca)
            vb = _score_from_percentile(score_lookup_b, db, cb)
            if (not np.isfinite(va)) or (not np.isfinite(vb)) or va <= 0 or vb <= 0:
                continue
            sA.append(va)
            sB.append(vb)
            oA.append(da)
            oB.append(db)
        return (
            np.asarray(sA, dtype=float),
            np.asarray(sB, dtype=float),
            np.asarray(oA, dtype=float),
            np.asarray(oB, dtype=float),
        )

    def run_one(liftover_path: str, out_prefix: str):
        res = infer_resolution_from_liftover(liftover_path)
        Order = ChromIndexingFAI(args.fadix)

        pA, pB, dA, dB, ok_dist = extract_arrays_for_contact_stat(liftover_path, res, Order)
        short_list = []

        use_extracted = pA.size > 0 and pB.size > 0
        if not use_extracted:
            short_list = read_contacts(liftover_path, Order, res, short=True)

        if use_extracted and float(res) > 0:
            dA_bins = np.asarray(dA, dtype=float) * 1e6 / float(res)
            mask_base = (
                np.isfinite(pA) & np.isfinite(pB) & np.isfinite(dA_bins) &
                (pA >= 0) & (pA <= 100) & (pB >= 0) & (pB <= 100) &
                (dA_bins > args.min_bins)
            )
            mask_window = mask_base & (dA_bins <= args.max_bins)
            PA = np.asarray(pA[mask_window], dtype=float)
            PB = np.asarray(pB[mask_window], dtype=float)
            if PA.size < 100:
                PA2 = np.asarray(pA[mask_base], dtype=float)
                PB2 = np.asarray(pB[mask_base], dtype=float)
                if PA2.size > PA.size:
                    print(
                        f"[INFO] percentile fallback: points in ({args.min_bins},{args.max_bins}] bins are sparse "
                        f"({PA.size}); widened to all intra-chromosomal bins (> {args.min_bins}) -> {PA2.size} points."
                    )
                    PA, PB = PA2, PB2
        else:
            PA, PB = [], []
            for c, q, lr, _ in short_list:
                if args.min_bins < lr <= args.max_bins:
                    PA.append(c); PB.append(q)
            PA = np.asarray(PA, dtype=float)
            PB = np.asarray(PB, dtype=float)
            if PA.size < 100:
                PA2, PB2 = [], []
                for c, q, lr, _ in short_list:
                    if lr > args.min_bins:
                        PA2.append(c); PB2.append(q)
                if len(PA2) > PA.size:
                    print(
                        f"[INFO] percentile fallback: points in ({args.min_bins},{args.max_bins}] bins are sparse "
                        f"({PA.size}); widened to all intra-chromosomal bins (> {args.min_bins}) -> {len(PA2)} points."
                    )
                    PA = np.asarray(PA2, dtype=float)
                    PB = np.asarray(PB2, dtype=float)

        if PA.size == 0 or PB.size == 0:
            print("[WARN] percentile heatmap has no valid points after filtering; using all finite [0,100] points.")
            if use_extracted:
                m = np.isfinite(pA) & np.isfinite(pB) & (pA >= 0) & (pA <= 100) & (pB >= 0) & (pB <= 100)
                PA = np.asarray(pA[m], dtype=float)
                PB = np.asarray(pB[m], dtype=float)
            else:
                PA = np.asarray([c for c, _, _, _ in short_list if np.isfinite(c) and 0 <= c <= 100], dtype=float)
                PB = np.asarray([q for _, q, _, _ in short_list if np.isfinite(q) and 0 <= q <= 100], dtype=float)

        bins_pct = _adaptive_bins(args.bins, PA.size, lower=40, upper=100)
        if PA.size > 0 and PB.size > 0:
            Hobs, _, _ = np.histogram2d(PB, PA, bins=(bins_pct, bins_pct), range=[(0,100),(0,100)], density=True)
            Hran = mean_hist2d_random(PB, PA, bins_pct, [(0,100),(0,100)], args.repeats)
            M = _log2_ratio_safe(Hobs, Hran)
            M[(Hobs == 0) & (Hran == 0)] = np.nan
        else:
            M = np.full((bins_pct, bins_pct), np.nan, dtype=float)

        fig, ax = plt.subplots(figsize=(5.5,5.2))
        cmap = make_pinkblue_cmap() if args.cmap == 'pinkblue' else plt.get_cmap(args.cmap)
        norm = make_diverging_norm(M, vcenter=0.0, vmax=args.vmax)
        im = ax.imshow(M.T, origin='lower', extent=[0,100,0,100], cmap=cmap, norm=norm)
        fig.canvas.draw()
        bbox = ax.get_position()
        w = 0.02
        pad = 0.12
        cax = fig.add_axes([bbox.x1 + pad, bbox.y0, w, bbox.height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Log2(observed/random)", rotation=270, labelpad=14)
        cax.yaxis.set_ticks_position('right')
        cax.yaxis.set_label_position('right')
        ax.set_xlabel(f"{args.label_b} percentile score")
        ax.set_ylabel(f"{args.label_a} percentile score")
        ax.set_xlim(0,100); ax.set_ylim(0,100)
        ax.set_xticks(range(0,101,10)); ax.set_yticks(range(0,101,10))
        _save_fig(fig, f"{out_prefix}.percentile_heatmap.{args.format}", fmt=args.format, dpi=args.dpi)
        plt.close(fig)

        # Distance heatmap: 2D histogram of source vs. target genomic distances
        # for successfully lifted-over contact pairs, colored by log2(observed/random).
        maskA = ok_dist & (dA >= 0) & (dB >= 0)
        dA_raw = np.asarray(dA[maskA], dtype=float)
        dB_raw = np.asarray(dB[maskA], dtype=float)
        cap_a = _adaptive_cap(dA_raw, args.max_dist_mb, floor=5.0)
        cap_b = _adaptive_cap(dB_raw, args.max_dist_mb, floor=5.0)
        # Use unified cap for both axes to make x and y limits the same
        unified_cap = max(cap_a, cap_b)
        dA_use = np.clip(dA_raw, 0, unified_cap)
        dB_use = np.clip(dB_raw, 0, unified_cap)
        bins_dist = _adaptive_bins(args.bins, min(dA_use.size, dB_use.size), lower=60, upper=320)
        Hobs2, _, _ = np.histogram2d(
            dB_use, dA_use,
            bins=(bins_dist, bins_dist),
            range=[(0, unified_cap), (0, unified_cap)],
            density=True
        )
        Hran2 = mean_hist2d_random(dB_use, dA_use, bins_dist, [(0, unified_cap), (0, unified_cap)], args.repeats)
        M2 = _log2_ratio_safe(Hobs2, Hran2)
        M2[(Hobs2 == 0) & (Hran2 == 0)] = np.nan
        print(
            f"[INFO] distance_heatmap points={dA_use.size}, bins={bins_dist}, "
            f"unified_cap_mb={unified_cap:.2f} (a={cap_a:.2f}, b={cap_b:.2f})"
        )

        fig2, ax2 = plt.subplots(figsize=(5.5,5.2))
        cmap2 = make_pinkblue_cmap() if args.cmap == 'pinkblue' else plt.get_cmap(args.cmap)
        norm2 = make_diverging_norm(M2, vcenter=0.0, vmax=args.vmax)
        im2 = ax2.imshow(M2.T, origin='lower', extent=[0, unified_cap, 0, unified_cap], cmap=cmap2, norm=norm2)
        fig2.canvas.draw()
        bbox2 = ax2.get_position()
        w2 = 0.02
        pad2 = 0.12
        cax2 = fig2.add_axes([bbox2.x1 + pad2, bbox2.y0, w2, bbox2.height])
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.set_label("Log2(observed/random)", rotation=270, labelpad=14)
        cax2.yaxis.set_ticks_position('right')
        cax2.yaxis.set_label_position('right')
        ax2.set_xlabel(f"{args.label_b} genomic distance, Mb")
        ax2.set_ylabel(f"{args.label_a} genomic distance, Mb")
        ax2.set_xlim(0, unified_cap); ax2.set_ylim(0, unified_cap)
        _save_fig(fig2, f"{out_prefix}.distance_heatmap.{args.format}", fmt=args.format, dpi=args.dpi)
        plt.close(fig2)

        if args.ratio_scatter_standard == "c-intersecture":
            score_mode = False
            pA_ratio = pB_ratio = dA_ratio = dB_ratio = np.asarray([], dtype=float)
            score_lookup_a = _load_score_lookup(args.stats_a)
            score_lookup_b = _load_score_lookup(args.stats_b)
            prefer_score = (args.ratio_y_mode in ("auto", "score"))
            if prefer_score and (score_lookup_a is not None) and (score_lookup_b is not None):
                if use_extracted:
                    pA_ratio, pB_ratio, dA_ratio, dB_ratio = _ratio_arrays_legacy_score_from_extracted(
                        pA, pB, dA, dB, res, score_lookup_a, score_lookup_b
                    )
                else:
                    pA_ratio, pB_ratio, dA_ratio, dB_ratio = _ratio_arrays_legacy_score(
                        short_list, res, score_lookup_a, score_lookup_b
                    )
                if pA_ratio.size > 0:
                    score_mode = True
                else:
                    print("[WARN] score-based ratio-scatter produced no valid points; falling back to percentile.")
            elif args.ratio_y_mode == "score":
                print("[WARN] --ratio-y-mode score requested but --stats-a/--stats-b is missing; using percentile mode.")
            if not score_mode:
                if use_extracted:
                    pA_ratio, pB_ratio, dA_ratio, dB_ratio = _ratio_arrays_legacy_from_extracted(pA, pB, dA, dB, res)
                else:
                    pA_ratio, pB_ratio, dA_ratio, dB_ratio = _ratio_arrays_legacy(short_list, res)
            independent_random = True
        else:
            score_mode = False
            pA_ratio, pB_ratio, dA_ratio, dB_ratio = _ratio_arrays_modern(pA, pB, dA, dB, ok_dist, args.max_dist_mb)
            independent_random = False

        eps = 1e-6
        logR_dist = _log2_ratio_safe(dA_ratio + eps, dB_ratio + eps)
        logR_pct = _log2_ratio_safe(pA_ratio + eps, pB_ratio + eps)
        rng = args.xy_range
        bins_ratio = _adaptive_bins(args.bins, min(logR_dist.size, logR_pct.size), lower=80, upper=320)
        if logR_dist.size > 0 and logR_pct.size > 0:
            Hobs3, _, _ = np.histogram2d(logR_dist, logR_pct, bins=(bins_ratio, bins_ratio), range=[(-rng, rng), (-rng, rng)], density=True)
            repeats = max(int(args.repeats), 1)
            Hran3 = np.zeros((bins_ratio, bins_ratio), dtype=float)
            if dB_ratio.size > 0 and pB_ratio.size > 0:
                for _ in range(repeats):
                    if independent_random:
                        dB_rnd = dB_ratio[np.random.permutation(dB_ratio.size)]
                        pB_rnd = pB_ratio[np.random.permutation(pB_ratio.size)]
                    else:
                        perm = np.random.permutation(dB_ratio.size)
                        dB_rnd = dB_ratio[perm]
                        pB_rnd = pB_ratio[perm]
                    logR_dist_rnd = _log2_ratio_safe(dA_ratio + eps, dB_rnd + eps)
                    logR_pct_rnd = _log2_ratio_safe(pA_ratio + eps, pB_rnd + eps)
                    Htmp, _, _ = np.histogram2d(
                        logR_dist_rnd, logR_pct_rnd,
                        bins=(bins_ratio, bins_ratio),
                        range=[(-rng, rng), (-rng, rng)],
                        density=True,
                    )
                    Hran3 += Htmp
                Hran3 /= repeats
            pos3 = np.concatenate([Hobs3[Hobs3 > 0], Hran3[Hran3 > 0]])
            if pos3.size:
                eps3 = max(float(np.quantile(pos3, 0.01)) * 0.5, 1e-12)
            else:
                eps3 = 1e-12
            M3 = np.log2((Hobs3 + eps3) / (Hran3 + eps3))
            M3[(Hobs3 == 0) & (Hran3 == 0)] = np.nan
        else:
            print("[WARN] ratio-scatter has no valid points after filtering; output will be empty.")
            M3 = np.full((bins_ratio, bins_ratio), np.nan, dtype=float)

        fig3, ax3 = plt.subplots(figsize=(5.5,5.2))
        cmap3 = make_pinkblue_cmap() if args.cmap == 'pinkblue' else plt.get_cmap(args.cmap)
        vmax3 = args.vmax
        if vmax3 is None:
            finite_abs = np.abs(M3[np.isfinite(M3)])
            if finite_abs.size:
                vmax3 = min(float(np.quantile(finite_abs, 0.99)), 6.0)
            else:
                vmax3 = 1.0
        norm3 = make_diverging_norm(M3, vcenter=0.0, vmax=vmax3)
        im3 = ax3.imshow(M3.T, origin='lower', extent=[-rng, rng, -rng, rng], cmap=cmap3, norm=norm3)
        fig3.canvas.draw()
        bbox3 = ax3.get_position()
        w3 = 0.02
        pad3 = 0.12
        cax3 = fig3.add_axes([bbox3.x1 + pad3, bbox3.y0, w3, bbox3.height])
        cbar3 = plt.colorbar(im3, cax=cax3)
        cbar3.set_label("Log2(observed/random)", rotation=270, labelpad=14)
        cax3.yaxis.set_ticks_position('right')
        cax3.yaxis.set_label_position('right')
        ax3.set_xlabel(f"log2({args.label_a} contact distance / {args.label_b} contact distance)")
        if score_mode:
            ax3.set_ylabel(f"log2({args.label_a} contact frequency / {args.label_b} contact frequency)")
        else:
            ax3.set_ylabel(f"log2({args.label_a} percentile score / {args.label_b} percentile score)")
        ax3.set_xlim(-rng, rng)
        ax3.set_ylim(-rng, rng)
        _save_fig(fig3, f"{out_prefix}.ratio_scatter.{args.format}", fmt=args.format, dpi=args.dpi)
        plt.close(fig3)

    for liftover_path, suffix in resolve_liftover_inputs(args.liftover, args.liftover_prefix):
        run_one(liftover_path, args.out_prefix + suffix)

if __name__ == "__main__":
    main()
