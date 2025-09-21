#!/usr/bin/env python3
import argparse
import numpy as np
from .post_common import (
    infer_resolution_from_liftover, ChromIndexingFAI, read_contacts,
    extract_arrays_for_contact_stat, randomize_vector, _log2_ratio_safe,
    make_pinkblue_cmap, make_diverging_norm, _save_fig
)
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(
        prog="contact-stat",
        description=("Generate three diagnostics from a .liftContacts file:\n"
            "  1) Percentile heatmap (obs vs. ctrl)\n"
            "  2) Distance heatmap (contact strength vs. genomic distance)\n"
            "  3) Ratio scatter (A vs. B)"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--liftover", required=True, help="Path to .liftContacts file")
    p.add_argument("--fadix", required=True, help="Path to FASTA index (.fai) file")
    p.add_argument("--min-bins", type=int, default=0,
        help="Lower bound on intra-chromosomal bin distance d (open): keep pairs with d > min-bins")
    p.add_argument("--max-bins", type=int, default=20, 
        help="Upper bound on intra-chromosomal bin distance d (closed): keep pairs with d ≤ max-bins")
    p.add_argument("--repeats", type=int, default=1, help="Number of randomization repeats for null/control estimation")
    p.add_argument("--bins", type=int, default=100, help="Number of bins for 2D histograms")
    p.add_argument("--cmap", default="pinkblue", choices=["pinkblue","RdBu_r","coolwarm","PuOr_r","viridis"], 
        help="Colormap for heatmaps")
    p.add_argument("--vmax", type=float, default=None,
        help="Symmetric color limit for diverging maps (|v| ≤ vmax); if None, use 99th percentile")
    p.add_argument("--max-dist-mb", type=float, default=5.0, help="Axis limit (Mb) for the distance heatmap")
    p.add_argument("--format", default="pdf", choices=["pdf","png","svg"], help="Figure format (default: pdf)")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs")
    p.add_argument("--xy-range", type=float, default=12.0, help="Axis range for ratio scatter")
    p.add_argument("--label-a", default="species A", help="Y-axis label (A)")
    p.add_argument("--label-b", default="species B", help="X-axis label (B)")
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    args = p.parse_args()

    res = infer_resolution_from_liftover(args.liftover)
    Order = ChromIndexingFAI(args.fadix)

    pA, pB, dA, dB, ok_dist = extract_arrays_for_contact_stat(args.liftover, res, Order)
    short_list = read_contacts(args.liftover, Order, res, short=True)

    PA, PB = [], []
    for c, q, lr, _ in short_list:
        if args.min_bins < lr <= args.max_bins:
            PA.append(c); PB.append(q)
    BP = np.array(PB, dtype=float)
    BP_rnd = BP.copy()
    if BP_rnd.size > 0:
        np.random.shuffle(BP_rnd)

    bins = args.bins
    Hobs, xedges, yedges = np.histogram2d(np.array(PB), np.array(PA), bins=(bins, bins), range=[(0,100),(0,100)], density=True)
    Hran, _, _ = np.histogram2d(BP_rnd, np.array(PA), bins=(bins, bins), range=[(0,100),(0,100)], density=True)
    M = _log2_ratio_safe(Hobs, Hran)

    fig, ax = plt.subplots(figsize=(5.5,5.2))
    cmap = make_pinkblue_cmap() if args.cmap == 'pinkblue' else plt.get_cmap(args.cmap)
    norm = make_diverging_norm(M, vcenter=0.0, vmax=args.vmax)
    im = ax.imshow(M.T, origin='lower', extent=[0,100,0,100], cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Log2(observed/random)", rotation=270, labelpad=14)
    ax.set_xlabel(f"{args.label_b} percentile score")
    ax.set_ylabel(f"{args.label_a} percentile score")
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.set_xticks(range(0,101,10)); ax.set_yticks(range(0,101,10))
    _save_fig(fig, f"{args.out_prefix}.percentile_heatmap.{args.format}", fmt=args.format, dpi=args.dpi)
    plt.close(fig)

    maskA = ok_dist & (dA >= 0) & (dB >= 0)
    dA_use = np.clip(dA[maskA], 0, args.max_dist_mb)
    dB_use = np.clip(dB[maskA], 0, args.max_dist_mb)
    dB_rnd = dB_use.copy()
    if dB_rnd.size > 0:
        np.random.shuffle(dB_rnd)

    Hobs2, _, _ = np.histogram2d(dB_use, dA_use, bins=(bins, bins), range=[(0,args.max_dist_mb),(0,args.max_dist_mb)], density=True)
    Hran2, _, _ = np.histogram2d(dB_rnd, dA_use, bins=(bins, bins), range=[(0,args.max_dist_mb),(0,args.max_dist_mb)], density=True)
    M2 = _log2_ratio_safe(Hobs2, Hran2)

    fig2, ax2 = plt.subplots(figsize=(5.5,5.2))
    cmap2 = make_pinkblue_cmap() if args.cmap == 'pinkblue' else plt.get_cmap(args.cmap)
    norm2 = make_diverging_norm(M2, vcenter=0.0, vmax=args.vmax)
    im2 = ax2.imshow(M2.T, origin='lower', extent=[0, args.max_dist_mb, 0, args.max_dist_mb], cmap=cmap2, norm=norm2)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Log2(observed/random)", rotation=270, labelpad=14)
    ax2.set_xlabel(f"{args.label_b} genomic distance, Mb")
    ax2.set_ylabel(f"{args.label_a} genomic distance, Mb")
    ax2.set_xlim(0,args.max_dist_mb); ax2.set_ylim(0,args.max_dist_mb)
    _save_fig(fig2, f"{args.out_prefix}.distance_heatmap.{args.format}", fmt=args.format, dpi=args.dpi)
    plt.close(fig2)

    eps = 1e-6
    pA_use = pA[maskA]; pB_use = pB[maskA]
    logR_dist = _log2_ratio_safe(dA_use + eps, dB_use + eps)
    logR_freq = _log2_ratio_safe(pA_use + eps, pB_use + eps)
    rng = args.xy_range
    Hobs3, _, _ = np.histogram2d(logR_dist, logR_freq, bins=(bins, bins), range=[(-rng, rng), (-rng, rng)], density=True)
    pB_rnd_for_ratio = pB_use.copy()
    if pB_rnd_for_ratio.size > 0:
        np.random.shuffle(pB_rnd_for_ratio)
    logR_freq_rnd = _log2_ratio_safe(pA_use + eps, pB_rnd_for_ratio + eps)
    Hran3, _, _ = np.histogram2d(logR_dist, logR_freq_rnd, bins=(bins, bins), range=[(-rng, rng), (-rng, rng)], density=True)
    M3 = _log2_ratio_safe(Hobs3, Hran3)

    fig3, ax3 = plt.subplots(figsize=(5.5,5.2))
    cmap3 = make_pinkblue_cmap() if args.cmap == 'pinkblue' else plt.get_cmap(args.cmap)
    norm3 = make_diverging_norm(M3, vcenter=0.0, vmax=args.vmax)
    im3 = ax3.imshow(M3.T, origin='lower', extent=[-rng, rng, -rng, rng], cmap=cmap3, norm=norm3)
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label("Log2(observed/random)", rotation=270, labelpad=14)
    ax3.set_xlabel(f"log2({args.label_a} contact distance / {args.label_b} contact distance)")
    ax3.set_ylabel(f"log2({args.label_a} contact frequency / {args.label_b} contact frequency)")
    _save_fig(fig3, f"{args.out_prefix}.ratio_scatter.{args.format}", fmt=args.format, dpi=args.dpi)
    plt.close(fig3)

if __name__ == "__main__":
    main()
