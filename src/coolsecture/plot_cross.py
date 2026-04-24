#!/usr/bin/env python3
import argparse, numpy as np, matplotlib, os, re, glob
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap

def make_redwhite_cmap():
    return LinearSegmentedColormap.from_list("RedWhite", [(1,1,1), (0.85,0,0)], N=256)

from .post_common import (
    infer_resolution_from_liftover, ChromIndexingFAI, read_chr_sizes_from_fai,
    read_contacts, _parse_locus, _bins_for_region, _matrix_for_regions,
    make_pinkblue_cmap, _save_fig
)

HEAT_TOKENS = {"obs", "ctrl", "diff", "log2ratio"}

def parse_heat_pair(s: str):
    s = (s or "").strip().lower()
    if not s:
        return ("obs","obs")
    if "-" in s:
        xx, yy = s.split("-", 1)
        xx, yy = xx.strip(), yy.strip()
        if xx not in HEAT_TOKENS or yy not in HEAT_TOKENS:
            raise ValueError(f"--heat expects token(s) in {HEAT_TOKENS}, got '{s}'")
        return (xx, yy)
    if s not in HEAT_TOKENS:
        raise ValueError(f"--heat expects token(s) in {HEAT_TOKENS}, got '{s}'")
    return (s, s)

def norm_and_cmap(mode: str, M: np.ndarray):
    if mode in ("obs", "ctrl"):
        norm = plt.Normalize(vmin=0, vmax=100)
        cmap = plt.get_cmap("Reds")
        label = "observed percentile" if mode == "obs" else "control percentile"
    else:
        finite = np.isfinite(M)
        lim = np.nanpercentile(np.abs(M[finite]), 98) if finite.any() else 1.0
        lim = max(float(lim), 1e-6)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        cmap = make_pinkblue_cmap()
        label = "obs - ctrl" if mode == "diff" else "log2(obs/ctrl)"
    return norm, cmap, label

def resolve_cmap(name: str, default: str):
    if not name:
        name = default
    key = name.strip().lower()
    if key == "redwhite":
        return make_redwhite_cmap()
    if key == "pinkblue":
        return make_pinkblue_cmap()
    try:
        return plt.get_cmap(name)
    except Exception as e:
        raise ValueError(f"Unknown colormap: {name}") from e

def main():
    p = argparse.ArgumentParser(
        prog="plot-cross",
        description="Plot a two-panel figure for a selected locus using a .liftContacts file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--liftover", help="Path to .liftContacts file")
    p.add_argument("--liftover-prefix", help="Prefix to batch process .liftContacts (e.g., out/A_B)")
    p.add_argument("--fadix", required=True, help="Path to FASTA index (.fai) file")
    p.add_argument("--locus", required=True, help="Target locus: 'chr' or 'chr:start-end'")
    p.add_argument("--heat", default="obs", help=(
            "Heat mode for the split heatmap. One of {obs, ctrl, diff, log2ratio}, "
            "or composite 'UPPER-LOWER' (e.g., 'obs-ctrl'). ")
    )
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    p.add_argument("--cmap1", default="pinkblue",
        help="Colormap for upper triangle (e.g., redwhite, pinkblue, RdBu_r, viridis)")
    p.add_argument("--cmap2", default="pinkblue",
        help="Colormap for lower triangle (e.g., pinkblue, redwhite, RdBu_r, viridis)")
    p.add_argument("--format", default="pdf", choices=["pdf","png","svg"], help="Figure format (default: pdf)")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs")
    args = p.parse_args()

    def resolve_inputs():
        if args.liftover_prefix:
            paths = sorted(glob.glob(f"{args.liftover_prefix}.r*.liftContacts"))
            if not paths:
                single = f"{args.liftover_prefix}.liftContacts"
                if os.path.exists(single):
                    return [(single, "")]
                raise SystemExit(f"No liftover files found for prefix: {args.liftover_prefix}")
            out = []
            for pth in paths:
                m = re.search(r"\\.r(\\d+)\\.liftContacts$", pth)
                suffix = f".r{m.group(1)}" if m else ""
                out.append((pth, suffix))
            return out
        if not args.liftover:
            raise SystemExit("Provide --liftover or --liftover-prefix")
        return [(args.liftover, "")]

    def run_one(liftover_path: str, out_prefix: str):
        Order = ChromIndexingFAI(args.fadix)
        res   = infer_resolution_from_liftover(liftover_path)
        sizes = read_chr_sizes_from_fai(args.fadix)
        chrom, start, end = _parse_locus(args.locus, sizes)
        reg = (chrom, start, end)
        contacts = read_contacts(liftover_path, Order, res, short=False)

        h_upper, h_lower = parse_heat_pair(args.heat)

        M_u = _matrix_for_regions(contacts, res, reg, reg, mode=h_upper)
        M_l = _matrix_for_regions(contacts, res, reg, reg, mode=h_lower)

        M_upper = M_u.copy()
        M_lower = M_l.copy()
        M_upper[np.tril_indices_from(M_upper, k=-1)] = np.nan
        M_lower[np.triu_indices_from(M_lower, k=+1)] = np.nan

        norm_u, _, lab_u = norm_and_cmap(h_upper, M_u)
        norm_l, _, lab_l = norm_and_cmap(h_lower, M_l)

        cmap_u = resolve_cmap(args.cmap1, "redwhite")
        cmap_l = resolve_cmap(args.cmap2, "pinkblue")

        fig = plt.figure(figsize=(6.2, 6.2))
        gs  = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 8], hspace=0.08)

        axT = fig.add_subplot(gs[0, 0])
        axT.axis('off')

        ax = fig.add_subplot(gs[1, 0])

        cy, by_s, by_e, _ = _bins_for_region(reg, res)
        cx, bx_s, bx_e, _ = _bins_for_region(reg, res)
        extent = [ (bx_s*res)/1e6, (bx_e*res)/1e6, (by_s*res)/1e6, (by_e*res)/1e6 ]

        im_l = ax.imshow(M_lower, origin='lower', cmap=cmap_l, norm=norm_l,
                         extent=extent, aspect='equal', interpolation='none', resample=False, zorder=1)
        im_u = ax.imshow(M_upper, origin='lower', cmap=cmap_u, norm=norm_u,
                         extent=extent, aspect='equal', interpolation='none', resample=False, zorder=2)

        ax.plot([extent[0], extent[1]], [extent[2], extent[3]], color='k', lw=0.8, alpha=0.6)

        ax.set_xlabel(f"{cx} (Mb)"); ax.set_ylabel(f"{cy} (Mb)")

        fig.canvas.draw()
        bbox = ax.get_position()

        w   = 0.018
        gap = 0.1
        pad = 0.16

        x_right_outer = bbox.x1 + pad
        x_right_inner = x_right_outer - w - gap

        cax_right_upper = fig.add_axes([x_right_outer, bbox.y0, w, bbox.height])
        cax_right_lower = fig.add_axes([x_right_inner, bbox.y0, w, bbox.height])

        cbarU = plt.colorbar(im_u, cax=cax_right_upper)
        cbarU.set_label(f"{h_upper}: {lab_u}", rotation=270, labelpad=12)
        cax_right_upper.yaxis.set_ticks_position('right')
        cax_right_upper.yaxis.set_label_position('right')

        cbarL = plt.colorbar(im_l, cax=cax_right_lower)
        cbarL.set_label(f"{h_lower}: {lab_l}", rotation=270, labelpad=12)
        cax_right_lower.yaxis.set_ticks_position('right')
        cax_right_lower.yaxis.set_label_position('right')

        region_tag = f"{chrom}.{start}-{end}"
        out = f"{out_prefix}.{region_tag}.cross.{h_upper}-{h_lower}.{args.format}"
        _save_fig(fig, out, fmt=args.format, dpi=args.dpi)
        plt.close(fig)

    for liftover_path, suffix in resolve_inputs():
        run_one(liftover_path, args.out_prefix + suffix)

if __name__ == "__main__":
    main()
