#!/usr/bin/env python3
import argparse, numpy as np, os
from .post_common import (
    configure_matplotlib_for_publication,
    infer_resolution_from_liftover, ChromIndexingFAI, read_chr_sizes_from_fai,
    read_contacts, _parse_locus, _bins_for_region, _matrix_for_regions,
    make_pinkblue_cmap, _save_fig, resolve_liftover_inputs
)
configure_matplotlib_for_publication()
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap

def make_redwhite_cmap():
    return LinearSegmentedColormap.from_list("RedWhite", [(1,1,1), (0.85,0,0)], N=256)

HEAT_TOKENS = {"obs", "tgt", "diff", "log2ratio"}

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
    if mode in ("obs", "tgt"):
        norm = plt.Normalize(vmin=0, vmax=100)
        cmap = plt.get_cmap("Reds")
        label = "observed percentile" if mode == "obs" else "target percentile"
    else:
        finite = np.isfinite(M)
        lim = np.nanpercentile(np.abs(M[finite]), 98) if finite.any() else 1.0
        lim = max(float(lim), 1e-6)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        cmap = make_pinkblue_cmap()
        label = "obs - tgt" if mode == "diff" else "log2(obs/tgt)"
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

def read_bedgraph_for_region(path: str, chrom: str, start: int, end: int):
    """Read bedGraph lines overlapping [start, end) on chrom."""
    data = []
    alias_targets = {chrom, chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            a = line.strip().split("\t")
            if len(a) < 4:
                continue
            try:
                c = a[0]
                s = int(a[1])
                e = int(a[2])
                v = float(a[3])
            except Exception:
                continue
            if c not in alias_targets:
                continue
            if e <= start or s >= end:
                continue
            data.append((max(s, start), min(e, end), v))
    return data

def _chrom_alias_to_fai(chrom: str, sizes):
    if chrom in sizes:
        return chrom
    alt = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
    return alt if alt in sizes else None

def read_bedgraph_genome(path: str, sizes):
    data = {chrom: [] for chrom in sizes}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            a = line.strip().split()
            if len(a) < 4:
                continue
            try:
                chrom = _chrom_alias_to_fai(a[0], sizes)
                s = int(a[1])
                e = int(a[2]) + 1
                v = float(a[3])
            except Exception:
                continue
            if chrom is None or not np.isfinite(v):
                continue
            s = max(0, s)
            e = min(int(sizes[chrom]), e)
            if e > s:
                data[chrom].append((s, e, v))
    for chrom in data:
        data[chrom].sort(key=lambda x: x[0])
    return data

def score_bedgraph_windows(path: str, sizes, window_bp: int, min_covered_frac: float):
    data = read_bedgraph_genome(path, sizes)
    step_bp = max(1, window_bp // 2)
    scored = []
    for chrom, rows in data.items():
        chrom_size = int(sizes[chrom])
        if chrom_size < window_bp or not rows:
            continue
        start_idx = 0
        for ws in range(0, chrom_size - window_bp + 1, step_bp):
            we = ws + window_bp
            while start_idx < len(rows) and rows[start_idx][1] <= ws:
                start_idx += 1
            total = 0.0
            covered = 0
            j = start_idx
            while j < len(rows) and rows[j][0] < we:
                s, e, v = rows[j]
                ov = max(0, min(e, we) - max(s, ws))
                if ov > 0:
                    total += ov * v
                    covered += ov
                j += 1
            covered_frac = covered / float(window_bp)
            if covered_frac >= min_covered_frac and covered > 0:
                scored.append((chrom, ws, we, total / covered, covered_frac))
    return scored

def _far_enough(candidate, selected, min_spacing_bp: int):
    chrom, start, end = candidate[:3]
    for row in selected:
        c, s, e = row[:3]
        if c != chrom:
            continue
        if not (end + min_spacing_bp <= s or start - min_spacing_bp >= e):
            return False
    return True

def select_auto_regions(scored, top_n: int, mode: str, min_spacing_bp: int):
    rows = []
    if mode in ("differential", "both"):
        selected = []
        for row in sorted(scored, key=lambda x: x[3], reverse=True):
            if _far_enough(row, selected, min_spacing_bp):
                selected.append(row)
                rows.append(("diff", len(selected), *row))
                if len(selected) >= top_n:
                    break
    if mode in ("conserved", "both"):
        selected = []
        for row in sorted(scored, key=lambda x: x[3]):
            if _far_enough(row, selected, min_spacing_bp):
                selected.append(row)
                rows.append(("cons", len(selected), *row))
                if len(selected) >= top_n:
                    break
    return rows

def write_top_regions(path: str, regions):
    with open(path, "w") as f:
        f.write("class\trank\tchrom\tstart\tend\tscore\tcovered_frac\n")
        for cls, rank, chrom, start, end, score, covered_frac in regions:
            f.write(f"{cls}\t{rank}\t{chrom}\t{start}\t{end}\t{score:.6g}\t{covered_frac:.6g}\n")
    print(f"[OK] wrote {path}")

def draw_pbad_track(ax, bed_data, extent):
    """Draw PBAD values as a light-green filled track with a dark-green line overlay."""
    if not bed_data:
        ax.set_visible(False)
        return
    # Fill positive/negative regions with light green
    xs_line = []
    ys_line = []
    for s, e, v in bed_data:
        x0, x1 = s / 1e6, e / 1e6
        xs_line.extend([x0, x1])
        ys_line.extend([v, v])
        if v >= 0:
            ax.fill_between([x0, x1], 0, v, alpha=0.35, color='#90EE90', step='post')
        else:
            ax.fill_between([x0, x1], v, 0, alpha=0.35, color='#90EE90', step='post')
    # Overlay dark-green line
    if xs_line:
        ax.plot(xs_line, ys_line, color='#006400', linewidth=1.0, drawstyle='steps-post')
    vals = [v for _, _, v in bed_data if np.isfinite(v)]
    if vals:
        vmin = min(vals)
        vmax = max(vals)
        if vmin >= 0:
            ax.set_ylim(0, max(vmax, 0.01))
        else:
            bound = max(abs(vmin), abs(vmax), 0.01)
            ax.set_ylim(-bound, bound)
            ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

def main():
    p = argparse.ArgumentParser(
        prog="plot-cross",
        description="Plot a two-panel figure for a selected locus using a .liftContacts file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--liftover", help="Path to .liftContacts file")
    p.add_argument("--liftover-prefix", help="Prefix to batch process .liftContacts (e.g., out/A_B)")
    p.add_argument("--fadix", required=True, help="Path to FASTA index (.fai) file")
    p.add_argument("--locus", default=None, help="Target locus: 'chr' or 'chr:start-end'. If omitted, top PBAD-ranked regions are plotted.")
    p.add_argument("--locus2", default=None, help=(
        "Optional second locus for the horizontal (x) axis in non-diagonal plots. "
        "If omitted, --locus is used for both axes (diagonal plot)."
    ))
    p.add_argument("--heat", default="obs", help=(
            "Heat mode for the split heatmap. One of {obs, tgt, diff, log2ratio}, "
            "or composite 'UPPER-LOWER' (e.g., 'obs-tgt'). ")
    )
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    p.add_argument("--cmap1", default=None,
        help="Colormap for upper triangle (e.g., redwhite, pinkblue, RdBu_r, viridis). Defaults depend on --heat mode.")
    p.add_argument("--cmap2", default=None,
        help="Colormap for lower triangle (e.g., pinkblue, redwhite, RdBu_r, viridis). Defaults depend on --heat mode.")
    p.add_argument("--format", default="pdf", choices=["pdf","png","svg"], help="Figure format (default: pdf)")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs")
    p.add_argument("--pbad-bedgraph", default=None, help=(
        "Optional comma-separated PBAD bedGraph track(s) to overlay above the heatmap. "
        "Example: track1.bedGraph,track2.bedGraph"
    ))
    p.add_argument("--rank-bedgraph", default=None, help="PBAD bedGraph used to rank automatic regions when --locus is omitted")
    p.add_argument("--auto-top-n", type=int, default=10, help="Number of automatic regions per class when --locus is omitted")
    p.add_argument("--auto-region-size-mb", type=float, default=2.0, help="Automatic region size in Mb")
    p.add_argument("--auto-region-mode", choices=["differential", "conserved", "both"], default="both", help="Automatic region class to plot")
    p.add_argument("--auto-min-covered-frac", type=float, default=0.5, help="Minimum bedGraph coverage fraction for automatic windows")
    p.add_argument("--auto-min-spacing-mb", type=float, default=2.0, help="Minimum spacing between automatic windows in Mb")
    args = p.parse_args()
    if args.locus is None and args.locus2 is not None:
        raise SystemExit("--locus2 can only be used together with --locus")

    def run_one(liftover_path: str, out_prefix: str, preloaded_contacts=None):
        Order = ChromIndexingFAI(args.fadix)
        res   = infer_resolution_from_liftover(liftover_path)
        sizes = read_chr_sizes_from_fai(args.fadix)
        pbad_paths = [p.strip() for p in (args.pbad_bedgraph or "").split(",") if p.strip()]
        if args.locus is None:
            rank_bg = args.rank_bedgraph or (pbad_paths[0] if pbad_paths else None)
            if not rank_bg:
                raise SystemExit("When --locus is omitted, provide --rank-bedgraph or --pbad-bedgraph for automatic region ranking.")
            if not os.path.exists(rank_bg):
                raise SystemExit(f"Ranking bedGraph not found: {rank_bg}")
            window_bp = max(1, int(args.auto_region_size_mb * 1e6))
            min_spacing_bp = max(0, int(args.auto_min_spacing_mb * 1e6))
            scored = score_bedgraph_windows(rank_bg, sizes, window_bp, args.auto_min_covered_frac)
            regions = select_auto_regions(scored, args.auto_top_n, args.auto_region_mode, min_spacing_bp)
            if not regions:
                raise SystemExit("No automatic regions passed coverage/spacing filters.")
            write_top_regions(f"{out_prefix}.top_regions.tsv", regions)
            old_locus = args.locus
            contacts = read_contacts(liftover_path, Order, res, short=False)
            try:
                for cls, rank, chrom, start, end, score, covered_frac in regions:
                    args.locus = f"{chrom}:{start}-{end}"
                    run_one(liftover_path, f"{out_prefix}.top{rank:02d}.{cls}", preloaded_contacts=contacts)
            finally:
                args.locus = old_locus
            return
        chrom_y, start_y, end_y = _parse_locus(args.locus, sizes)
        reg_y = (chrom_y, start_y, end_y)
        if args.locus2:
            chrom_x, start_x, end_x = _parse_locus(args.locus2, sizes)
            reg_x = (chrom_x, start_x, end_x)
            is_diag = False
        else:
            reg_x = reg_y
            chrom_x, start_x, end_x = chrom_y, start_y, end_y
            is_diag = True
        contacts = preloaded_contacts if preloaded_contacts is not None else read_contacts(liftover_path, Order, res, short=False)

        h_upper, h_lower = parse_heat_pair(args.heat)

        M_u = _matrix_for_regions(contacts, res, reg_y, reg_x, mode=h_upper)
        M_l = _matrix_for_regions(contacts, res, reg_y, reg_x, mode=h_lower)

        M_upper = M_u.copy()
        M_lower = M_l.copy()
        if is_diag:
            M_upper[np.tril_indices_from(M_upper, k=-1)] = np.nan
            M_lower[np.triu_indices_from(M_lower, k=+1)] = np.nan
        else:
            # For non-diagonal plots, mask upper/lower by comparing bin indices
            ny, nx = M_upper.shape
            yy, xx = np.indices((ny, nx))
            M_upper[yy > xx] = np.nan
            M_lower[yy <= xx] = np.nan

        norm_u, _, lab_u = norm_and_cmap(h_upper, M_u)
        norm_l, _, lab_l = norm_and_cmap(h_lower, M_l)

        def mode_to_cmap(mode):
            """Map heat mode to its default colormap."""
            if mode == "obs":
                return plt.get_cmap("Reds")
            if mode == "tgt":
                return plt.get_cmap("Blues")
            if mode == "diff":
                return plt.get_cmap("PiYG")
            return resolve_cmap(None, "pinkblue")

        def infer_default_cmaps(h_upper, h_lower, user_cmap1, user_cmap2):
            """Infer sensible colormap defaults from heat mode unless user explicitly set them.

            Defaults:
              - obs -> Blues
              - tgt -> Reds
              - diff -> PiYG
            """
            if user_cmap1:
                cmap_u = resolve_cmap(user_cmap1, "pinkblue")
            else:
                cmap_u = mode_to_cmap(h_upper)

            if user_cmap2:
                cmap_l = resolve_cmap(user_cmap2, "pinkblue")
            else:
                cmap_l = mode_to_cmap(h_lower)
            return cmap_u, cmap_l

        cmap_u, cmap_l = infer_default_cmaps(h_upper, h_lower, args.cmap1, args.cmap2)

        pbad_paths = [p.strip() for p in (args.pbad_bedgraph or "").split(",") if p.strip()]
        has_pbad = bool(pbad_paths)

        fig = plt.figure(figsize=(6.2, 6.2))
        if has_pbad:
            gs = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[0.25, 0.005, 8], hspace=0.01)
            ax_pbad = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[2, 0])
        else:
            gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 8], hspace=0.08)
            axT = fig.add_subplot(gs[0, 0])
            axT.axis('off')
            ax = fig.add_subplot(gs[1, 0])

        cy, by_s, by_e, _ = _bins_for_region(reg_y, res)
        cx, bx_s, bx_e, _ = _bins_for_region(reg_x, res)
        extent = [ (bx_s*res)/1e6, (bx_e*res)/1e6, (by_s*res)/1e6, (by_e*res)/1e6 ]

        im_l = ax.imshow(M_lower, origin='lower', cmap=cmap_l, norm=norm_l,
                         extent=extent, aspect='equal', interpolation='none', resample=False, zorder=1)
        im_u = ax.imshow(M_upper, origin='lower', cmap=cmap_u, norm=norm_u,
                         extent=extent, aspect='equal', interpolation='none', resample=False, zorder=2)

        if is_diag:
            ax.plot([extent[0], extent[1]], [extent[2], extent[3]], color='k', lw=0.8, alpha=0.6)

        ax.set_xlabel(f"{cx} (Mb)"); ax.set_ylabel(f"{cy} (Mb)")

        fig.canvas.draw()

        # Align track width to heatmap after rendering so that x-limits map to the same pixels.
        if has_pbad and 'ax_pbad' in locals():
            # When aspect='equal', the actual data region may be centered with padding.
            # Use transData to get the precise data region in figure coordinates.
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            pts_disp = ax.transData.transform([(xlim[0], ylim[0]), (xlim[1], ylim[1])])
            inv = fig.transFigure.inverted()
            pts_fig = inv.transform(pts_disp)
            data_left  = float(pts_fig[0][0])
            data_right = float(pts_fig[1][0])
            data_width = data_right - data_left
            track_bbox = ax_pbad.get_position()
            ax_pbad.set_position([
                data_left,
                track_bbox.y0,
                data_width,
                track_bbox.height
            ])
            for pth in pbad_paths:
                if not os.path.exists(pth):
                    print(f"[WARN] PBAD bedGraph not found: {pth}")
                    continue
                bed_data = read_bedgraph_for_region(pth, chrom_x, start_x, end_x)
                draw_pbad_track(ax_pbad, bed_data, extent)
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

        if is_diag:
            region_tag = f"{chrom_y}.{start_y}-{end_y}"
        else:
            region_tag = f"{chrom_y}.{start_y}-{end_y}.{chrom_x}.{start_x}-{end_x}"
        out = f"{out_prefix}.{region_tag}.cross.{h_upper}-{h_lower}.{args.format}"
        _save_fig(fig, out, fmt=args.format, dpi=args.dpi)
        plt.close(fig)

    for liftover_path, suffix in resolve_liftover_inputs(args.liftover, args.liftover_prefix):
        run_one(liftover_path, args.out_prefix + suffix)

if __name__ == "__main__":
    main()
