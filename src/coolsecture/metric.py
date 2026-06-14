#!/usr/bin/env python3
import argparse, numpy as np, os, re, glob
from .post_common import (
    infer_resolution_from_liftover, ChromIndexingFAI, read_contacts,
    metricCalc, randomize_contacts, make_pinkblue_cmap, make_diverging_norm, _save_fig
)
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(
        prog="metric",
        description="Compute windowed contact metrics from a .liftContacts table and write bedGraph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    p.add_argument("--liftover", help="Path to .liftContacts file")
    p.add_argument("--liftover-prefix", help="Prefix to batch process .liftContacts (e.g., out/A_B)")
    p.add_argument("--fadix", required=True, help="Path to FASTA index (.fai) file")
    p.add_argument("--frames", type=int, nargs="+", default=[8], help="Half-window size in bins")
    p.add_argument("--metric", choices=["pbad","log","stripe","pearsone","spearman"], default="pbad", help="Metric type")
    p.add_argument("--max-dist-mb", type=float, default=100.0, help="Max intra-chrom distance (Mb)")
    p.add_argument("--format", default="pdf", choices=["pdf","png","svg"], help="Figure format (default: pdf)")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs")
    p.add_argument("--out-prefix", required=True, help="Output prefix for bedGraph and figures")
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
        res = infer_resolution_from_liftover(liftover_path)
        Order = ChromIndexingFAI(args.fadix)
        contacts = read_contacts(liftover_path, Order, res, short=False)

        for frame in args.frames:
            R = metricCalc(contacts, res, frame=frame, metric=args.metric, max_dist_mb=args.max_dist_mb)
            R.sort(key=lambda x: (Order[x[0]], x[1], x[2]))
            out_bg = f"{out_prefix}.{args.metric}.{frame}frame.bedGraph"
            with open(out_bg, 'w') as f:
                for (chrom, b1, b2, v) in R:
                    print(f"{chrom}\t{b1*res}\t{b2*res-1}\t{v}", file=f)
            print(f"[OK] {out_bg}")

            obs_vals = np.array([x[3] for x in R if len(x) == 4 and np.isfinite(x[3])], dtype=float)
            ctrl_vals = None
            try:
                ctrl = randomize_contacts(contacts, idx=1)
                Rc = metricCalc(ctrl, res, frame=frame, metric=args.metric, max_dist_mb=args.max_dist_mb)
                ctrl_vals = np.array([x[3] for x in Rc if len(x) == 4 and np.isfinite(x[3])], dtype=float)
            except Exception:
                ctrl_vals = None

            fig, ax = plt.subplots(figsize=(5.6, 4.2))
            if obs_vals.size == 0:
                ax.text(0.5, 0.5, "No metric data", ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
            else:
                vmin = np.quantile(obs_vals, 0.01)
                vmax = np.quantile(obs_vals, 0.99)
                if abs(vmax) > 0 and abs(vmin) > 0 and args.metric in ("log", "pbad"):
                    lim = max(abs(vmin), abs(vmax))
                    vmin, vmax = -lim, lim
                bins = 50
                ax.hist(obs_vals, bins=bins, range=(vmin, vmax), density=True, alpha=0.6, label="observed", edgecolor='none')

                if ctrl_vals is not None and ctrl_vals.size > 0:
                    xs = np.linspace(vmin, vmax, 400)
                    try:
                        from sklearn.neighbors import KernelDensity
                        kde = KernelDensity(kernel='gaussian', bandwidth=(vmax - vmin) / 40.0).fit(ctrl_vals[:, None])
                        ys = np.exp(kde.score_samples(xs[:, None]))
                    except Exception:
                        h, edges = np.histogram(ctrl_vals, bins=bins, range=(vmin, vmax), density=True)
                        xs = (edges[:-1] + edges[1:]) / 2
                        ys = h
                    ax.plot(xs, ys, linestyle='--', linewidth=1.8, label="random")

                ax.set_xlabel(f"{args.metric} score (frame={frame})")
                ax.set_ylabel("density")
                ax.legend(frameon=False)

            out_fig = f"{out_prefix}.{args.metric}.{frame}frame.stat.{args.format}"
            _save_fig(fig, out_fig, fmt=args.format, dpi=args.dpi)
            plt.close(fig)

    for liftover_path, suffix in resolve_inputs():
        run_one(liftover_path, args.out_prefix + suffix)

if __name__ == "__main__":
    main()
