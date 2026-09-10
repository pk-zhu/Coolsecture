#!/usr/bin/env python3
import argparse
import glob
import os
import re
import numpy as np

from .post_common import (
    infer_resolution_from_liftover, ChromIndexingFAI, read_contacts,
    _import_plotly, _resolve_interactive, _write_plotly_html,
)
from .metric import metricCalc

def _resolve_liftover_prefix(prefix: str):
    paths = sorted(glob.glob(f"{prefix}.r*.liftContacts"))
    if not paths:
        single = f"{prefix}.liftContacts"
        if os.path.exists(single):
            return [(single, None)]
        raise SystemExit(f"No liftover files found for prefix: {prefix}")
    out = []
    for p in paths:
        m = re.search(r"\.r(\d+)\.liftContacts$", p)
        r = int(m.group(1)) if m else None
        out.append((p, r))
    return out

# Per-locus metric values have different sign/orientation. For multi-scale
# aggregation we map every metric to a common DIVERGENCE score: higher = more
# divergent, 0 = identical source/target.
#   pbad           : already a non-negative divergence
#   log / stripe   : signed log10 ratio -> magnitude
#   pearsone/spearman: correlation r in [-1,1] -> 1 - r
def _to_divergence(raw: float, metric: str) -> float:
    if metric in ("pearsone", "spearman"):
        return 1.0 - raw
    if metric in ("log", "stripe"):
        return abs(raw)
    return raw


# Default "divergent contact" threshold on the common divergence scale, per metric.
_DEFAULT_THR = {
    "pbad": 0.2,
    "log": 0.3,
    "stripe": 0.3,
    "pearsone": 0.5,    # r <= 0.5
    "spearman": 0.5,
}


def main():
    p = argparse.ArgumentParser(
        prog="multiscale",
        description="Multi-scale divergence summary across multi-resolution liftover files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--liftover-prefix", required=True, help="Prefix to batch process .liftContacts")
    p.add_argument("--fadix", required=True, help="FASTA index (.fai) file")
    p.add_argument("--frame", type=int, default=8, help="Half-window size in bins for the metric")
    p.add_argument("--metric", choices=["pbad", "log", "stripe", "pearsone", "spearman"],
        default="pbad", help="Divergence metric (spearman needs the [stats] extra)")
    p.add_argument("--metric-thr", type=float, default=None,
        help="Divergence threshold defining high-divergence contacts (default depends on --metric)")
    p.add_argument("--interactive", default="auto", choices=["auto","on","off"],
        help="Also write interactive Plotly HTML summary charts")
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    args = p.parse_args()

    if args.metric == "spearman":
        try:
            import scipy  # noqa: F401
        except Exception:
            raise SystemExit("spearman metric requires scipy (install the [stats] extra).")

    thr = args.metric_thr if args.metric_thr is not None else _DEFAULT_THR[args.metric]

    rows = []
    for path, res_hint in _resolve_liftover_prefix(args.liftover_prefix):
        res = res_hint or infer_resolution_from_liftover(path)
        Order = ChromIndexingFAI(args.fadix)
        contacts = read_contacts(path, Order, res, short=False)
        vals = metricCalc(contacts, res, frame=args.frame, metric=args.metric)
        arr = np.array([_to_divergence(v[3], args.metric)
                        for v in vals if len(v) == 4 and np.isfinite(v[3])], dtype=float)
        if arr.size == 0:
            mean = median = frac_high = 0.0
        else:
            mean = float(np.mean(arr))
            median = float(np.median(arr))
            frac_high = float(np.mean(arr > thr))
        rows.append((res, mean, median, frac_high))

    rows.sort(key=lambda x: x[0])
    means = np.array([r[1] for r in rows], dtype=float)
    stability = 1.0 - (np.std(means) / (abs(np.mean(means)) + 1e-9)) if means.size else 0.0

    tag = args.metric
    out_tsv = args.out_prefix + f".multiscale.{tag}.tsv"
    with open(out_tsv, "w") as f:
        f.write(f"resolution\tmean_{tag}\tmedian_{tag}\tfrac_high_{tag}\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]:.6f}\t{r[2]:.6f}\t{r[3]:.6f}\n")
    out_sum = args.out_prefix + f".multiscale.{tag}.summary.tsv"
    with open(out_sum, "w") as f:
        f.write("metric\tvalue\n")
        f.write(f"resolution_stability_score\t{stability:.6f}\n")

        coarse = [r for r in rows if r[0] >= 40000]
        fine = [r for r in rows if r[0] <= 20000]
        coarse_mean = float(np.mean([r[1] for r in coarse])) if coarse else 0.0
        fine_mean = float(np.mean([r[1] for r in fine])) if fine else 0.0
        coarse_flag = coarse_mean <= thr
        fine_flag = fine_mean > thr
        f.write(f"coarse_scale_conserved_folding\t{int(coarse_flag)}\n")
        f.write(f"fine_scale_divergent_contacts\t{int(fine_flag)}\n")

    print(f"[OK] {out_tsv}")
    print(f"[OK] {out_sum}")

    try:
        enable_plotly = _resolve_interactive(args.interactive)
    except Exception as e:
        raise SystemExit(str(e))
    if enable_plotly and rows:
        try:
            go, _, make_subplots = _import_plotly()
            if go is not None:
                res = [r[0] for r in rows]
                mean = [r[1] for r in rows]
                median = [r[2] for r in rows]
                frac = [r[3] for r in rows]
                fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
                score_labels = {
                    "pbad": "PBAD",
                    "log": "|log10 ratio|",
                    "stripe": "|stripe log10 ratio|",
                    "pearsone": "1 - Pearson r",
                    "spearman": "1 - Spearman rho",
                }
                slabel = score_labels[args.metric]
                fig.add_trace(go.Scatter(x=res, y=mean, mode="lines+markers", name=f"mean {slabel}"), secondary_y=False)
                fig.add_trace(go.Scatter(x=res, y=median, mode="lines+markers", name=f"median {slabel}"), secondary_y=False)
                fig.add_trace(go.Bar(x=res, y=frac, name="frac high-divergence"), secondary_y=True)
                fig.update_xaxes(type="log", title_text="resolution (bp)")
                fig.update_yaxes(title_text=slabel, secondary_y=False)
                fig.update_yaxes(title_text="fraction high", secondary_y=True)
                _write_plotly_html(fig, args.out_prefix + f".multiscale.{tag}.plotly.html",
                                   title=f"Multi-scale {slabel} summary")

                metrics = ["resolution_stability_score", "coarse_scale_conserved_folding", "fine_scale_divergent_contacts"]
                values = [stability, int(coarse_flag), int(fine_flag)]
                fig2 = go.Figure(data=[go.Bar(x=metrics, y=values, text=[f"{v:.4g}" for v in values], textposition="auto")])
                fig2.update_layout(xaxis_title="metric", yaxis_title="value")
                _write_plotly_html(fig2, args.out_prefix + f".multiscale.{tag}.summary.plotly.html",
                                   title="Multi-scale summary metrics")
        except Exception as e:
            print(f"[WARN] Plotly output failed: {e}")

if __name__ == "__main__":
    main()
