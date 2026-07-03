#!/usr/bin/env python3
import os
import sys
import argparse
import time
import concurrent.futures
from pathlib import Path
from typing import Tuple

import numpy as np
import cooler
import pandas as pd

CTS_DTYPE = np.dtype([
    ("i", np.int32), ("j", np.int32),
    ("count", np.float32),
    ("val", np.float32), ("low", np.float32), ("high", np.float32)
])

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="prepare_vectorized",
        description=("Preprocess a .cool/.mcool/.hic into contact tables with percentile scores by distance bucket (vectorized, single-process)."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--matrix", required=True, help="Path to .cool/.mcool::resolutions/RES or .hic")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (e.g., out/Asu_Ath)")
    ap.add_argument("--chunksize", type=int, default=5000000, help="Pixels per chunk to read and process")
    ap.add_argument("--max-distance", type=int, default=10000000, help="Max genomic distance (bp) for intra contacts")
    ap.add_argument("--inter", action="store_true", help="Also keep inter-chrom contacts (bucket key = -1)")
    ap.add_argument("--nthreads", type=int, default=1, help="Thread cap for BLAS/OMP backends (no multiprocessing)")
    ap.add_argument("--resolution", "-res", default=None,
        help="Resolution(s) in bp. Use comma-separated values for multi-resolution .mcool or .hic input.")

    ap.add_argument("--summary", action="store_true",
        help="Write multi-resolution summary table/plot when --resolution provides multiple values")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs")
    ap.add_argument("--interactive", default="auto", choices=["auto","on","off"],
        help="Write interactive Plotly HTML for summary table (default: auto)")
    return ap.parse_args()

def set_thread_env(n: int) -> None:
    n = max(1, int(n))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ[k] = str(n)

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def is_hic_path(p: str) -> bool:
    return str(p).lower().endswith(".hic")

def convert_hic_to_cool_hicstraw(hic_path: str, out_prefix: Path, res: int, nthreads: int = 1) -> str:
    if res is None:
        raise RuntimeError("For .hic input, --resolution is required (bp).")
    try:
        from hic2cool import hic2cool_convert, hic2cool_extractnorms
    except Exception as e:
        raise RuntimeError("hic2cool is required for .hic input. Please install hic2cool.") from e
    out_path = f"{out_prefix}.hic.{res}.cool"
    nproc = max(1, int(nthreads))
    try:
        hic2cool_convert(hic_path, out_path, res, nproc=nproc, silent=True)
    except NameError as e:
        # hic2cool has a known multiprocessing failure on some Windows setups.
        if nproc == 1 or "reqarr" not in str(e):
            raise
        print("[WARN] hic2cool parallel conversion failed; retrying with nproc=1")
        hic2cool_convert(hic_path, out_path, res, nproc=1, silent=True)
    hic2cool_extractnorms(hic_path, out_path, silent=True)
    return out_path

def parse_resolution_arg(s: str):
    if not s:
        return []
    out = []
    for item in str(s).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except ValueError as e:
            raise RuntimeError(f"Invalid resolution in --resolution: {item}") from e
    return out

def load_bins_arrays(clr: cooler.Cooler) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = clr.bins()[:]
    chroms = bins["chrom"].to_numpy(object, copy=False)
    starts = bins["start"].to_numpy(np.int64, copy=False)
    ends = bins["end"].to_numpy(np.int64, copy=False)
    
    # 优先选择归一化向量：KR > VC_SQRT > VC > weight
    norm_cols = []
    for col in ["KR", "VC_SQRT", "VC", "weight"]:
        if col in bins.columns:
            norm_cols.append(col)
    
    if not norm_cols:
        raise RuntimeError(
            "Input cooler file does not have any normalization vectors. "
            "Coolsecture requires a balanced matrix with normalization vectors "
            "(KR, VC_SQRT, VC, or weight column in bins table)."
        )
    
    # 使用第一个找到的归一化向量
    norm_col = norm_cols[0]
    weights = bins[norm_col].to_numpy(np.float32, copy=True)
    invalid = ~np.isfinite(weights) | (weights <= 0)
    if invalid.any():
        weights[invalid] = 1.0
    return chroms, starts, ends, weights

def iter_pixels_chunks(clr: cooler.Cooler, chunksize: int):
    nnz = int(clr.info["nnz"])
    px = clr.pixels()
    cols = ["bin1_id", "bin2_id", "count"]
    for lo in range(0, nnz, chunksize):
        hi = min(lo + chunksize, nnz)
        df = px[lo:hi][cols]
        yield df

def bucket_tmp_paths(tmpdir: Path, key: int) -> Tuple[Path, Path]:
    return tmpdir / f"b{key}.vals.f32", tmpdir / f"b{key}.cts.bin"

def append_vals(path: Path, arr: np.ndarray) -> None:
    with open(path, "ab") as f:
        arr.astype(np.float32, copy=False).tofile(f)

def append_cts(path: Path, rec: np.ndarray) -> None:
    with open(path, "ab") as f:
        rec.tofile(f)

def compute_values_for_chunk(df: pd.DataFrame,
                             chroms: np.ndarray,
                             starts: np.ndarray,
                             ends: np.ndarray,
                             weights: np.ndarray,
                             resolution: int,
                             max_distance_bp: int,
                             keep_inter: bool):
    i = df["bin1_id"].to_numpy(np.int32, copy=False)
    j = df["bin2_id"].to_numpy(np.int32, copy=False)
    c = df["count"].to_numpy(np.float32, copy=False)
    is_intra = (chroms[i] == chroms[j])
    dist_bp = (starts[j] - starts[i]).astype(np.int64, copy=False)
    keep_intra = is_intra & (dist_bp >= 0) & (dist_bp <= max_distance_bp)
    if keep_inter:
        mask = keep_intra | (~is_intra)
    else:
        mask = keep_intra
    if not mask.any():
        return None
    i = i[mask]; j = j[mask]; c = c[mask]
    is_intra = is_intra[mask]
    dist_bp = dist_bp[mask]
    w1 = weights[i]
    w2 = weights[j]
    denom = w1 * w2
    denom[~np.isfinite(denom) | (denom <= 0)] = 1.0
    val = c / denom
    low = val
    high = val
    dist_bins = np.where(is_intra, (dist_bp // int(resolution)).astype(np.int32, copy=False), -1)
    order = np.argsort(dist_bins, kind="stable")
    i = i[order]; j = j[order]; c = c[order]
    val = val[order]; low = low[order]; high = high[order]
    dist_bins = dist_bins[order]
    if dist_bins.size == 0:
        return None
    edges = np.flatnonzero(np.diff(dist_bins)) + 1
    starts_idx = np.r_[0, edges]
    ends_idx = np.r_[edges, dist_bins.size]
    return (dist_bins, starts_idx, ends_idx, i, j, c, val, low, high)

def first_pass_build(tmpdir: Path,
                     clr: cooler.Cooler,
                     chroms: np.ndarray,
                     starts: np.ndarray,
                     ends: np.ndarray,
                     weights: np.ndarray,
                     chunksize: int,
                     max_distance_bp: int,
                     keep_inter: bool,
                     n_bins: int) -> Tuple[int, set, np.ndarray]:
    keys_seen: set = set()
    res = int(clr.binsize)
    coverage = np.zeros(n_bins, dtype=np.float64)
    for df in iter_pixels_chunks(clr, chunksize):
        out = compute_values_for_chunk(df, chroms, starts, ends, weights, res, max_distance_bp, keep_inter)
        if out is None:
            continue
        dist_bins, s_idx, e_idx, i, j, c, val, low, high = out
        coverage += np.bincount(i, weights=c, minlength=n_bins)
        coverage += np.bincount(j, weights=c, minlength=n_bins)
        for s, e in zip(s_idx, e_idx):
            key = int(dist_bins[s])
            keys_seen.add(key)
            vals = val[s:e]
            rec = np.empty(e - s, dtype=CTS_DTYPE)
            rec["i"] = i[s:e]
            rec["j"] = j[s:e]
            rec["count"] = c[s:e]
            rec["val"] = vals
            rec["low"] = low[s:e]
            rec["high"] = high[s:e]
            vpath, cpath = bucket_tmp_paths(tmpdir, key)
            append_vals(vpath, vals)
            append_cts(cpath, rec)
    return res, keys_seen, coverage.astype(np.float32)

def quantiles_100(sorted_vals: np.ndarray) -> np.ndarray:
    if sorted_vals.size == 0:
        return np.zeros(100, dtype=np.float32)
    qs = np.linspace(0.005, 0.995, 100, dtype=np.float64)
    return np.quantile(sorted_vals, qs, method="linear").astype(np.float32)

def second_pass_write(outputs_prefix: Path,
                      tmpdir: Path,
                      keys_seen: set,
                      chroms: np.ndarray,
                      starts: np.ndarray,
                      ends: np.ndarray,
                      coverage: np.ndarray) -> None:
    contacts_file = Path(str(outputs_prefix) + ".contacts.tsv")
    stats_file = Path(str(outputs_prefix) + ".stats.tsv")
    ensure_dir(contacts_file)
    ensure_dir(stats_file)
    with open(contacts_file, "w", buffering=1024*1024) as f_cts, open(stats_file, "w", buffering=1024*1024) as f_stats:
        f_cts.write("chrom1\tstart1\tend1\tbin1\tchrom2\tstart2\tend2\tbin2\trank\tstrict\tweak\tcov1\tcov2\tdist_bins\n")
        f_stats.write("dist_bin\tn\tp05\tp50\tp95\tslice_medians_100(sep=;)\n")
        keys = sorted([k for k in keys_seen if k >= 0])
        if -1 in keys_seen:
            keys.append(-1)
        for key in keys:
            vpath, cpath = bucket_tmp_paths(tmpdir, key)
            if (not vpath.exists()) or (not cpath.exists()):
                continue
            vals = np.fromfile(vpath, dtype=np.float32)
            if vals.size == 0:
                continue
            vals.sort(kind="quicksort")
            n = vals.size
            p05, p50, p95 = np.quantile(vals, [0.05, 0.5, 0.95], method="linear").astype(np.float32)
            meds100 = quantiles_100(vals)
            f_stats.write(f"{key}\t{n}\t{p05:.6g}\t{p50:.6g}\t{p95:.6g}\t" + "\t".join(f"{x:.6g}" for x in meds100) + "\n")
            out_batch = 2000000
            total_bytes = cpath.stat().st_size
            rec_size = CTS_DTYPE.itemsize
            total_rec = total_bytes // rec_size
            with open(cpath, "rb", buffering=1024*1024) as fbin:
                offset = 0
                while offset < total_rec:
                    take = min(out_batch, total_rec - offset)
                    buf = np.fromfile(fbin, dtype=CTS_DTYPE, count=take)
                    offset += take
                    left = np.searchsorted(vals, buf["val"], side="left")
                    right = np.searchsorted(vals, buf["val"], side="right")
                    rank = (left + right) * 0.5 / n
                    strict = np.searchsorted(vals, buf["low"], side="left") / n
                    weak = np.searchsorted(vals, buf["high"], side="right") / n
                    rank_i = np.minimum((rank * 100).astype(np.int32), 99)
                    strict_i = np.minimum((strict * 100).astype(np.int32), 99)
                    weak_i = np.minimum((weak * 100).astype(np.int32), 99)
                    i_idx = buf["i"]
                    j_idx = buf["j"]
                    c1 = chroms[i_idx]
                    s1 = starts[i_idx]
                    e1 = ends[i_idx]
                    c2 = chroms[j_idx]
                    s2 = starts[j_idx]
                    e2 = ends[j_idx]
                    cov1 = coverage[i_idx]
                    cov2 = coverage[j_idx]
                    lines = [
                        f"{c1[k]}\t{s1[k]}\t{e1[k]}\t{int(i_idx[k])}\t{c2[k]}\t{s2[k]}\t{e2[k]}\t{int(j_idx[k])}\t{rank_i[k]:.6g}\t{strict_i[k]:.6g}\t{weak_i[k]:.6g}\t{cov1[k]:.6g}\t{cov2[k]:.6g}\t{key}\n"
                        for k in range(take)
                    ]
                    f_cts.writelines(lines)

def main():
    a = parse_args()
    set_thread_env(a.nthreads)
    cool_path = a.matrix
    out_prefix = Path(a.out_prefix)
    res_arg = a.resolution
    res_vals = parse_resolution_arg(res_arg)
    hic_res = None
    res_list = []
    if res_vals:
        if is_hic_path(cool_path):
            if len(res_vals) == 1:
                hic_res = res_vals[0]
            else:
                res_list = res_vals
        else:
            res_list = res_vals

    def run_single(cooler_path: str, out_pref: Path):
        t0 = time.time()
        clr = cooler.Cooler(cooler_path)
        if clr.binsize is None:
            raise RuntimeError("Input cooler must have fixed bin size; use mcool path with ::resolutions/RES")
        chroms, starts, ends, weights = load_bins_arrays(clr)
        n_bins = len(chroms)
        nnz = int(clr.info.get("nnz", 0))
        raw_sum = float(clr.info.get("sum", 0.0))
        tmpdir = Path(str(out_pref) + ".tmp")
        tmpdir.mkdir(parents=True, exist_ok=True)
        res, keys_seen, coverage = first_pass_build(tmpdir, clr, chroms, starts, ends, weights, a.chunksize, a.max_distance, a.inter, n_bins)
        coverage_total = float(np.sum(coverage))
        covered_bins = int(np.count_nonzero(coverage > 0))
        genome_bp = int(n_bins * clr.binsize)
        coverage_bin_frac = (covered_bins / n_bins) if n_bins > 0 else 0.0
        coverage_per_mb = (coverage_total / (genome_bp / 1e6)) if genome_bp > 0 else 0.0
        second_pass_write(out_pref, tmpdir, keys_seen, chroms, starts, ends, coverage)
        elapsed = time.time() - t0
        return {
            "resolution": int(clr.binsize),
            "nnz": nnz,
            "sum": raw_sum,
            "coverage_sum": coverage_total,
            "coverage_bin_frac": coverage_bin_frac,
            "coverage_per_mb": coverage_per_mb,
            "elapsed_s": elapsed,
            "out_prefix": str(out_pref),
        }

    if res_list:
        if "::resolutions/" in cool_path:
            raise RuntimeError("--resolution expects a raw .mcool or .hic path without ::resolutions/RES")
        if not (cool_path.lower().endswith(".mcool") or is_hic_path(cool_path)):
            raise RuntimeError("--resolution only supports .mcool or .hic input")
        summary_rows = []
        # Use thread pool to process multiple resolutions in parallel
        max_workers = min(a.nthreads, len(res_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks for each resolution
            future_to_res = {}
            for r in res_list:
                out_pref = Path(f"{out_prefix}.r{r}")
                if is_hic_path(cool_path):
                    future = executor.submit(
                        lambda rr, op: run_single(convert_hic_to_cool_hicstraw(cool_path, op, rr, a.nthreads), op),
                        r, out_pref,
                    )
                else:
                    mcool_uri = f"{cool_path}::resolutions/{r}"
                    future = executor.submit(run_single, mcool_uri, out_pref)
                future_to_res[future] = r
            # Collect results
            for future in concurrent.futures.as_completed(future_to_res):
                r = future_to_res[future]
                try:
                    row = future.result()
                    summary_rows.append(row)
                except Exception as e:
                    print(f"Error processing resolution {r}: {e}", file=sys.stderr)
        # Sort results by resolution
        summary_rows.sort(key=lambda x: x["resolution"])
        if a.summary and summary_rows:
            df = pd.DataFrame(summary_rows)
            sum_path = f"{out_prefix}.multi_resolution.summary.tsv"
            df.to_csv(sum_path, sep="	", index=False)
            try:
                from .post_common import _save_fig, _import_plotly, _resolve_interactive, _write_plotly_html
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.2, 7.2), sharex=True)
                ax = axes[0]
                ax.plot(df["resolution"], df["elapsed_s"], marker="o", label="runtime (s)")
                ax.set_ylabel("runtime (s)")
                ax2 = ax.twinx()
                ax2.plot(df["resolution"], df["nnz"], marker="s", color="#d04a02", label="nnz")
                ax2.set_ylabel("nnz")
                ax.grid(True, alpha=0.3)
                lines = ax.get_lines() + ax2.get_lines()
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, frameon=False, loc="upper right")

                axb = axes[1]
                axb.plot(df["resolution"], df["coverage_bin_frac"], marker="o", label="covered bin fraction")
                axb.set_xlabel("resolution (bp)")
                axb.set_ylabel("covered bin fraction")
                axb2 = axb.twinx()
                axb2.plot(df["resolution"], df["coverage_per_mb"], marker="s", color="#2a7f62", label="coverage per Mb")
                axb2.set_ylabel("coverage per Mb")
                axb.grid(True, alpha=0.3)
                lines = axb.get_lines() + axb2.get_lines()
                labels = [l.get_label() for l in lines]
                axb.legend(lines, labels, frameon=False, loc="upper right")

                for axt in axes:
                    axt.set_xscale("log", base=10)
                fig_path = f"{out_prefix}.multi_resolution.summary.pdf"
                _save_fig(fig, fig_path, fmt="pdf", dpi=a.dpi)
                plt.close(fig)
            except Exception:
                pass
            try:
                enable_plotly = _resolve_interactive(a.interactive)
            except Exception as e:
                raise SystemExit(str(e))
            if enable_plotly:
                try:
                    go, _, make_subplots = _import_plotly()
                    if go is not None:
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                            specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
                        fig.add_trace(go.Scatter(x=df["resolution"], y=df["elapsed_s"], mode="lines+markers", name="runtime (s)"),
                                      row=1, col=1, secondary_y=False)
                        fig.add_trace(go.Scatter(x=df["resolution"], y=df["nnz"], mode="lines+markers", name="nnz"),
                                      row=1, col=1, secondary_y=True)
                        fig.add_trace(go.Scatter(x=df["resolution"], y=df["coverage_bin_frac"], mode="lines+markers",
                                                 name="covered bin fraction"), row=2, col=1, secondary_y=False)
                        fig.add_trace(go.Scatter(x=df["resolution"], y=df["coverage_per_mb"], mode="lines+markers",
                                                 name="coverage per Mb"), row=2, col=1, secondary_y=True)
                        fig.update_xaxes(type="log", title_text="resolution (bp)", row=2, col=1)
                        fig.update_yaxes(title_text="runtime (s)", row=1, col=1, secondary_y=False)
                        fig.update_yaxes(title_text="nnz", row=1, col=1, secondary_y=True)
                        fig.update_yaxes(title_text="covered bin fraction", row=2, col=1, secondary_y=False)
                        fig.update_yaxes(title_text="coverage per Mb", row=2, col=1, secondary_y=True)
                        _write_plotly_html(fig, f"{out_prefix}.multi_resolution.summary.plotly.html",
                                           title="Multi-resolution summary")
                except Exception:
                    pass
    else:
        if is_hic_path(cool_path) and hic_res is not None:
            # 如果是 hic 文件且有单个分辨率，先在 out_prefix 上添加分辨率后缀
            out_prefix_with_res = Path(f"{out_prefix}.r{hic_res}")
            cool_path = convert_hic_to_cool_hicstraw(cool_path, out_prefix_with_res, hic_res, a.nthreads)
            run_single(cool_path, out_prefix_with_res)
        else:
            if cool_path.lower().endswith(".mcool") and "::resolutions/" not in str(cool_path):
                raise RuntimeError("For .mcool input, provide --resolution (single or list) or use ::resolutions/RES.")
            if hic_res is None and is_hic_path(cool_path):
                raise RuntimeError("For .hic input, --resolution is required (bp).")
            run_single(cool_path, out_prefix)
    print("Done.")

if __name__ == "__main__":
    main()
