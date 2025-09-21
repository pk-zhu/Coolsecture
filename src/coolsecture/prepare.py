#!/usr/bin/env python3
import os
import sys
import io
import math
import argparse
from pathlib import Path
from typing import Tuple

# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

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
        description=("Preprocess a .cool/.mcool into contact tables with percentile scores by distance bucket (vectorized, single-process)."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--cool", required=True, help="Path to .cool or .mcool::resolutions/RES")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (e.g., out/Asu_Ath)")
    ap.add_argument("--chunksize", type=int, default=5000000, help="Pixels per chunk to read and process")
    ap.add_argument("--max-distance", type=int, default=10000000, help="Max genomic distance (bp) for intra contacts")
    ap.add_argument("--inter", action="store_true", help="Also keep inter-chrom contacts (bucket key = -1)")
    ap.add_argument("--nthreads", type=int, default=1, help="Thread cap for BLAS/OMP backends (no multiprocessing)")
    return ap.parse_args()

def set_thread_env(n: int) -> None:
    n = max(1, int(n))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ[k] = str(n)

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def load_bins_arrays(clr: cooler.Cooler) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = clr.bins()[:][["chrom", "start", "end", "weight"]]
    chroms = bins["chrom"].to_numpy(object, copy=False)
    starts = bins["start"].to_numpy(np.int64, copy=False)
    ends = bins["end"].to_numpy(np.int64, copy=False)
    w = bins.get("weight")
    if w is None:
        weights = np.ones(len(bins), dtype=np.float32)
    else:
        weights = w.to_numpy(np.float32, copy=True)
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
    clr = cooler.Cooler(a.cool)
    if clr.binsize is None:
        raise RuntimeError("Input cooler must have fixed bin size; use mcool path with ::resolutions/RES")
    chroms, starts, ends, weights = load_bins_arrays(clr)
    n_bins = len(chroms)
    out_prefix = Path(a.out_prefix)
    tmpdir = Path(str(out_prefix) + ".tmp")
    tmpdir.mkdir(parents=True, exist_ok=True)
    res, keys_seen, coverage = first_pass_build(tmpdir, clr, chroms, starts, ends, weights, a.chunksize, a.max_distance, a.inter, n_bins)
    second_pass_write(out_prefix, tmpdir, keys_seen, chroms, starts, ends, coverage)
    print("Done.")

if __name__ == "__main__":
    main()
