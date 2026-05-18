#!/usr/bin/env python3
"""
Optimized liftcontacts module for Coolsecture.

This module provides optimized liftcontacts functionality that
automatically leverages Linux system features (fork mode multiprocessing,
/proc/meminfo, etc.) when available, while maintaining cross-platform compatibility.
"""

import os
import sys
import argparse
import re
import glob
import tempfile
import signal
import logging
import gc
import sqlite3
from typing import Dict, List, Tuple, Any, Optional
import multiprocessing as mp
from multiprocessing import get_context
from pathlib import Path
import numpy as np

from .post_common import (
    _parse_remap_token, _read_lift_as_map, _reciprocal_stats,
    _import_plotly, _resolve_interactive, _write_plotly_html
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle signals for graceful shutdown."""
    global _shutdown_requested
    logger.warning(f"Received signal {signum}, shutting down gracefully...")
    _shutdown_requested = True


def _setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown."""
    if sys.platform.startswith('linux'):
        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
            try:
                signal.signal(signal.SIGHUP, _signal_handler)
            except AttributeError:
                pass
        except Exception:
            pass


def _get_memory_limit() -> int:
    """Get memory limit in bytes (Linux-specific with fallback)."""
    try:
        if sys.platform.startswith('linux'):
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                if soft != resource.RLIM_INFINITY:
                    return int(soft * 0.7)
            except (ImportError, AttributeError, ValueError):
                pass
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemAvailable:'):
                            mem_available = int(line.split()[1]) * 1024
                            return int(mem_available * 0.7)
            except (FileNotFoundError, IndexError, ValueError):
                pass
    except Exception:
        pass
    return 8 * 1024 * 1024 * 1024


def _normalize_chrom_name(name: str) -> str:
    """Normalize chromosome name."""
    name = str(name)
    if name.startswith('chr'):
        return name
    if name.isdigit() or name in ['X', 'Y', 'M', 'MT']:
        return f'chr{name}'
    return name


def ChromIndexingFAI(path: str) -> Dict:
    """Index chromosome names from FASTA index file."""
    ChrInd = {}
    open_kwargs = {'buffering': 1024 * 1024} if sys.platform.startswith('linux') else {}
    with open(path, 'r', **open_kwargs) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            name = line.split()[0]
            ChrInd[name] = i + 1
            ChrInd[i + 1] = name
            if name.startswith('chr'):
                alias = name[3:]
                ChrInd[alias] = i + 1
    return ChrInd


def _is_rich(path: str) -> bool:
    """Check if file is in RICH format."""
    with open(path, 'r') as f:
        header = f.readline().strip().split()
    rich_columns = ["chrom1", "start1", "end1", "bin1", "chrom2", "start2", "end2", "bin2"]
    if len(header) >= 8 and all(col in header for col in rich_columns[:4]):
        return True
    return False


def read_contacts_rich(contacts_path: str) -> Tuple[Dict, Dict]:
    """
    Read RICH format contacts.
    Optimized for memory efficiency with direct dictionary construction.
    """
    bins = {}
    out = {}
    open_kwargs = {'buffering': 1024 * 1024} if sys.platform.startswith('linux') else {}
    line_count = 0
    with open(contacts_path, 'r', **open_kwargs) as f:
        _ = f.readline().strip().split()
        for line in f:
            line_count += 1
            if (line_count % 10000) == 0 and _shutdown_requested:
                raise KeyboardInterrupt("Shutdown requested")
            if not line:
                continue
            parts = line.rstrip('\n').split('\t', 13)
            if len(parts) < 14:
                parts = line.split(None, 13)
            if len(parts) < 14:
                continue
            (chrom1, start1, end1, bin1,
             chrom2, start2, end2, bin2,
             rank, strict, weak, cov1, cov2, dist_bins) = parts
            start1 = int(start1)
            end1 = int(end1)
            bin1 = int(bin1)
            start2 = int(start2)
            end2 = int(end2)
            bin2 = int(bin2)
            dist_bins_f = float(dist_bins)
            if bin1 not in bins:
                bins[bin1] = (chrom1, start1, end1)
            if bin2 not in bins:
                bins[bin2] = (chrom2, start2, end2)
            key = ((chrom1, bin1), (chrom2, bin2)) if bin1 <= bin2 else ((chrom2, bin2), (chrom1, bin1))
            out[key] = (int(rank), int(strict), int(weak), float(cov1), float(cov2), dist_bins_f)
            if (line_count % 5000000) == 0:
                logger.info(f"read_contacts_rich progress: {line_count} lines from {contacts_path}")
    return out, bins


def build_contact_hash_from_rich(rich_contacts: Dict, ChrIdxs: Dict, consume: bool = False) -> Dict:
    """Build contact hash from RICH contacts.

    When consume=True, entries are popped in insertion order to reduce peak memory.
    """
    out = {}
    if consume:
        while rich_contacts:
            if _shutdown_requested:
                raise KeyboardInterrupt("Shutdown requested")
            first_key, vals = rich_contacts.popitem()
            (chrom1, b1), (chrom2, b2) = first_key
            rank, strict, weak, cov1, cov2, dist_bins = vals
            N1 = ChrIdxs[chrom1]
            N2 = ChrIdxs[chrom2]
            if (N2 < N1) or (N2 == N1 and b2 < b1):
                N1, N2 = N2, N1
                b1, b2 = b2, b1
                cov1, cov2 = cov2, cov1
            out[(N1, b1, N2, b2)] = (rank, strict, weak, cov1, cov2, dist_bins)
        return out

    for ((chrom1, b1), (chrom2, b2)), (rank, strict, weak, cov1, cov2, dist_bins) in rich_contacts.items():
        if _shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested")
        N1 = ChrIdxs[chrom1]
        N2 = ChrIdxs[chrom2]
        if (N2 < N1) or (N2 == N1 and b2 < b1):
            N1, N2 = N2, N1
            b1, b2 = b2, b1
            cov1, cov2 = cov2, cov1
        out[(N1, b1, N2, b2)] = (rank, strict, weak, cov1, cov2, dist_bins)
    return out


def label_from_keys(keyset: List, bins: Dict) -> str:
    """Generate label from keys."""
    ks = sorted(list(keyset))
    if not ks:
        return '-------'
    if len(ks) < 3:
        return ''.join([f"{bins[k[1]][0]}:{bins[k[1]][1]}-" for k in ks])
    first = ks[0]
    return f"{bins[first[1]][0]}:{bins[first[1]][1]}-"


def iDuplicateContact(Contact_disp_L: List, ObjCoorMP1: Tuple, ObjCoorMP2: Tuple) -> List:
    """Duplicate contact with weights."""
    c = [0, 0, 0, 0, 0, 0, 0]
    base = ObjCoorMP1[0] * ObjCoorMP2[0] * ObjCoorMP1[1] * ObjCoorMP2[1]
    c[-1] = base
    c[0] = base * Contact_disp_L[0]
    c[1] = base * Contact_disp_L[1]
    c[2] = base * Contact_disp_L[2]
    c[3] = base * Contact_disp_L[3]
    c[4] = base * Contact_disp_L[4]
    c[-2] = base * Contact_disp_L[-1]
    return c


def choose_best(writed: Tuple, dupled: Tuple, crit: str) -> bool:
    """Choose best contact based on criteria."""
    if crit == 'coverage':
        return (writed[-2] * writed[-3]) < (dupled[-2] * dupled[-3])
    elif crit == 'deviation':
        return writed[5] < dupled[5]
    elif crit == 'length':
        return (writed[-2] < dupled[-2]) or (dupled[-2] < 0)
    elif crit == 'none':
        return True
    else:
        return writed[-1] < dupled[-1]


def _infer_resolution_from_bins(bins_dict: Dict) -> int:
    """Infer resolution from bins dictionary."""
    widths = np.fromiter((v[2] - v[1] for v in bins_dict.values()), dtype=np.int64)
    if widths.size == 0:
        return 1
    vals, cnts = np.unique(widths, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _build_chr_arrays(bins_dict: Dict, idx: Dict) -> Tuple[Dict, Dict, Dict]:
    """Build chromosome arrays for efficient bin lookup."""
    chr_bins = {}
    chr_starts = {}
    chr_ends = {}
    for bid, (chrom, start, end) in bins_dict.items():
        N = idx[chrom]
        chr_bins.setdefault(N, []).append((start, end, bid))
    for N, arr in chr_bins.items():
        arr.sort(key=lambda x: x[0])
        starts = np.array([a[0] for a in arr], dtype=np.int64)
        ends = np.array([a[1] for a in arr], dtype=np.int64)
        bins = [a[2] for a in arr]
        chr_bins[N] = bins
        chr_starts[N] = starts
        chr_ends[N] = ends
    return chr_bins, chr_starts, chr_ends


def _locate_bin_index(center: int, starts_np: np.ndarray, ends_np: np.ndarray) -> int:
    """Locate bin index for a given center position."""
    pos = int(np.searchsorted(starts_np, center, side='right') - 1)
    if pos < 0 or pos >= starts_np.size:
        return -1
    if center >= int(ends_np[pos]):
        return -1
    return pos


def iReadingMarkPoints(
    mark_path: str,
    ChrIdxs1: Dict,
    ChrIdxs2: Dict,
    agg_bp: int,
    A_chr_bins: Dict,
    A_chr_starts: Dict,
    A_chr_ends: Dict,
    B_chr_bins: Dict,
    B_chr_starts: Dict,
    B_chr_ends: Dict,
    resB: int
) -> Dict:
    """
    Read mark points with optimized memory usage.
    Uses list-based storage instead of OrderedDict where appropriate.
    """
    frame = max(1, int(round(agg_bp / max(resB, 1))))
    rev = {}
    open_kwargs = {'buffering': 1024 * 1024} if sys.platform.startswith('linux') else {}
    with open(mark_path, 'r', **open_kwargs) as f:
        for line in f:
            if _shutdown_requested:
                raise KeyboardInterrupt("Shutdown requested")
            if not line.strip():
                continue
            a = line.split()
            N1 = ChrIdxs1.get(a[0])
            N2 = ChrIdxs2.get(a[3])
            if N1 is None or N2 is None:
                continue
            c1 = (int(a[1]) + int(a[2])) // 2
            c2 = (int(a[4]) + int(a[5])) // 2
            Astarts = A_chr_starts.get(N1)
            Aends = A_chr_ends.get(N1)
            Bstarts = B_chr_starts.get(N2)
            Bends = B_chr_ends.get(N2)
            if Astarts is None or Bstarts is None:
                continue
            idx1 = _locate_bin_index(c1, Astarts, Aends)
            idx2 = _locate_bin_index(c2, Bstarts, Bends)
            if idx1 < 0 or idx2 < 0:
                continue
            b1 = A_chr_bins[N1][idx1]
            b2 = B_chr_bins[N2][idx2]
            rev.setdefault((N2, b2), []).append((N1, b1))
    
    fwd = {}
    for (N2, b2), src_list in rev.items():
        src_set = set(src_list)
        w = 1.0 / max(len(src_set), 1)
        for (N1, b1) in src_set:
            fwd.setdefault((N1, b1), []).append((N2, b2, 1.0, w))
    
    del rev
    
    Obj = {}
    for src_key, triples in fwd.items():
        by_chr = {}
        for (N2, b2, cnt, w) in triples:
            by_chr.setdefault(N2, []).append((b2, cnt, w))
        groups = []
        for N2, arr in by_chr.items():
            arr.sort()
            cur = []
            last_b = None
            for (b2, cnt, w) in arr:
                if last_b is None or abs(b2 - last_b) <= frame:
                    cur.append((N2, b2, cnt, w))
                else:
                    groups.append(cur)
                    cur = [(N2, b2, cnt, w)]
                last_b = b2
            if cur:
                groups.append(cur)
        normed = []
        for g in groups:
            tot = sum(v[2] for v in g)
            if tot <= 0:
                continue
            og = [((k_N, k_b), (round(v_cnt / tot, 2), round(v_w, 2)))
                  for (k_N, k_b, v_cnt, v_w) in g]
            if og:
                normed.append(dict(og))
        if normed:
            Obj[src_key] = normed
    return Obj


def _stat_init() -> Tuple:
    """Initialize statistics structure."""
    return (
        ['all', 'remappable', 'processed', 'unique', 'duplicated', 'dropped'],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    )


def _stat_merge(a: Tuple, b: Tuple) -> Tuple:
    """Merge two statistics structures."""
    a[1][0] += b[1][0]
    a[1][1] += b[1][1]
    a[1][2] += b[1][2]
    a[1][3] += b[1][3]
    a[1][4] += b[1][4]
    a[1][5] += b[1][5]
    for i in range(6):
        a[2][i] += b[2][i]
    return a


def _stat_finalize(stat: Tuple) -> Tuple:
    """Finalize statistics structure."""
    return stat


def _write_stat(stat_list: Tuple, stat_out_prefix: str) -> None:
    """Write statistics to file."""
    with open(stat_out_prefix + '.stat', 'w') as f:
        for i in range(6):
            f.write(f"{stat_list[0][i]} {stat_list[1][i]} {stat_list[2][i]}\n")


def _iDifferContact_core(
    Contact_disp_0: Dict,
    Contact_disp_1: Dict,
    ObjCoorMP: Dict,
    model: str,
    criteria: str,
    Bbins: Dict
) -> Tuple[Dict, Dict, Tuple]:
    """Core of iDifferContact with optimized memory usage."""
    DifferContact = {}
    Dups = {}
    Statistic = _stat_init()
    
    for i, payload0 in Contact_disp_0.items():
        if _shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested")
        key1 = i[:2]
        key2 = i[2:]
        Statistic[1][0] += 1
        Statistic[2][0] += 1
        if (key1 in ObjCoorMP) and (key2 in ObjCoorMP):
            end1 = len(ObjCoorMP[key1])
            end2 = len(ObjCoorMP[key2])
            Statistic[1][1] += 1
            Statistic[2][1] += 1
            if end1 == 1:
                Statistic[2][3] += 1
            else:
                Statistic[2][4] += 1
            if end2 == 1:
                Statistic[2][3] += 1
            else:
                Statistic[2][4] += 1
            for j1 in range(end1):
                for j2 in range(end2):
                    c = [0, 0, 0, 0, 0, 0, 0, [], []]
                    k_hits = 0
                    for k1, w1 in ObjCoorMP[key1][j1].items():
                        for k2, w2 in ObjCoorMP[key2][j2].items():
                            k_combined = k1 + k2
                            k_combined_rev = k2 + k1
                            if k_combined in Contact_disp_1:
                                dc = iDuplicateContact(Contact_disp_1[k_combined], w1, w2)
                                for t in range(5):
                                    c[t] += dc[t]
                                c[-4] += dc[-2]
                                c[-3] += dc[-1]
                                c[-2].append(k1)
                                c[-1].append(k2)
                                k_hits += 1
                            elif k_combined_rev in Contact_disp_1:
                                dc = iDuplicateContact(Contact_disp_1[k_combined_rev], w1, w2)
                                for t in range(5):
                                    c[t] += dc[t]
                                c[-4] += dc[-2]
                                c[-3] += dc[-1]
                                c[-2].append(k2)
                                c[-1].append(k1)
                                k_hits += 1
                            else:
                                Statistic[2][5] += 1
                    if k_hits == 0:
                        continue
                    Statistic[1][2] += 1
                    Statistic[2][2] += 1
                    norm = 1.0 if model != 'balanced' else (c[-3] if c[-3] != 0 else 1.0)
                    if c[-3] == 0:
                        continue
                    disp1 = max((payload0[2] - payload0[0]), (payload0[0] - payload0[1]))
                    disp2 = max((c[2] - c[0]), (c[0] - c[1]))
                    c[-2] = list(set(c[-2]))
                    c[-1] = list(set(c[-1]))
                    to_write = (
                        label_from_keys(c[-2], Bbins)[:-1],
                        label_from_keys(c[-1], Bbins)[:-1],
                        int(round(payload0[0])),
                        int(round(c[0] / norm)),
                        int(round(disp1)),
                        int(round(disp2)),
                        int(round(payload0[3])),
                        int(round(payload0[4])),
                        int(round(c[3] / norm)),
                        int(round(c[4] / norm)),
                        float(round(c[-4] / norm, 2)),
                        float(round(c[-3], 5)),
                    )
                    if i not in DifferContact:
                        DifferContact[i] = to_write
                        Statistic[1][3] += 1
                    else:
                        Statistic[1][4] += 1
                        if choose_best(DifferContact[i], to_write, criteria):
                            Dups.setdefault(i, []).append(to_write)
                        else:
                            Dups.setdefault(i, []).append(DifferContact[i])
                            DifferContact[i] = to_write
    return DifferContact, Dups, Statistic


_W_CB = None
_W_OBJ = None
_W_MODEL = None
_W_CRIT = None
_W_BBINS = None


def _init_worker(cb: Dict, obj: Dict, model: str, criteria: str, bbins: Dict) -> None:
    """Initialize worker process with shared data."""
    global _W_CB, _W_OBJ, _W_MODEL, _W_CRIT, _W_BBINS
    _W_CB = cb
    _W_OBJ = obj
    _W_MODEL = model
    _W_CRIT = criteria
    _W_BBINS = bbins


def _iDifferContact_worker(sub_contacts: Dict) -> Tuple[Dict, Dict, Tuple]:
    """Worker function for parallel processing."""
    return _iDifferContact_core(sub_contacts, _W_CB, _W_OBJ, _W_MODEL, _W_CRIT, _W_BBINS)


def _calculate_optimal_threads(
    Contact_disp_1: Dict,
    ObjCoorMP: Dict,
    nthreads: int
) -> int:
    """Calculate optimal number of threads based on available memory."""
    if nthreads <= 1:
        return 1
    
    try:
        mem_limit = _get_memory_limit()
        
        est_per_thread = (
            len(Contact_disp_1) * 500 +
            len(ObjCoorMP) * 2000
        )
        
        max_safe_threads = max(1, int(mem_limit / max(est_per_thread, 1024 * 1024)))
        safe_nthreads = min(nthreads, max_safe_threads)
        
        if est_per_thread > 2 * 1024 * 1024 * 1024:
            safe_nthreads = 1
        
        logger.info(f"Memory limit: {mem_limit / (1024**3):.1f}GB, "
                   f"estimated per thread: {est_per_thread / (1024**3):.2f}GB, "
                   f"optimal threads: {safe_nthreads}")
        
        return safe_nthreads
    except Exception as e:
        logger.warning(f"Failed to calculate optimal threads: {e}, using {nthreads}")
        return nthreads


def iDifferContact_parallel(
    Contact_disp_0: Dict,
    Contact_disp_1: Dict,
    ObjCoorMP: Dict,
    model: str,
    criteria: str,
    ChrIdxs2: Dict,
    Bbins: Dict,
    stat_out_prefix: str,
    nthreads: int
) -> Tuple[Dict, Dict]:
    """
    Parallel version of iDifferContact.
    Automatically uses fork mode on Linux for better memory efficiency.
    """
    safe_nthreads = _calculate_optimal_threads(Contact_disp_1, ObjCoorMP, nthreads)

    if safe_nthreads <= 1 or len(Contact_disp_0) == 0:
        DifferContact, Dups, Statistic = _iDifferContact_core(
            Contact_disp_0, Contact_disp_1, ObjCoorMP, model, criteria, Bbins
        )
        stat_list = _stat_finalize(Statistic)
        _write_stat(stat_list, stat_out_prefix)
        return DifferContact, Dups
    
    buckets = [dict() for _ in range(safe_nthreads)]
    for idx, (k, v) in enumerate(Contact_disp_0.items()):
        buckets[idx % safe_nthreads][k] = v
    buckets = [b for b in buckets if b]
    
    logger.info(f"Starting parallel processing with {len(buckets)} processes")
    
    pool = None
    try:
        if sys.platform.startswith('linux'):
            try:
                ctx = get_context('fork')
                pool = ctx.Pool(
                    processes=len(buckets),
                    initializer=_init_worker,
                    initargs=(Contact_disp_1, ObjCoorMP, model, criteria, Bbins)
                )
            except (AttributeError, ValueError):
                logger.warning("Fork mode not available, falling back to spawn")
                pool = mp.Pool(
                    processes=len(buckets),
                    initializer=_init_worker,
                    initargs=(Contact_disp_1, ObjCoorMP, model, criteria, Bbins)
                )
        else:
            pool = mp.Pool(
                processes=len(buckets),
                initializer=_init_worker,
                initargs=(Contact_disp_1, ObjCoorMP, model, criteria, Bbins)
            )
        
        results = pool.map(_iDifferContact_worker, buckets)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    
    merged = {}
    merged_dups = {}
    stat = _stat_init()
    for diff, dups, st in results:
        merged.update(diff)
        for k, v in dups.items():
            merged_dups.setdefault(k, []).extend(v)
        _stat_merge(stat, st)
    
    stat_list = _stat_finalize(stat)
    _write_stat(stat_list, stat_out_prefix)
    
    logger.info(f"Parallel processing complete, merged {len(merged)} contacts")
    return merged, merged_dups


def iPrintDifferContact(data: Dict, Abins: Dict, out_prefix: str) -> None:
    """Print differentiated contacts to file."""
    out_path = out_prefix + '.liftContacts'
    open_kwargs = {'buffering': 1024 * 1024} if sys.platform.startswith('linux') else {}
    with open(out_path, 'w', **open_kwargs) as f:
        header = (
            'chr1_observed\tpos1_observed\tchr2_observed\tpos2_observed\t'
            'remap1_target\tremap2_target\tobserved_contacts\ttarget_contacts\t'
            'observed_deviations\ttarget_deviations\t'
            'observed_coverages_pos1\tobserved_coverages_pos2\t'
            'target_coverages_pos1\ttarget_coverages_pos2\t'
            'target_contact_distances\tremapping_coverages'
        )
        f.write(header + "\n")
        Keys = sorted(list(data.keys()), key=lambda x: (x[0], x[2], x[1], x[3]))
        for key in Keys:
            i = data[key]
            c1name, c1start = Abins[key[1]][0], Abins[key[1]][1]
            c2name, c2start = Abins[key[3]][0], Abins[key[3]][1]
            line = (
                f"{c1name}\t{c1start}\t{c2name}\t{c2start}\t"
                f"{i[0]}\t{i[1]}\t{i[2]}\t{i[3]}\t{i[4]}\t{i[5]}\t"
                f"{i[6]}\t{i[7]}\t{i[8]}\t{i[9]}\t{float(i[10]):.2f}\t{float(i[11]):.5f}\n"
            )
            f.write(line)


def iPrintDupsContact(kept: Dict, discarded: Dict, Abins: Dict, out_prefix: str) -> None:
    """Print discarded duplicate contacts to file."""
    if not discarded:
        return
    out_path = out_prefix + '.discarded_dups.tsv'
    open_kwargs = {'buffering': 1024 * 1024} if sys.platform.startswith('linux') else {}
    with open(out_path, 'w', **open_kwargs) as f:
        header = (
            'chr1_observed\tpos1_observed\tchr2_observed\tpos2_observed\t'
            'kept_remap1\tkept_remap2\t'
            'discarded_remap1\tdiscarded_remap2\t'
            'discarded_observed_contacts\tdiscarded_target_contacts\t'
            'discarded_observed_deviations\tdiscarded_target_deviations\t'
            'discarded_observed_coverages_pos1\tdiscarded_observed_coverages_pos2\t'
            'discarded_target_coverages_pos1\tdiscarded_target_coverages_pos2\t'
            'discarded_target_contact_distances\tdiscarded_remapping_coverages'
        )
        f.write(header + "\n")
        for key in sorted(discarded.keys(), key=lambda x: (x[0], x[2], x[1], x[3])):
            kept_item = kept[key]
            c1name, c1start = Abins[key[1]][0], Abins[key[1]][1]
            c2name, c2start = Abins[key[3]][0], Abins[key[3]][1]
            for dup in discarded[key]:
                f.write(
                    f"{c1name}\t{c1start}\t{c2name}\t{c2start}\t"
                    f"{kept_item[0]}\t{kept_item[1]}\t"
                    f"{dup[0]}\t{dup[1]}\t"
                    f"{int(dup[2])}\t{int(dup[3])}\t"
                    f"{int(dup[4])}\t{int(dup[5])}\t"
                    f"{int(dup[6])}\t{int(dup[7])}\t"
                    f"{int(dup[8])}\t{int(dup[9])}\t"
                    f"{float(dup[10]):.2f}\t{float(dup[11]):.5f}\n"
                )


def _write_dups_from_sqlite(conn, dups, Abins, out_prefix):
    """Write discarded duplicates from SQLite results."""
    if not dups:
        return
    out_path = out_prefix + '.discarded_dups.tsv'
    dups_keys_set = set(dups.keys())
    with open(out_path, 'w') as f:
        header = (
            'chr1_observed\tpos1_observed\tchr2_observed\tpos2_observed\t'
            'kept_remap1\tkept_remap2\t'
            'discarded_remap1\tdiscarded_remap2\t'
            'discarded_observed_contacts\tdiscarded_target_contacts\t'
            'discarded_observed_deviations\tdiscarded_target_deviations\t'
            'discarded_observed_coverages_pos1\tdiscarded_observed_coverages_pos2\t'
            'discarded_target_coverages_pos1\tdiscarded_target_coverages_pos2\t'
            'discarded_target_contact_distances\tdiscarded_remapping_coverages'
        )
        f.write(header + "\n")
        cur = conn.execute(
            "SELECT n1,b1,n2,b2,remap1,remap2 FROM results"
        )
        for row in cur:
            key = (row[0], row[1], row[2], row[3])
            if key not in dups_keys_set:
                continue
            if key[1] not in Abins or key[3] not in Abins:
                continue
            c1name, c1start = Abins[key[1]][0], Abins[key[1]][1]
            c2name, c2start = Abins[key[3]][0], Abins[key[3]][1]
            for dup in dups[key]:
                f.write(
                    f"{c1name}\t{c1start}\t{c2name}\t{c2start}\t"
                    f"{row[4]}\t{row[5]}\t"
                    f"{dup[0]}\t{dup[1]}\t"
                    f"{int(dup[2])}\t{int(dup[3])}\t"
                    f"{int(dup[4])}\t{int(dup[5])}\t"
                    f"{int(dup[6])}\t{int(dup[7])}\t"
                    f"{int(dup[8])}\t{int(dup[9])}\t"
                    f"{float(dup[10]):.2f}\t{float(dup[11]):.5f}\n"
                )


def _spill_enabled_for_paths(paths: List[str], threshold_mb: float) -> bool:
    if threshold_mb is None or float(threshold_mb) <= 0:
        return False
    thr = float(threshold_mb)
    for p in paths:
        try:
            if os.path.getsize(p) / (1024.0 * 1024.0) >= thr:
                return True
        except OSError:
            continue
    return False


def _tmp_workspace(tmp_dir: Optional[str]):
    if tmp_dir:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=tmp_dir)
    return tempfile.TemporaryDirectory()


def _stable_shard_id(key: Tuple[int, int, int, int], nshards: int) -> int:
    n1, b1, n2, b2 = key
    mix = (n1 * 1000003 + b1 * 9176 + n2 * 1315423911 + b2 * 2654435761) & 0x7FFFFFFF
    return mix % max(int(nshards), 1)


def _stream_contact_hash_to_shards(
    contacts_path: str,
    ChrIdxs: Dict,
    nshards: int,
    shard_dir: Path
) -> Tuple[List[str], Dict]:
    shard_count = max(1, int(nshards))
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f"ca_shard_{i:04d}.tsv" for i in range(shard_count)]
    handles = [open(p, "w", buffering=1024 * 1024) for p in shard_paths]
    bins = {}
    open_kwargs = {'buffering': 1024 * 1024} if sys.platform.startswith('linux') else {}
    try:
        with open(contacts_path, 'r', **open_kwargs) as f:
            _ = f.readline().strip().split()
            for line in f:
                if _shutdown_requested:
                    raise KeyboardInterrupt("Shutdown requested")
                if not line.strip():
                    continue
                parts = line.split()[:14]
                if len(parts) < 14:
                    continue
                (
                    chrom1, start1, end1, bin1,
                    chrom2, start2, end2, bin2,
                    rank, strict, weak, cov1, cov2, dist_bins
                ) = parts
                try:
                    start1 = int(start1)
                    end1 = int(end1)
                    b1 = int(bin1)
                    start2 = int(start2)
                    end2 = int(end2)
                    b2 = int(bin2)
                    rank_i = int(rank)
                    strict_i = int(strict)
                    weak_i = int(weak)
                    cov1_f = float(cov1)
                    cov2_f = float(cov2)
                    dist_bins_f = float(dist_bins)
                except Exception:
                    continue

                N1 = ChrIdxs.get(chrom1)
                N2 = ChrIdxs.get(chrom2)
                if N1 is None or N2 is None:
                    continue

                if b1 not in bins:
                    bins[b1] = (chrom1, start1, end1)
                if b2 not in bins:
                    bins[b2] = (chrom2, start2, end2)

                # Keep rich-key semantics identical to read_contacts_rich:
                # key ordering is by bin id only; cov ordering is not changed here.
                if b1 <= b2:
                    k_chrom1, k_bin1, k_chrom2, k_bin2 = chrom1, b1, chrom2, b2
                    k_n1, k_n2 = N1, N2
                else:
                    k_chrom1, k_bin1, k_chrom2, k_bin2 = chrom2, b2, chrom1, b1
                    k_n1, k_n2 = N2, N1

                key = (k_n1, k_bin1, k_n2, k_bin2)
                sid = _stable_shard_id(key, shard_count)
                handles[sid].write(
                    f"{k_chrom1}\t{k_bin1}\t{k_chrom2}\t{k_bin2}\t"
                    f"{rank_i}\t{strict_i}\t{weak_i}\t{cov1_f}\t{cov2_f}\t{dist_bins_f}\n"
                )
    finally:
        for h in handles:
            try:
                h.close()
            except Exception:
                pass
    return [str(p) for p in shard_paths], bins


def _load_contact_hash_shard(path: str, ChrIdxs: Dict) -> Dict:
    out = {}
    if not os.path.exists(path):
        return out
    try:
        if os.path.getsize(path) == 0:
            return out
    except OSError:
        return out
    rich = {}
    with open(path, 'r', buffering=1024 * 1024) as f:
        for line in f:
            a = line.split()
            if len(a) < 10:
                continue
            try:
                chrom1 = a[0]
                b1 = int(a[1])
                chrom2 = a[2]
                b2 = int(a[3])
                rank = int(a[4])
                strict = int(a[5])
                weak = int(a[6])
                cov1 = float(a[7])
                cov2 = float(a[8])
                dist_bins = float(a[9])
            except Exception:
                continue
            # Same dedup behavior as read_contacts_rich: later rows overwrite earlier rows.
            rich[(chrom1, b1, chrom2, b2)] = (rank, strict, weak, cov1, cov2, dist_bins)

    for (chrom1, b1, chrom2, b2), (rank, strict, weak, cov1, cov2, dist_bins) in rich.items():
        N1 = ChrIdxs.get(chrom1)
        N2 = ChrIdxs.get(chrom2)
        if N1 is None or N2 is None:
            continue
        if (N2 < N1) or (N2 == N1 and b2 < b1):
            N1, N2 = N2, N1
            b1, b2 = b2, b1
            cov1, cov2 = cov2, cov1
        out[(N1, b1, N2, b2)] = (rank, strict, weak, cov1, cov2, dist_bins)
    return out


def _make_result_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            n1 INTEGER NOT NULL,
            b1 INTEGER NOT NULL,
            n2 INTEGER NOT NULL,
            b2 INTEGER NOT NULL,
            remap1 TEXT NOT NULL,
            remap2 TEXT NOT NULL,
            observed_contacts INTEGER NOT NULL,
            target_contacts INTEGER NOT NULL,
            observed_deviations INTEGER NOT NULL,
            target_deviations INTEGER NOT NULL,
            observed_coverages_pos1 INTEGER NOT NULL,
            observed_coverages_pos2 INTEGER NOT NULL,
            target_coverages_pos1 INTEGER NOT NULL,
            target_coverages_pos2 INTEGER NOT NULL,
            target_contact_distances REAL NOT NULL,
            remapping_coverages REAL NOT NULL,
            PRIMARY KEY (n1, b1, n2, b2)
        )
        """
    )
    return conn


def _insert_differ_rows_sqlite(conn: sqlite3.Connection, differ: Dict) -> None:
    if not differ:
        return
    rows = []
    for key, v in differ.items():
        n1, b1, n2, b2 = key
        rows.append((
            int(n1), int(b1), int(n2), int(b2),
            str(v[0]), str(v[1]),
            int(v[2]), int(v[3]), int(v[4]), int(v[5]),
            int(v[6]), int(v[7]), int(v[8]), int(v[9]),
            float(v[10]), float(v[11]),
        ))
    conn.executemany(
        """
        INSERT OR REPLACE INTO results(
            n1,b1,n2,b2,remap1,remap2,observed_contacts,target_contacts,
            observed_deviations,target_deviations,observed_coverages_pos1,observed_coverages_pos2,
            target_coverages_pos1,target_coverages_pos2,target_contact_distances,remapping_coverages
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()


def _write_differ_from_sqlite(conn: sqlite3.Connection, Abins: Dict, out_prefix: str) -> None:
    out_path = out_prefix + '.liftContacts'
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', buffering=1024 * 1024) as f:
        header = (
            'chr1_observed\tpos1_observed\tchr2_observed\tpos2_observed\t'
            'remap1_target\tremap2_target\tobserved_contacts\ttarget_contacts\t'
            'observed_deviations\ttarget_deviations\t'
            'observed_coverages_pos1\tobserved_coverages_pos2\t'
            'target_coverages_pos1\ttarget_coverages_pos2\t'
            'target_contact_distances\tremapping_coverages'
        )
        f.write(header + "\n")
        cur = conn.execute(
            """
            SELECT n1,b1,n2,b2,remap1,remap2,observed_contacts,target_contacts,
                   observed_deviations,target_deviations,observed_coverages_pos1,observed_coverages_pos2,
                   target_coverages_pos1,target_coverages_pos2,target_contact_distances,remapping_coverages
            FROM results
            ORDER BY n1, n2, b1, b2
            """
        )
        for row in cur:
            _, b1, _, b2 = row[0], row[1], row[2], row[3]
            if b1 not in Abins or b2 not in Abins:
                continue
            c1name, c1start = Abins[b1][0], Abins[b1][1]
            c2name, c2start = Abins[b2][0], Abins[b2][1]
            f.write(
                f"{c1name}\t{c1start}\t{c2name}\t{c2start}\t"
                f"{row[4]}\t{row[5]}\t{int(row[6])}\t{int(row[7])}\t{int(row[8])}\t{int(row[9])}\t"
                f"{int(row[10])}\t{int(row[11])}\t{int(row[12])}\t{int(row[13])}\t"
                f"{float(row[14]):.2f}\t{float(row[15]):.5f}\n"
            )


def iDifferContact(Contact_disp_L, ObjCoorMP1, ObjCoorMP2):
    """Compatibility wrapper for backward compatibility."""
    return iDuplicateContact(Contact_disp_L, ObjCoorMP1, ObjCoorMP2)


def run_liftover(
    contact_a: str,
    contact_b: str,
    fadix_a: str,
    fadix_b: str,
    mark: str,
    agg_frame: int,
    dups_filter: str,
    model: str,
    out_prefix: str,
    nthreads: int = 1,
    tmp_dir: Optional[str] = None,
    spill_threshold_mb: float = 0.0,
    hash_shards: int = 16,
) -> None:
    """
    Main entry point for liftcontacts.
    Optimized for Linux with automatic cross-platform fallback.
    """
    _setup_signal_handlers()

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    if not _is_rich(contact_a) or not _is_rich(contact_b):
        sys.exit("Only RICH contacts (*.contacts.rich.tsv) are accepted.")
    
    logger.info("Indexing chromosomes...")
    idxA = ChromIndexingFAI(fadix_a)
    idxB = ChromIndexingFAI(fadix_b)

    mark_path = mark if mark.endswith('.mark') else (mark + '.mark')
    if not os.path.exists(mark_path):
        sys.exit(f".mark not found: {mark_path}")

    use_spill = _spill_enabled_for_paths([contact_a, contact_b], spill_threshold_mb)
    logger.info(
        f"Spill mode={'on' if use_spill else 'off'} "
        f"(threshold={float(spill_threshold_mb):.2f}MB, shards={max(1, int(hash_shards))})"
    )

    if use_spill:
        with _tmp_workspace(tmp_dir) as td:
            logger.info("Reading contact-b and sharding contact-a...")
            CB_rich, Bbins = read_contacts_rich(contact_b)
            shard_paths, Abins = _stream_contact_hash_to_shards(
                contact_a,
                idxA,
                max(1, int(hash_shards)),
                Path(td) / "ca_shards",
            )

            A_chr_bins2, A_chr_starts, A_chr_ends = _build_chr_arrays(Abins, idxA)
            B_chr_bins2, B_chr_starts, B_chr_ends = _build_chr_arrays(Bbins, idxB)

            logger.info("Building contact hash for contact-b...")
            CB = build_contact_hash_from_rich(CB_rich, idxB, consume=True)
            del CB_rich
            gc.collect()

            resB = _infer_resolution_from_bins(Bbins)
            logger.info("Reading mark points...")
            MP_A_to_B = iReadingMarkPoints(
                mark_path, idxA, idxB, agg_frame,
                A_chr_bins2, A_chr_starts, A_chr_ends,
                B_chr_bins2, B_chr_starts, B_chr_ends, resB
            )
            del A_chr_bins2
            del A_chr_starts
            del A_chr_ends
            del B_chr_bins2
            del B_chr_starts
            del B_chr_ends
            gc.collect()

            conn = _make_result_db(Path(td) / "liftcontacts_spill.sqlite")
            all_dups = {}
            try:
                stat = _stat_init()
                non_empty = 0
                total = len(shard_paths)
                logger.info("Processing sharded contact-a chunks (spill mode runs single-threaded to bound RSS)...")
                for shard_path in shard_paths:
                    if _shutdown_requested:
                        raise KeyboardInterrupt("Shutdown requested")
                    CA_shard = _load_contact_hash_shard(shard_path, idxA)
                    if not CA_shard:
                        continue
                    non_empty += 1
                    logger.info(f"Processing shard {non_empty}/{total} (contacts={len(CA_shard)})")
                    DifferA, shard_dups, Statistic = _iDifferContact_core(
                        CA_shard, CB, MP_A_to_B, model, dups_filter, Bbins
                    )
                    for k, v in shard_dups.items():
                        all_dups.setdefault(k, []).extend(v)
                    _stat_merge(stat, Statistic)
                    _insert_differ_rows_sqlite(conn, DifferA)
                    del CA_shard
                    del DifferA
                    gc.collect()

                stat_list = _stat_finalize(stat)
                _write_stat(stat_list, out_prefix)

                del CB
                del MP_A_to_B
                gc.collect()

                logger.info("Writing output...")
                _write_differ_from_sqlite(conn, Abins, out_prefix)
                if all_dups:
                    logger.info(f"Writing {len(all_dups)} duplicate entries...")
                    _write_dups_from_sqlite(conn, all_dups, Abins, out_prefix)
            finally:
                conn.close()

        logger.info("Done!")
        return

    logger.info("Reading contact-a...")
    CA_rich, Abins = read_contacts_rich(contact_a)
    logger.info("Building contact hash for contact-a...")
    CA = build_contact_hash_from_rich(CA_rich, idxA, consume=True)
    del CA_rich
    gc.collect()

    logger.info("Reading contact-b...")
    CB_rich, Bbins = read_contacts_rich(contact_b)
    logger.info("Building contact hash for contact-b...")
    CB = build_contact_hash_from_rich(CB_rich, idxB, consume=True)
    del CB_rich
    gc.collect()

    A_chr_bins2, A_chr_starts, A_chr_ends = _build_chr_arrays(Abins, idxA)
    B_chr_bins2, B_chr_starts, B_chr_ends = _build_chr_arrays(Bbins, idxB)

    resB = _infer_resolution_from_bins(Bbins)

    logger.info("Reading mark points...")
    MP_A_to_B = iReadingMarkPoints(
        mark_path, idxA, idxB, agg_frame,
        A_chr_bins2, A_chr_starts, A_chr_ends,
        B_chr_bins2, B_chr_starts, B_chr_ends, resB
    )
    # Bin-index helper structures are no longer needed after mapping is built.
    del A_chr_bins2
    del A_chr_starts
    del A_chr_ends
    del B_chr_bins2
    del B_chr_starts
    del B_chr_ends
    gc.collect()

    nthreads_val = int(nthreads)

    DupsA = None
    if nthreads_val > 1:
        logger.info(f"Running in parallel mode with {nthreads_val} threads...")
        DifferA, DupsA = iDifferContact_parallel(
            CA, CB, MP_A_to_B, model, dups_filter, idxB, Bbins, out_prefix, nthreads_val
        )
    else:
        logger.info("Running in single-threaded mode...")
        DifferA, DupsA, Statistic = _iDifferContact_core(
            CA, CB, MP_A_to_B, model, dups_filter, Bbins
        )
        stat_list = _stat_finalize(Statistic)
        _write_stat(stat_list, out_prefix)

    # Large in-memory hashes can be released before output write.
    del CA
    del CB
    del MP_A_to_B
    gc.collect()

    logger.info("Writing output...")
    iPrintDifferContact(DifferA, Abins, out_prefix)
    if DupsA:
        logger.info(f"Writing {len(DupsA)} duplicate entries...")
        iPrintDupsContact(DifferA, DupsA, Abins, out_prefix)
    logger.info("Done!")


def _invert_mark(mark_path: str, out_path: str) -> None:
    """Invert mark file direction."""
    with open(mark_path, 'r') as f, open(out_path, "w") as fo:
        for line in f:
            if not line.strip():
                continue
            a = line.split()
            if len(a) < 8:
                continue
            ca, sa, ea, cb, sb, eb, d, tag = a[:8]
            try:
                d = str(-int(d))
            except Exception:
                pass
            fo.write(f"{cb}\t{sb}\t{eb}\t{ca}\t{sa}\t{ea}\t{d}\t{tag}\n")


def main():
    """Main function for compatibility wrapper."""
    p = argparse.ArgumentParser(
        prog='liftcontracts',
        description='[Deprecated alias for liftcontacts] Run A->B and B->A liftover.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--contact-a', help='A: .contacts.rich.tsv')
    p.add_argument('--contact-b', help='B: .contacts.rich.tsv')
    p.add_argument('--matrix-a-prefix', help='Batch mode: prefix for A multi-resolution .contacts.tsv')
    p.add_argument('--matrix-b-prefix', help='Batch mode: prefix for B multi-resolution .contacts.tsv')
    p.add_argument('--fadix-a', required=True, help='FASTA index (.fai) for A')
    p.add_argument('--fadix-b', required=True, help='FASTA index (.fai) for B')
    p.add_argument('--mark', required=True, help='Path to A->B .mark')
    p.add_argument('--mark-ba', help='Path to B->A .mark (optional; if not provided, invert --mark)')
    p.add_argument('--agg-frame', type=int, default=150000, help='Aggregation frame on B (bp) to merge adjacent remapped bins')
    p.add_argument('--dups-filter', choices=['length','coverage','deviation','none','default'], default='default', help='Duplicate selection rule')
    p.add_argument('--model', choices=['balanced','raw'], default='raw', help='Normalization model')
    p.add_argument('--tmp-dir', help='Directory for temporary spill files')
    p.add_argument('--spill-threshold-mb', type=float, default=0.0, help='Enable spill mode when contact file size >= this MB (0 disables)')
    p.add_argument('--hash-shards', type=int, default=16, help='Shard count used by spill mode')
    p.add_argument("--interactive", default="on", choices=["on","off"], help="Write interactive Plotly HTML for reciprocal-summary outputs")
    p.add_argument('--out-prefix', required=True, help='Output prefix')
    a = p.parse_args()

    print("[WARN] 'liftcontracts' is a deprecated alias. Forwarding to 'liftcontacts'.")

    fwd = [
        'coolsecture', 'liftcontacts',
        '--fadix-a', a.fadix_a,
        '--fadix-b', a.fadix_b,
        '--mark-ab', a.mark,
        '--out-prefix', a.out_prefix,
        '--agg-frame', str(a.agg_frame),
        '--dups-filter', a.dups_filter,
        '--model', a.model,
        '--spill-threshold-mb', str(a.spill_threshold_mb),
        '--hash-shards', str(a.hash_shards),
        '--interactive', a.interactive,
        '--no-tags',
    ]
    if a.tmp_dir:
        fwd += ['--tmp-dir', a.tmp_dir]
    if a.mark_ba:
        fwd += ['--mark-ba', a.mark_ba]
    if a.contact_a:
        fwd += ['--contact-a', a.contact_a]
    if a.contact_b:
        fwd += ['--contact-b', a.contact_b]
    if a.matrix_a_prefix:
        fwd += ['--matrix-a-prefix', a.matrix_a_prefix]
    if a.matrix_b_prefix:
        fwd += ['--matrix-b-prefix', a.matrix_b_prefix]

    from . import bidirectional as _b
    import sys as _sys
    _sys.argv = fwd
    return _b.main()


if __name__ == '__main__':
    main()
