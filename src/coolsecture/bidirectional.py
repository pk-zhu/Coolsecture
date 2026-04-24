#!/usr/bin/env python3
import argparse
import os
import sqlite3
import tempfile
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

from .liftcontacts import run_liftover
from .post_common import (
    infer_resolution_from_liftover, ChromIndexingFAI, read_contacts,
    _parse_remap_token, _read_lift_as_map, _reciprocal_stats,
    _import_plotly, _resolve_interactive, _write_plotly_html,
)
from .metric import metricCalc

def _pair_key(order, chrom1, pos1, chrom2, pos2):
    left = (order.get(chrom1, 10**12), chrom1, int(pos1))
    right = (order.get(chrom2, 10**12), chrom2, int(pos2))
    if left <= right:
        return (chrom1, int(pos1), chrom2, int(pos2)), False
    return (chrom2, int(pos2), chrom1, int(pos1)), True

def _swap_row_orientation(row):
    row = list(row)
    for i, j in ((0, 2), (1, 3), (4, 5), (10, 11), (12, 13)):
        row[i], row[j] = row[j], row[i]
    return row

def _canonicalize_row(row, order):
    key, swapped = _pair_key(order, row[0], int(row[1]), row[2], int(row[3]))
    if swapped:
        row = _swap_row_orientation(row)
    return key, row

def _control_key_from_row(row, order, res):
    rm1 = _parse_remap_token(row[4])
    rm2 = _parse_remap_token(row[5])
    if not rm1 or not rm2:
        return None
    p1 = (int(rm1[1]) // res) * res
    p2 = (int(rm2[1]) // res) * res
    key, _ = _pair_key(order, rm1[0], p1, rm2[0], p2)
    return key

def _interval_token(chrom, pos, res):
    start = int(pos)
    return f"{chrom}:{start}-{chrom}:{start + int(res)}"

def _row_score(row):
    try:
        score = float(row[15])
        return score if np.isfinite(score) else -1.0
    except Exception:
        return -1.0

def _quality_tuple(is_consistent: bool, score: float, source: str):
    pref = 1 if source == "ab" else 0
    return (1 if is_consistent else 0, float(score), pref)

def _is_reciprocal_from_ab_row(row_ab, key_a, ba_rows, order_a, order_b, res_ab, res_ba):
    key_b = _control_key_from_row(row_ab, order_b, res_ab)
    if key_b is None:
        return False
    row_ba = ba_rows.get(key_b)
    if row_ba is None:
        return False
    back_key = _control_key_from_row(row_ba, order_a, res_ba)
    return back_key == key_a

def _is_reciprocal_from_ba_row(row_ba, key_a, ab_rows_by_b, order_a, order_b):
    key_b, _ = _pair_key(order_b, row_ba[0], int(row_ba[1]), row_ba[2], int(row_ba[3]))
    row_ab = ab_rows_by_b.get(key_b)
    if row_ab is None:
        return False
    ab_key, _ = _pair_key(order_a, row_ab[0], int(row_ab[1]), row_ab[2], int(row_ab[3]))
    return ab_key == key_a

def _convert_ba_row_to_a(row_ba, order_a, res_a, res_b):
    rm1 = _parse_remap_token(row_ba[4])
    rm2 = _parse_remap_token(row_ba[5])
    if not rm1 or not rm2:
        return None, None

    a1 = (rm1[0], (int(rm1[1]) // res_a) * res_a)
    a2 = (rm2[0], (int(rm2[1]) // res_a) * res_a)
    b1 = (row_ba[0], int(row_ba[1]))
    b2 = (row_ba[2], int(row_ba[3]))

    try:
        b_dist = abs(b2[1] - b1[1]) / 1e6 if b1[0] == b2[0] else -1.0
    except Exception:
        b_dist = -1.0

    row_a = [
        a1[0], str(a1[1]), a2[0], str(a2[1]),
        _interval_token(b1[0], b1[1], res_b),
        _interval_token(b2[0], b2[1], res_b),
        row_ba[7], row_ba[6],
        row_ba[9], row_ba[8],
        row_ba[12], row_ba[13],
        row_ba[10], row_ba[11],
        f"{b_dist:.2f}", row_ba[15],
    ]
    key_a, row_a = _canonicalize_row(row_a, order_a)
    return key_a, row_a

def _parse_liftover_row(line: str):
    row = line.rstrip("\n").split("\t")
    if len(row) < 16:
        row = line.split()
    if len(row) < 16:
        return None
    return row[:16]


def _spill_enabled_for_files(paths, threshold_mb: float) -> bool:
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


def _tmp_workspace(tmp_dir: str):
    if tmp_dir:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=tmp_dir)
    return tempfile.TemporaryDirectory()


def _make_merge_db(path: Path) -> sqlite3.Connection:
    row_defs = ", ".join([f"r{i} TEXT NOT NULL" for i in range(16)])
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS ba_rows(
            c1 TEXT NOT NULL,
            p1 INTEGER NOT NULL,
            c2 TEXT NOT NULL,
            p2 INTEGER NOT NULL,
            ord INTEGER NOT NULL,
            score REAL NOT NULL,
            {row_defs},
            PRIMARY KEY(c1, p1, c2, p2)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS ab_rows(
            c1 TEXT NOT NULL,
            p1 INTEGER NOT NULL,
            c2 TEXT NOT NULL,
            p2 INTEGER NOT NULL,
            score REAL NOT NULL,
            {row_defs},
            PRIMARY KEY(c1, p1, c2, p2)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ab_keys(
            c1 TEXT NOT NULL,
            p1 INTEGER NOT NULL,
            c2 TEXT NOT NULL,
            p2 INTEGER NOT NULL,
            PRIMARY KEY(c1, p1, c2, p2)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS merged_rows(
            c1 TEXT NOT NULL,
            p1 INTEGER NOT NULL,
            c2 TEXT NOT NULL,
            p2 INTEGER NOT NULL,
            n1 INTEGER NOT NULL,
            n2 INTEGER NOT NULL,
            source TEXT NOT NULL,
            q_cons INTEGER NOT NULL,
            q_score REAL NOT NULL,
            q_pref INTEGER NOT NULL,
            {row_defs},
            PRIMARY KEY(c1, p1, c2, p2)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_merged_order ON merged_rows(n1, p1, n2, p2)")
    return conn


def _write_merged_liftover_spill(
    lift_ab: str,
    lift_ba: str,
    fadix_a: str,
    fadix_b: str,
    out_path: str,
    tmp_dir: str,
):
    logger.info("Writing merged liftover in spill mode...")
    res_ab = infer_resolution_from_liftover(lift_ab)
    res_ba = infer_resolution_from_liftover(lift_ba)
    order_a = ChromIndexingFAI(fadix_a)
    order_b = ChromIndexingFAI(fadix_b)
    row_cols = [f"r{i}" for i in range(16)]
    row_col_sql = ", ".join(row_cols)
    row_qmarks = ", ".join(["?"] * 16)
    row_update_sql = ", ".join([f"{c}=excluded.{c}" for c in row_cols])
    header = None
    header_ba = None
    reciprocal_seed = 0
    replaced = 0
    recovered = 0
    skipped_unparseable = 0

    with _tmp_workspace(tmp_dir) as td:
        conn = _make_merge_db(Path(td) / "merged_spill.sqlite")
        try:
            ba_upsert_sql = (
                f"INSERT INTO ba_rows(c1,p1,c2,p2,ord,score,{row_col_sql}) "
                f"VALUES(?,?,?,?,?,?,{row_qmarks}) "
                "ON CONFLICT(c1,p1,c2,p2) DO UPDATE SET "
                f"ord=ba_rows.ord,score=excluded.score,{row_update_sql} "
                "WHERE excluded.score > ba_rows.score"
            )
            ab_upsert_sql = (
                f"INSERT INTO ab_rows(c1,p1,c2,p2,score,{row_col_sql}) "
                f"VALUES(?,?,?,?,?,{row_qmarks}) "
                "ON CONFLICT(c1,p1,c2,p2) DO UPDATE SET "
                f"score=excluded.score,{row_update_sql} "
                "WHERE excluded.score > ab_rows.score"
            )
            ab_key_insert_sql = "INSERT OR IGNORE INTO ab_keys(c1,p1,c2,p2) VALUES(?,?,?,?)"
            merged_upsert_sql = (
                f"INSERT INTO merged_rows(c1,p1,c2,p2,n1,n2,source,q_cons,q_score,q_pref,{row_col_sql}) "
                f"VALUES(?,?,?,?,?,?,?,?,?,?,{row_qmarks}) "
                "ON CONFLICT(c1,p1,c2,p2) DO UPDATE SET "
                f"n1=excluded.n1,n2=excluded.n2,source=excluded.source,"
                "q_cons=excluded.q_cons,q_score=excluded.q_score,q_pref=excluded.q_pref,"
                f"{row_update_sql} "
                "WHERE "
                "excluded.q_cons > merged_rows.q_cons OR "
                "(excluded.q_cons = merged_rows.q_cons AND excluded.q_score > merged_rows.q_score) OR "
                "(excluded.q_cons = merged_rows.q_cons AND excluded.q_score = merged_rows.q_score AND excluded.q_pref > merged_rows.q_pref)"
            )

            buf = []
            ab_buf = []
            ab_key_buf = []
            ord_idx = 0
            with open(lift_ba, "r", buffering=1024 * 1024) as f:
                header_ba = f.readline().rstrip("\n")
                for line in f:
                    row = _parse_liftover_row(line)
                    if row is None:
                        continue
                    _, row = _canonicalize_row(row, order_b)
                    key_b, _ = _pair_key(order_b, row[0], int(row[1]), row[2], int(row[3]))
                    buf.append((
                        key_b[0], int(key_b[1]), key_b[2], int(key_b[3]), int(ord_idx), _row_score(row), *row
                    ))
                    ord_idx += 1
                    if len(buf) >= 50000:
                        conn.executemany(ba_upsert_sql, buf)
                        conn.commit()
                        buf.clear()
            if buf:
                conn.executemany(ba_upsert_sql, buf)
                conn.commit()
                buf.clear()

            with open(lift_ab, "r", buffering=1024 * 1024) as f:
                header = f.readline().rstrip("\n")
                for line in f:
                    row = _parse_liftover_row(line)
                    if row is None:
                        continue
                    key_a, row = _canonicalize_row(row, order_a)
                    score = _row_score(row)
                    key_b = _control_key_from_row(row, order_b, res_ab)
                    is_consistent = False
                    if key_b is not None:
                        row_ba = conn.execute(
                            f"SELECT {row_col_sql} FROM ba_rows WHERE c1=? AND p1=? AND c2=? AND p2=?",
                            (key_b[0], int(key_b[1]), key_b[2], int(key_b[3])),
                        ).fetchone()
                        if row_ba is not None:
                            back_key = _control_key_from_row(list(row_ba), order_a, res_ba)
                            is_consistent = back_key == key_a
                            ab_buf.append((
                                key_b[0], int(key_b[1]), key_b[2], int(key_b[3]), score, *row
                            ))
                    q_cons, q_score, q_pref = _quality_tuple(is_consistent, score, "ab")
                    n1 = int(order_a.get(key_a[0], 10**12))
                    n2 = int(order_a.get(key_a[2], 10**12))
                    buf.append((
                        key_a[0], int(key_a[1]), key_a[2], int(key_a[3]),
                        n1, n2, "ab", q_cons, q_score, q_pref, *row
                    ))
                    ab_key_buf.append((key_a[0], int(key_a[1]), key_a[2], int(key_a[3])))
                    if len(buf) >= 20000:
                        conn.executemany(merged_upsert_sql, buf)
                        if ab_buf:
                            conn.executemany(ab_upsert_sql, ab_buf)
                            ab_buf.clear()
                        if ab_key_buf:
                            conn.executemany(ab_key_insert_sql, ab_key_buf)
                            ab_key_buf.clear()
                        conn.commit()
                        buf.clear()
            if buf:
                conn.executemany(merged_upsert_sql, buf)
                if ab_buf:
                    conn.executemany(ab_upsert_sql, ab_buf)
                    ab_buf.clear()
                if ab_key_buf:
                    conn.executemany(ab_key_insert_sql, ab_key_buf)
                    ab_key_buf.clear()
                conn.commit()
                buf.clear()

            reciprocal_seed = int(
                conn.execute("SELECT COUNT(*) FROM merged_rows WHERE source='ab' AND q_cons=1").fetchone()[0]
            )

            cur = conn.execute(f"SELECT {row_col_sql} FROM ba_rows ORDER BY ord")
            for item in cur:
                row_ba = list(item)
                key_a, row_a = _convert_ba_row_to_a(row_ba, order_a, res_ab, res_ba)
                if key_a is None or row_a is None:
                    skipped_unparseable += 1
                    continue
                key_b_ba, _ = _pair_key(order_b, row_ba[0], int(row_ba[1]), row_ba[2], int(row_ba[3]))
                row_ab = conn.execute(
                    f"SELECT {row_col_sql} FROM ab_rows WHERE c1=? AND p1=? AND c2=? AND p2=?",
                    (key_b_ba[0], int(key_b_ba[1]), key_b_ba[2], int(key_b_ba[3])),
                ).fetchone()
                is_consistent = False
                if row_ab is not None:
                    ab_key, _ = _pair_key(order_a, row_ab[0], int(row_ab[1]), row_ab[2], int(row_ab[3]))
                    is_consistent = ab_key == key_a
                q_cons, q_score, q_pref = _quality_tuple(is_consistent, _row_score(row_a), "ba")
                n1 = int(order_a.get(key_a[0], 10**12))
                n2 = int(order_a.get(key_a[2], 10**12))
                buf.append((
                    key_a[0], int(key_a[1]), key_a[2], int(key_a[3]),
                    n1, n2, "ba", q_cons, q_score, q_pref, *row_a
                ))
                if len(buf) >= 20000:
                    conn.executemany(merged_upsert_sql, buf)
                    conn.commit()
                    buf.clear()
            if buf:
                conn.executemany(merged_upsert_sql, buf)
                conn.commit()
                buf.clear()

            merged_total = int(conn.execute("SELECT COUNT(*) FROM merged_rows").fetchone()[0])
            ba_total = int(
                conn.execute("SELECT COUNT(*) FROM merged_rows WHERE source='ba'").fetchone()[0]
            )
            replaced = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM merged_rows m
                    JOIN ab_keys a
                      ON a.c1=m.c1 AND a.p1=m.p1 AND a.c2=m.c2 AND a.p2=m.p2
                    WHERE m.source='ba'
                    """
                ).fetchone()[0]
            )
            recovered = ba_total - replaced

            with open(out_path, "w", buffering=1024 * 1024) as fo:
                fo.write((header or header_ba or "") + "\n")
                out_cur = conn.execute(
                    f"SELECT {row_col_sql} FROM merged_rows ORDER BY n1, p1, n2, p2"
                )
                for row in out_cur:
                    fo.write("\t".join(str(x) for x in row) + "\n")
        finally:
            conn.close()

    logger.info(f"[OK] {out_path}")
    logger.info(
        f"[INFO] merged_contacts={merged_total} reciprocal_seed={reciprocal_seed} "
        f"recovered_from_BtoA={recovered} replaced_AtoB_by_BtoA={replaced} "
        f"skipped_unparseable={skipped_unparseable}"
    )


def _write_merged_liftover(
    lift_ab: str,
    lift_ba: str,
    fadix_a: str,
    fadix_b: str,
    out_path: str,
    tmp_dir: str = None,
    spill_threshold_mb: float = 0.0,
):
    if _spill_enabled_for_files([lift_ab, lift_ba], spill_threshold_mb):
        return _write_merged_liftover_spill(
            lift_ab, lift_ba, fadix_a, fadix_b, out_path, tmp_dir=tmp_dir
        )

    logger.info("Writing merged liftover...")
    res_ab = infer_resolution_from_liftover(lift_ab)
    res_ba = infer_resolution_from_liftover(lift_ba)
    order_a = ChromIndexingFAI(fadix_a)
    order_b = ChromIndexingFAI(fadix_b)

    merged = OrderedDict()
    reciprocal_seed = 0
    replaced = 0
    recovered = 0
    skipped_unparseable = 0
    header = None

    ba_rows = {}
    with open(lift_ba, 'r', buffering=1024 * 1024) as f:
        header_ba = f.readline().rstrip("\n")
        for line in f:
            row = _parse_liftover_row(line)
            if row is None:
                continue
            _, row = _canonicalize_row(row, order_b)
            key_b, _ = _pair_key(order_b, row[0], int(row[1]), row[2], int(row[3]))
            prev = ba_rows.get(key_b)
            if prev is None or _row_score(row) > _row_score(prev):
                ba_rows[key_b] = row

    ab_rows_by_b = {}
    ab_candidates = {}
    with open(lift_ab, 'r', buffering=1024 * 1024) as f:
        header = f.readline().rstrip("\n")
        for line in f:
            row = _parse_liftover_row(line)
            if row is None:
                continue
            key_a, row = _canonicalize_row(row, order_a)
            score = _row_score(row)
            key_b = _control_key_from_row(row, order_b, res_ab)
            if key_b is not None:
                prev_b = ab_rows_by_b.get(key_b)
                if prev_b is None or score > _row_score(prev_b):
                    ab_rows_by_b[key_b] = row
            is_consistent = _is_reciprocal_from_ab_row(
                row, key_a, ba_rows, order_a, order_b, res_ab, res_ba
            )
            quality = _quality_tuple(is_consistent, score, "ab")
            prev = ab_candidates.get(key_a)
            if prev is None or quality > prev["quality"]:
                ab_candidates[key_a] = {"row": row, "quality": quality}

    reciprocal_seed = sum(1 for item in ab_candidates.values() if item["quality"][0] == 1)

    ba_candidates = {}
    for row_ba in ba_rows.values():
        key_a, row_a = _convert_ba_row_to_a(row_ba, order_a, res_ab, res_ba)
        if key_a is None or row_a is None:
            skipped_unparseable += 1
            continue
        is_consistent = _is_reciprocal_from_ba_row(
            row_ba, key_a, ab_rows_by_b, order_a, order_b
        )
        quality = _quality_tuple(is_consistent, _row_score(row_a), "ba")
        prev = ba_candidates.get(key_a)
        if prev is None or quality > prev["quality"]:
            ba_candidates[key_a] = {"row": row_a, "quality": quality}

    for key_a, item in ab_candidates.items():
        merged[key_a] = {"row": item["row"], "quality": item["quality"], "source": "ab"}

    for key_a, item in ba_candidates.items():
        prev = merged.get(key_a)
        if prev is None:
            merged[key_a] = {"row": item["row"], "quality": item["quality"], "source": "ba"}
            recovered += 1
            continue
        if item["quality"] > prev["quality"]:
            merged[key_a] = {"row": item["row"], "quality": item["quality"], "source": "ba"}
            replaced += 1

    with open(out_path, "w", buffering=1024 * 1024) as fo:
        fo.write((header or header_ba or "") + "\n")
        for key in sorted(merged.keys(), key=lambda x: (order_a.get(x[0], 10**12), x[1], order_a.get(x[2], 10**12), x[3])):
            fo.write("\t".join(str(x) for x in merged[key]["row"]) + "\n")
    logger.info(f"[OK] {out_path}")
    logger.info(
        f"[INFO] merged_contacts={len(merged)} reciprocal_seed={reciprocal_seed} "
        f"recovered_from_BtoA={recovered} replaced_AtoB_by_BtoA={replaced} "
        f"skipped_unparseable={skipped_unparseable}"
    )

def _invert_mark(mark_path: str, out_path: str):
    with open(mark_path) as f, open(out_path, "w") as fo:
        for line in f:
            if not line.strip():
                continue
            a = line.split()
            if len(a) < 8:
                continue
            # swap A and B, flip direction
            ca, sa, ea, cb, sb, eb, d, tag = a[:8]
            d = str(-int(d)) if d.strip().lstrip("+-").isdigit() else d
            fo.write(f"{cb}\t{sb}\t{eb}\t{ca}\t{sa}\t{ea}\t{d}\t{tag}\n")

def _mean_pbad_from_liftover(lift_path: str, fadix: str, frame: int):
    res = infer_resolution_from_liftover(lift_path)
    Order = ChromIndexingFAI(fadix)
    contacts = read_contacts(lift_path, Order, res, short=False)
    vals = metricCalc(contacts, res, frame=frame, metric="pbad")
    if not vals:
        return 0.0
    arr = np.array([v[3] for v in vals if len(v) == 4 and np.isfinite(v[3])], dtype=float)
    return float(np.mean(arr)) if arr.size else 0.0


def _should_compute_pbad(lift_ab: str, lift_ba: str, mode: str, auto_threshold_mb: float):
    mode = (mode or "auto").lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    threshold = float(auto_threshold_mb)
    if threshold <= 0:
        return True
    for p in (lift_ab, lift_ba):
        try:
            sz_mb = os.path.getsize(p) / (1024.0 * 1024.0)
            if sz_mb > threshold:
                logger.info(
                    f"Skipping PBAD in auto mode: {p} size={sz_mb:.2f}MB exceeds threshold={threshold:.2f}MB"
                )
                return False
        except OSError:
            continue
    return True

def main():
    p = argparse.ArgumentParser(
        prog="liftcontracts",
        description="Run A->B and B->A liftover, always emit merged liftover, and compute reciprocal metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--contact-a", help="A: .contacts.rich.tsv")
    p.add_argument("--contact-b", help="B: .contacts.rich.tsv")
    p.add_argument("--matrix-a-prefix", help="Batch mode: prefix for A multi-resolution .contacts.tsv")
    p.add_argument("--matrix-b-prefix", help="Batch mode: prefix for B multi-resolution .contacts.tsv")
    p.add_argument("--fadix-a", required=True, help="FASTA index (.fai) for A")
    p.add_argument("--fadix-b", required=True, help="FASTA index (.fai) for B")
    p.add_argument("--mark-ab", required=True, help="A->B .mark")
    p.add_argument("--mark-ba", help="B->A .mark (optional; if not provided, invert mark-ab)")
    p.add_argument("--agg-frame", type=int, default=150000, help="Aggregation frame on B (bp)")
    p.add_argument("--dups-filter", choices=['length','coverage','deviation','none','default'], default='default')
    p.add_argument("--model", choices=['balanced','raw'], default='raw')
    p.add_argument("--uncert-thr", type=float, default=0.5, help="Remapping coverage threshold for uncertainty tag")
    p.add_argument("--frame", type=int, default=8, help="Window size (bins) for PBAD summary")
    p.add_argument(
        "--pbad-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="PBAD computation mode: auto skips PBAD for large liftContacts files",
    )
    p.add_argument(
        "--pbad-auto-threshold-mb",
        type=float,
        default=1024.0,
        help="In --pbad-mode auto, skip PBAD when any liftContacts file exceeds this MB (<=0 disables auto-skip)",
    )
    p.add_argument("--interactive", default="auto", choices=["auto","on","off"],
        help="Write interactive Plotly HTML for summary/tag tables (default: auto)")
    p.add_argument("--no-tags", action="store_true",
        help="Disable uncertainty tag file output (for compatibility with liftcontacts)")
    # Deprecated compatibility flag: merged output is always generated.
    p.add_argument("--write-merged", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--nthreads", type=int, default=1, help="Threads for contact-diff computation (run_liftover)")
    p.add_argument("--tmp-dir", help="Directory for temporary spill files")
    p.add_argument(
        "--spill-threshold-mb",
        type=float,
        default=0.0,
        help="Enable spill mode when contact/liftover file size >= this MB (0 disables)",
    )
    p.add_argument("--hash-shards", type=int, default=16, help="Shard count used by spill mode in liftcontacts")
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    args = p.parse_args()

    def resolve_contacts(prefix: str):
        import glob, re
        files = glob.glob(f"{prefix}.r*.contacts.tsv")
        if not files:
            single = f"{prefix}.contacts.tsv"
            if os.path.exists(single):
                return {None: single}
            return {}
        out = {}
        for fpath in files:
            m = re.search(r"\\.r(\\d+)\\.contacts\\.tsv$", fpath)
            if m:
                out[int(m.group(1))] = fpath
        return out

    if args.matrix_a_prefix or args.matrix_b_prefix:
        if args.contact_a or args.contact_b:
            raise SystemExit("Use --matrix-a-prefix/--matrix-b-prefix or --contact-a/--contact-b, not both.")
        if not args.matrix_a_prefix or not args.matrix_b_prefix:
            raise SystemExit("Both --matrix-a-prefix and --matrix-b-prefix are required.")
        set_a = resolve_contacts(args.matrix_a_prefix.strip())
        set_b = resolve_contacts(args.matrix_b_prefix.strip())
        if not set_a or not set_b:
            raise SystemExit("No contacts found for matrix prefix.")
        keys = sorted(set(set_a.keys()) & set(set_b.keys()))
        if not keys:
            raise SystemExit("No matching resolutions between prefixes.")
        for k in keys:
            suffix = f".r{k}" if k is not None else ""
            _run_bidirectional(set_a[k], set_b[k], args, suffix)
        return

    if not args.contact_a or not args.contact_b:
        raise SystemExit("Provide --contact-a/--contact-b or use --matrix-a-prefix/--matrix-b-prefix.")

    _run_bidirectional(args.contact_a, args.contact_b, args, "")

def _run_bidirectional(contact_a, contact_b, args, suffix):
    """Run bidirectional liftcontacts with improved logging and performance."""
    logger.info(f"Running bidirectional liftcontacts with {args.nthreads} threads")
    mark_ab = args.mark_ab if args.mark_ab.endswith(".mark") else (args.mark_ab + ".mark")
    if not os.path.exists(mark_ab):
        raise SystemExit(f".mark not found: {mark_ab}")
    cleanup_mark_ba = False
    if args.mark_ba:
        mark_ba = args.mark_ba if args.mark_ba.endswith(".mark") else (args.mark_ba + ".mark")
    else:
        fd, tmp = tempfile.mkstemp(suffix=".mark", prefix="inv.")
        os.close(fd)
        _invert_mark(mark_ab, tmp)
        mark_ba = tmp
        cleanup_mark_ba = True

    try:
        logger.info("Running A->B liftover...")
        run_liftover(
            contact_a,
            contact_b,
            args.fadix_a,
            args.fadix_b,
            mark_ab,
            args.agg_frame,
            args.dups_filter,
            args.model,
            args.out_prefix + suffix + ".AtoB",
            args.nthreads,
            tmp_dir=args.tmp_dir,
            spill_threshold_mb=args.spill_threshold_mb,
            hash_shards=args.hash_shards,
        )
        
        logger.info("Running B->A liftover...")
        run_liftover(
            contact_b,
            contact_a,
            args.fadix_b,
            args.fadix_a,
            mark_ba,
            args.agg_frame,
            args.dups_filter,
            args.model,
            args.out_prefix + suffix + ".BtoA",
            args.nthreads,
            tmp_dir=args.tmp_dir,
            spill_threshold_mb=args.spill_threshold_mb,
            hash_shards=args.hash_shards,
        )

        lift_ab = args.out_prefix + suffix + ".AtoB.liftContacts"
        lift_ba = args.out_prefix + suffix + ".BtoA.liftContacts"
        merged_path = args.out_prefix + ".Merged" + suffix + ".liftContacts"
        _write_merged_liftover(
            lift_ab,
            lift_ba,
            args.fadix_a,
            args.fadix_b,
            merged_path,
            tmp_dir=args.tmp_dir,
            spill_threshold_mb=args.spill_threshold_mb,
        )
        res_ab = infer_resolution_from_liftover(lift_ab)
        res_ba = infer_resolution_from_liftover(lift_ba)
        map_ab = _read_lift_as_map(lift_ab, res_ab)
        map_ba = _read_lift_as_map(lift_ba, res_ba)

        reciprocal, symmetry, inter, n_ab, n_ba = _reciprocal_stats(map_ab, map_ba)
        asym_penalty = 1.0 - reciprocal
        if _should_compute_pbad(lift_ab, lift_ba, args.pbad_mode, args.pbad_auto_threshold_mb):
            mean_pbad_ab = _mean_pbad_from_liftover(lift_ab, args.fadix_a, args.frame)
            mean_pbad_ba = _mean_pbad_from_liftover(lift_ba, args.fadix_b, args.frame)
        else:
            mean_pbad_ab = 0.0
            mean_pbad_ba = 0.0
        mean_pbad = 0.5 * (mean_pbad_ab + mean_pbad_ba)
        reciprocal_pbad = mean_pbad * reciprocal
        denom = abs(mean_pbad_ab) + abs(mean_pbad_ba) + 1e-9
        bidir_conserve = (1.0 - abs(mean_pbad_ab - mean_pbad_ba) / denom) * reciprocal

        out_sum = args.out_prefix + suffix + ".bidirectional.summary.tsv"
        with open(out_sum, "w", buffering=1024 * 1024) as f:
            f.write("metric\tvalue\n")
            f.write(f"reciprocal_consistency\t{reciprocal:.6f}\n")
            f.write(f"consensus_score\t{symmetry:.6f}\n")
            f.write(f"asymmetric_penalty\t{asym_penalty:.6f}\n")
            f.write(f"liftover_symmetry_index\t{symmetry:.6f}\n")
            f.write(f"reciprocal_pbad\t{reciprocal_pbad:.6f}\n")
            f.write(f"bidirectional_conservation_score\t{bidir_conserve:.6f}\n")
            f.write(f"n_AtoB\t{n_ab}\n")
            f.write(f"n_BtoA\t{n_ba}\n")
            f.write(f"n_reciprocal\t{inter}\n")
        logger.info(f"[OK] {out_sum}")

        tag_path = None
        if not args.no_tags:
            tag_path = args.out_prefix + suffix + ".bidirectional.tags.tsv"
            with open(lift_ab, 'r') as f, open(tag_path, "w", buffering=1024 * 1024) as fo:
                _ = f.readline()
                fo.write("chr1\tpos1\tchr2\tpos2\ttag\n")
                for line in f:
                    a = line.split()
                    if len(a) < 16:
                        continue
                    c1, p1 = a[0], int(a[1])
                    c2, p2 = a[2], int(a[3])
                    rm1 = _parse_remap_token(a[4])
                    rm2 = _parse_remap_token(a[5])
                    if not rm1 or not rm2:
                        continue
                    b1 = (c1, p1 // res_ab)
                    b2 = (c2, p2 // res_ab)
                    key = (b1, b2) if b1 <= b2 else (b2, b1)
                    v = map_ab.get(key)
                    recip = v in map_ba if v is not None else False
                    try:
                        cov = float(a[-1])
                    except Exception:
                        cov = 0.0
                    if cov < args.uncert_thr:
                        tag = "uncertain"
                    elif recip:
                        tag = "reciprocal"
                    else:
                        tag = "oneway"
                    fo.write(f"{c1}\t{p1}\t{c2}\t{p2}\t{tag}\n")
            logger.info(f"[OK] {tag_path}")

        try:
            enable_plotly = _resolve_interactive(args.interactive)
        except Exception as e:
            raise SystemExit(str(e))
        if enable_plotly:
            try:
                go, _, _ = _import_plotly()
                if go is not None:
                    metrics = ["reciprocal_consistency","consensus_score","asymmetric_penalty",
                               "liftover_symmetry_index","reciprocal_pbad","bidirectional_conservation_score",
                               "n_AtoB","n_BtoA","n_reciprocal"]
                    values = [reciprocal, symmetry, asym_penalty, symmetry, reciprocal_pbad, bidir_conserve, n_ab, n_ba, inter]
                    fig = go.Figure(data=[go.Bar(x=metrics, y=values, text=[f"{v:.4g}" for v in values], textposition="auto")])
                    fig.update_layout(xaxis_title="metric", yaxis_title="value")
                    _write_plotly_html(fig, args.out_prefix + suffix + ".bidirectional.summary.plotly.html",
                                       title="Bidirectional summary metrics")

                    if tag_path:
                        tag_counts = {"reciprocal": 0, "oneway": 0, "uncertain": 0}
                        with open(tag_path, 'r') as f:
                            _ = f.readline()
                            for line in f:
                                toks = line.split()
                                if len(toks) >= 5:
                                    tag_counts[toks[4]] = tag_counts.get(toks[4], 0) + 1
                        fig2 = go.Figure(data=[go.Bar(x=list(tag_counts.keys()), y=list(tag_counts.values()))])
                        fig2.update_layout(xaxis_title="tag", yaxis_title="count")
                        _write_plotly_html(fig2, args.out_prefix + suffix + ".bidirectional.tags.plotly.html",
                                           title="Bidirectional tag counts")
            except Exception as e:
                logger.warning(f"Plotly output failed: {e}")
    finally:
        if cleanup_mark_ba:
            try:
                os.remove(mark_ba)
            except OSError:
                pass

if __name__ == "__main__":
    main()
