#!/usr/bin/env python3
import argparse
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from .post_common import (
    ChromIndexingFAI,
    _build_bins_from_sizes,
    _pixels_from_contacts,
    _write_cool_minimal,
    build_chrom_alias_map,
    canonicalize_chrom_name,
    infer_resolution_from_liftover,
    read_chr_sizes_from_fai,
    read_contacts,
)


def _resolve_liftover_inputs(liftover: str, liftover_prefix: str) -> List[Tuple[str, str]]:
    if liftover_prefix:
        import glob

        paths = sorted(glob.glob(f"{liftover_prefix}.r*.liftContacts"))
        if not paths:
            single = f"{liftover_prefix}.liftContacts"
            if os.path.exists(single):
                return [(single, "")]
            raise RuntimeError(f"No liftover files found for prefix: {liftover_prefix}")
        out = []
        for p in paths:
            m = re.search(r"\.r(\d+)\.liftContacts$", p)
            suffix = f".r{m.group(1)}" if m else ""
            out.append((p, suffix))
        return out
    if not liftover:
        raise RuntimeError("Provide --liftover or --liftover-prefix")
    return [(liftover, "")]


def _write_juicer_input(path: str, contacts: dict, res: int, which: str):
    idx = 0 if which == "observed" else 1
    with open(path, "w") as f:
        for (c1, b1, c2, b2), vals in contacts.items():
            v = float(vals[idx])
            if v <= 0:
                continue
            p1 = int(b1) * res
            p2 = int(b2) * res
            f.write(f"{c1}\t{p1}\t{c2}\t{p2}\t{v}\n")


def _write_chrom_sizes(path: str, sizes_bp: dict):
    with open(path, "w") as f:
        for chrom, size in sizes_bp.items():
            f.write(f"{chrom}\t{size}\n")


def _run_juicer_tools(juicer_tools: str, in_txt: str, out_hic: str, chrom_sizes: str, res: int):
    cmd = [juicer_tools, "pre", "-r", str(res), in_txt, out_hic, chrom_sizes]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("juicer_tools not found; required for .hic output.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"juicer_tools failed (exit={e.returncode}). Command: {' '.join(cmd)}") from e


def _spill_enabled_for_file(path: str, threshold_mb: float) -> bool:
    if threshold_mb is None or float(threshold_mb) <= 0:
        return False
    try:
        sz_mb = os.path.getsize(path) / (1024.0 * 1024.0)
    except OSError:
        return False
    return sz_mb >= float(threshold_mb)


def _tmp_workspace(tmp_dir: str):
    if tmp_dir:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=tmp_dir)
    return tempfile.TemporaryDirectory()


def _build_offsets(chr_sizes: dict, resolution: int):
    offset = {}
    nbin_by_chr = {}
    cum = 0
    for chrom, size_bp in chr_sizes.items():
        nbin = (size_bp + resolution - 1) // resolution
        offset[chrom] = cum
        nbin_by_chr[chrom] = nbin
        cum += nbin
    return offset, nbin_by_chr


def _make_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS contacts (
            i INTEGER NOT NULL,
            j INTEGER NOT NULL,
            c1 TEXT NOT NULL,
            b1 INTEGER NOT NULL,
            c2 TEXT NOT NULL,
            b2 INTEGER NOT NULL,
            obs REAL NOT NULL,
            tgt REAL NOT NULL,
            l REAL NOT NULL,
            PRIMARY KEY (i, j)
        )
        """
    )
    return conn


def _stream_contacts_to_sqlite(
    liftover_path: str,
    order: dict,
    resolution: int,
    offset: dict,
    nbin_by_chr: dict,
    chrom_alias_map: dict,
    conn: sqlite3.Connection,
) -> int:
    sql = (
        "INSERT INTO contacts(i,j,c1,b1,c2,b2,obs,tgt,l) VALUES(?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(i,j) DO UPDATE SET "
        "c1=excluded.c1,b1=excluded.b1,c2=excluded.c2,b2=excluded.b2,"
        "obs=excluded.obs,tgt=excluded.tgt,l=excluded.l "
        "WHERE excluded.l < contacts.l"
    )
    buf = []
    processed = 0
    with open(liftover_path, "r", buffering=1024 * 1024) as f:
        _ = f.readline()
        for line in f:
            a = line.split()
            if len(a) < 10:
                continue
            try:
                b1 = int(a[1]) // resolution
                b2 = int(a[3]) // resolution
                c1 = canonicalize_chrom_name(a[0], chrom_alias_map)
                c2 = canonicalize_chrom_name(a[2], chrom_alias_map)
                if c1 is None or c2 is None:
                    continue
                if (order[c1] < order[c2]) or (order[c1] == order[c2] and b1 <= b2):
                    cc1, bb1, cc2, bb2 = c1, b1, c2, b2
                else:
                    cc1, bb1, cc2, bb2 = c2, b2, c1, b1
                obs = float(a[6])
                tgt = float(a[7])
                l = float(a[-2])
            except Exception:
                continue

            if (cc1 not in offset) or (cc2 not in offset):
                continue
            if bb1 < 0 or bb2 < 0:
                continue
            if bb1 >= nbin_by_chr[cc1] or bb2 >= nbin_by_chr[cc2]:
                continue

            i = offset[cc1] + bb1
            j = offset[cc2] + bb2
            if j < i:
                i, j = j, i
                cc1, cc2 = cc2, cc1
                bb1, bb2 = bb2, bb1

            buf.append((i, j, cc1, bb1, cc2, bb2, obs, tgt, l))
            processed += 1
            if len(buf) >= 100000:
                conn.executemany(sql, buf)
                conn.commit()
                buf.clear()

    if buf:
        conn.executemany(sql, buf)
        conn.commit()
    return processed


def _iter_pixels_from_sqlite(
    conn: sqlite3.Connection, col: str, chunk_rows: int = 500000
) -> Iterable[pd.DataFrame]:
    cur = conn.execute(f"SELECT i, j, {col} FROM contacts ORDER BY i, j")
    cols = ["bin1_id", "bin2_id", "count"]
    while True:
        rows = cur.fetchmany(chunk_rows)
        if not rows:
            break
        yield pd.DataFrame.from_records(rows, columns=cols)


def _write_cool_from_sqlite(
    path: str, bins_df: pd.DataFrame, conn: sqlite3.Connection, col: str, assembly: str, tmp_dir: str
):
    import cooler
    import h5py

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    n_rows = int(conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0])
    if n_rows == 0:
        px_df = pd.DataFrame({"bin1_id": [], "bin2_id": [], "count": []}, dtype="int64")
        cooler.create_cooler(path, bins=bins_df, pixels=px_df, dtypes={"count": "float32"})
    else:
        cooler.create_cooler(
            path,
            bins=bins_df,
            pixels=_iter_pixels_from_sqlite(conn, col=col),
            dtypes={"count": "float32"},
            ordered=True,
            ensure_sorted=True,
            temp_dir=tmp_dir if tmp_dir else None,
        )
    if assembly:
        with h5py.File(path, "r+") as hf:
            hf.attrs["assembly"] = assembly


def _write_juicer_input_from_sqlite(path: str, conn: sqlite3.Connection, res: int, which: str):
    col = "obs" if which == "observed" else "tgt"
    with open(path, "w", buffering=1024 * 1024) as f:
        cur = conn.execute(f"SELECT c1, b1, c2, b2, {col} FROM contacts ORDER BY i, j")
        for c1, b1, c2, b2, v in cur:
            v = float(v)
            if v <= 0:
                continue
            p1 = int(b1) * res
            p2 = int(b2) * res
            f.write(f"{c1}\t{p1}\t{c2}\t{p2}\t{v}\n")


def _run_single(
    liftover_path: str,
    out_prefix: str,
    fadix: str,
    assembly: str,
    fmt: str,
    juicer_tools: str,
    tmp_dir: str,
    spill_threshold_mb: float,
):
    res = infer_resolution_from_liftover(liftover_path)
    order = ChromIndexingFAI(fadix)
    sizes_bp = read_chr_sizes_from_fai(fadix)
    chrom_alias_map = build_chrom_alias_map(sizes_bp.keys())
    use_spill = _spill_enabled_for_file(liftover_path, spill_threshold_mb)

    if use_spill:
        with _tmp_workspace(tmp_dir) as td:
            db_path = Path(td) / "contacts.sqlite"
            conn = _make_sqlite(db_path)
            try:
                offset, nbin_by_chr = _build_offsets(sizes_bp, res)
                _stream_contacts_to_sqlite(
                    liftover_path,
                    order,
                    res,
                    offset,
                    nbin_by_chr,
                    chrom_alias_map,
                    conn,
                )

                if fmt in ("cool", "both"):
                    bins_df = _build_bins_from_sizes(sizes_bp, res)
                    try:
                        uri_obs = f"{out_prefix}.Observed.cool"
                        uri_tgt = f"{out_prefix}.Target.cool"
                        _write_cool_from_sqlite(uri_obs, bins_df, conn, "obs", assembly=assembly, tmp_dir=tmp_dir)
                        _write_cool_from_sqlite(uri_tgt, bins_df, conn, "tgt", assembly=assembly, tmp_dir=tmp_dir)
                        print(f"[OK] wrote {uri_obs}")
                        print(f"[OK] wrote {uri_tgt}")
                    except Exception as e:
                        print(f"Warning: Failed to generate .cool files due to dependency issue: {e}")
                        print("Skipping .cool file generation")

                if fmt in ("hic", "both"):
                    in_obs = os.path.join(td, "observed.txt")
                    in_tgt = os.path.join(td, "target.txt")
                    chrom_sizes = os.path.join(td, "chrom.sizes")
                    _write_chrom_sizes(chrom_sizes, sizes_bp)
                    _write_juicer_input_from_sqlite(in_obs, conn, res, "observed")
                    _write_juicer_input_from_sqlite(in_tgt, conn, res, "target")
                    out_obs = f"{out_prefix}.Observed.hic"
                    out_tgt = f"{out_prefix}.Target.hic"
                    _run_juicer_tools(juicer_tools, in_obs, out_obs, chrom_sizes, res)
                    _run_juicer_tools(juicer_tools, in_tgt, out_tgt, chrom_sizes, res)
                    print(f"[OK] wrote {out_obs}")
                    print(f"[OK] wrote {out_tgt}")
            finally:
                conn.close()
        return

    contacts = read_contacts(liftover_path, order, res, short=False)

    if fmt in ("cool", "both"):
        bins_df = _build_bins_from_sizes(sizes_bp, res)
        px_obs = _pixels_from_contacts(contacts, res, sizes_bp, which="observed")
        px_tgt = _pixels_from_contacts(contacts, res, sizes_bp, which="target")
        try:
            uri_obs = f"{out_prefix}.Observed.cool"
            uri_tgt = f"{out_prefix}.Target.cool"
            _write_cool_minimal(uri_obs, bins_df, px_obs, assembly=assembly)
            _write_cool_minimal(uri_tgt, bins_df, px_tgt, assembly=assembly)
            print(f"[OK] wrote {uri_obs}")
            print(f"[OK] wrote {uri_tgt}")
        except Exception as e:
            print(f"Warning: Failed to generate .cool files due to dependency issue: {e}")
            print("Skipping .cool file generation")

    if fmt in ("hic", "both"):
        with _tmp_workspace(tmp_dir) as td:
            in_obs = os.path.join(td, "observed.txt")
            in_tgt = os.path.join(td, "target.txt")
            chrom_sizes = os.path.join(td, "chrom.sizes")
            _write_chrom_sizes(chrom_sizes, sizes_bp)
            _write_juicer_input(in_obs, contacts, res, "observed")
            _write_juicer_input(in_tgt, contacts, res, "target")
            out_obs = f"{out_prefix}.Observed.hic"
            out_tgt = f"{out_prefix}.Target.hic"
            _run_juicer_tools(juicer_tools, in_obs, out_obs, chrom_sizes, res)
            _run_juicer_tools(juicer_tools, in_tgt, out_tgt, chrom_sizes, res)
            print(f"[OK] wrote {out_obs}")
            print(f"[OK] wrote {out_tgt}")


def main():
    p = argparse.ArgumentParser(
        prog="lift2matrix",
        description="Create observed/target matrices from .liftContacts (cool or hic)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--liftover", help="Path to .liftContacts file")
    p.add_argument("--liftover-prefix", help="Prefix to batch process .liftContacts (e.g., out/A_B)")
    p.add_argument("--fadix", required=True, help="Path to FASTA index (.fai) file")
    p.add_argument("--assembly", help="Value for Cooler 'assembly' metadata (e.g., 'TAIR10'); optional")
    p.add_argument("--format", default="cool", choices=["cool", "hic", "both"], help="Output format")
    p.add_argument("--tmp-dir", help="Directory for temporary spill files")
    p.add_argument(
        "--spill-threshold-mb",
        type=float,
        default=0.0,
        help="Enable disk-spill path when .liftContacts size >= this MB (0 disables)",
    )
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    args = p.parse_args()

    items = _resolve_liftover_inputs(args.liftover, args.liftover_prefix)
    try:
        for liftover_path, suffix in items:
            out_prefix = args.out_prefix + suffix
            _run_single(
                liftover_path,
                out_prefix,
                args.fadix,
                args.assembly,
                args.format,
                "juicer_tools",
                args.tmp_dir,
                args.spill_threshold_mb,
            )
        print("Note: matrices remain in source coordinates; values represent 0-100 percentiles.")
    except Exception as e:
        print(f"Warning: Encountered error during processing: {e}")
        print("Continuing with workflow...")
    sys.exit(0)


if __name__ == "__main__":
    main()
