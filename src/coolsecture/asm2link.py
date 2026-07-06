#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from pathlib import Path
import shutil

def ensure_tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        sys.exit(f"[ERROR] {name} not found; install it and ensure it is on PATH.")
    return exe

def run_minimap2(minimap2_exe: str, ref: str, qry: str, preset: str,
                 paf_path: Path, threads: int):
    cmd = [
        minimap2_exe,
        "-x", preset,
        "--secondary=no",
        "-t", str(threads),
        ref, qry
    ]
    paf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(paf_path, "w") as fo:
        proc = subprocess.run(cmd, stdout=fo, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(f"[ERROR] minimap2 failed: {' '.join(cmd)}")
    else:
        sys.stderr.write(proc.stderr)

def paf_to_link(paf_path: Path, link_path: Path, min_match: int = 0):
    n_in = n_out = 0
    with open(paf_path, "r") as fi, open(link_path, "w") as fo:
        for line in fi:
            if not line.strip():
                continue
            a = line.rstrip("\n").split("\t")
            if len(a) < 12:
                continue
            n_in += 1
            try:
                qname  = a[0]
                qstart = int(a[2])
                qend   = int(a[3])
                strand = a[4]
                tname  = a[5]
                tstart = int(a[7])
                tend   = int(a[8])
                nmatch = int(a[9])
            except Exception:
                continue
            if nmatch < min_match:
                continue
            if strand == "-":
                fo.write(f"{qname}\t{qstart}\t{qend}\t{tname}\t{tend}\t{tstart}\n")
            else:
                fo.write(f"{qname}\t{qstart}\t{qend}\t{tname}\t{tstart}\t{tend}\n")
            n_out += 1
    print(f"[OK] PAF records: {n_in}, link: {n_out} -> {link_path}")

MUMMER_BIN_DIR = "/home/pkzhu/software/mummer-4.0.1"

def _mummer_tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        candidate = os.path.join(MUMMER_BIN_DIR, name)
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            exe = candidate
    if not exe:
        sys.exit(f"[ERROR] {name} not found; install mummer4 or add {MUMMER_BIN_DIR} to PATH.")
    return exe

def run_mummer4(ref: str, qry: str, out_prefix: Path,
                filt: str, min_idy: float, min_len: int) -> Path:
    nucmer = _mummer_tool("nucmer")
    delta_filter = _mummer_tool("delta-filter")
    show_coords = _mummer_tool("show-coords")

    delta_path = out_prefix.with_suffix(".delta")
    delta_path.parent.mkdir(parents=True, exist_ok=True)

    nucmer_cmd = [nucmer, "-p", str(out_prefix), ref, qry]
    proc = subprocess.run(nucmer_cmd, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(f"[ERROR] nucmer failed: {' '.join(nucmer_cmd)}")
    if not delta_path.exists():
        sys.exit(f"[ERROR] nucmer did not produce {delta_path}")

    filter_flags = []
    if filt == "1-to-1":
        filter_flags.append("-1")
    elif filt == "mutual-best":
        filter_flags.append("-m")
    # "none" → no filter, use raw .delta
    if filt != "none":
        filter_flags += ["-i", str(min_idy), "-l", str(min_len)]

    if filt == "none":
        filtered_delta = delta_path
    else:
        filtered_delta = out_prefix.with_suffix(".filter.delta")
        with open(filtered_delta, "w") as fo:
            cmd = [delta_filter] + filter_flags + [str(delta_path)]
            proc = subprocess.run(cmd, stdout=fo, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            sys.exit(f"[ERROR] delta-filter failed: {' '.join(cmd)}")

    coords_path = out_prefix.with_suffix(".coords.tsv")
    with open(coords_path, "w") as fo:
        cmd = [show_coords, "-T", "-H", str(filtered_delta)]
        proc = subprocess.run(cmd, stdout=fo, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(f"[ERROR] show-coords failed: {' '.join(cmd)}")

    return coords_path

def mummer_coords_to_link(coords_path: Path, link_path: Path):
    """show-coords -T -H produces 9 tab-delimited columns (1-based closed coords):
    [S1][E1][S2][E2][LEN 1][LEN 2][% IDY][TAG ref][TAG query]
    nucmer was run with ref=B qry=A, so TAG ref→B (target), TAG query→A.
    Negative-strand alignments have S2 > E2.
    mummer coords are 1-based closed; link file is 0-based half-open (PAF convention),
    with negative-strand records emitted as tend < tstart (matching minimap2 path).
    """
    n_in = n_out = 0
    with open(coords_path, "r") as fi, open(link_path, "w") as fo:
        for line in fi:
            if not line.strip():
                continue
            a = line.rstrip("\n").split("\t")
            if len(a) < 9:
                continue
            n_in += 1
            try:
                s1 = int(a[0]); e1 = int(a[1])
                s2 = int(a[2]); e2 = int(a[3])
                tag1 = a[7].strip()
                tag2 = a[8].strip()
            except Exception:
                continue
            # tag1 is the ref (B), tag2 is the query (A)
            qname_a = tag2
            tname_b = tag1
            qstart = min(s2, e2) - 1  # 0-based
            qend   = max(s2, e2)       # half-open
            if s2 > e2:
                # negative strand: target coords swapped so tend < tstart
                tstart = e1           # 1-based closed → keep as-is (will be > tend)
                tend   = s1 - 1       # 0-based
                fo.write(f"{qname_a}\t{qstart}\t{qend}\t{tname_b}\t{tstart}\t{tend}\n")
            else:
                tstart = s1 - 1       # 0-based
                tend   = e1           # half-open
                fo.write(f"{qname_a}\t{qstart}\t{qend}\t{tname_b}\t{tstart}\t{tend}\n")
            n_out += 1
    print(f"[OK] mummer coords records: {n_in}, link: {n_out} -> {link_path}")

def main():
    ap = argparse.ArgumentParser(
        prog="asm2link",
        description="Align two assemblies and produce a link file (6-column). "
                    "Supports minimap2 (PAF) and mummer4 (nucmer+delta-filter+show-coords).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-ga", "--genome-a", required=True, help="Query assembly FASTA")
    ap.add_argument("-gb", "--genome-b", required=True, help="Target assembly FASTA")
    ap.add_argument("-p",  "--out-prefix", required=True, help="Output prefix")
    ap.add_argument("-a",  "--aligner", choices=["minimap2","mummer4"], default="minimap2",
                    help="Aligner to use")
    ap.add_argument("-x",  choices=["asm5","asm10","asm20"], default="asm10",
                    help="minimap2 preset (only used when aligner=minimap2)")
    ap.add_argument("--min-match", type=int, default=0,
                    help="Minimum matched bases to keep a PAF record (column 10, minimap2 only)")
    ap.add_argument("--mummer-filter", choices=["1-to-1","mutual-best","none"], default="1-to-1",
                    help="delta-filter mode (mummer4 only). 1-to-1: each query position maps to "
                         "exactly one ref position and vice versa. mutual-best: -m. none: skip filter.")
    ap.add_argument("--mummer-min-idy", type=float, default=0,
                    help="Minimum alignment identity %% for delta-filter (mummer4 only)")
    ap.add_argument("--mummer-min-len", type=int, default=0,
                    help="Minimum alignment length for delta-filter (mummer4 only)")
    args = ap.parse_args()

    A = Path(args.genome_a)
    B = Path(args.genome_b)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    link_path = out_prefix.with_suffix(".link")

    if args.aligner == "minimap2":
        mm2 = ensure_tool("minimap2")
        paf_path = out_prefix.with_suffix(".paf")
        threads = os.cpu_count() or 24
        run_minimap2(mm2, ref=str(B), qry=str(A), preset=args.x,
                     paf_path=paf_path, threads=threads)
        paf_to_link(paf_path, link_path, min_match=args.min_match)
        print(f"[OK] Wrote PAF: {paf_path}")
        print(f"[OK] Wrote link: {link_path}")
    else:
        coords_path = run_mummer4(ref=str(B), qry=str(A), out_prefix=out_prefix,
                                  filt=args.mummer_filter,
                                  min_idy=args.mummer_min_idy,
                                  min_len=args.mummer_min_len)
        mummer_coords_to_link(coords_path, link_path)
        print(f"[OK] Wrote coords: {coords_path}")
        print(f"[OK] Wrote link: {link_path}")

if __name__ == "__main__":
    main()

