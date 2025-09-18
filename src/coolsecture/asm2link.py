#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from pathlib import Path
import shutil

def ensure_minimap2() -> str:
    exe = shutil.which("minimap2")
    if not exe:
        sys.exit("[ERROR] minimap2 not found; install it and ensure it is on PATH.")
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

def main():
    ap = argparse.ArgumentParser(
        prog="asm2link",
        description="Align two assemblies with minimap2 and produce a PAF file and a 6-column link file."
    )
    ap.add_argument("-ga", "--genome-a", required=True, help="Query assembly FASTA")
    ap.add_argument("-gb", "--genome-b", required=True, help="Target assembly FASTA")
    ap.add_argument("-p",  "--out-prefix", required=True, help="Output prefix")
    ap.add_argument("-x" , choices=["asm5","asm10","asm20"], required=True, help="minimap2 preset")
    ap.add_argument("--min-match", type=int, default=0, help="Minimum matched bases to keep a PAF record (column 10)")
    args = ap.parse_args()

    mm2 = ensure_minimap2()
    A = Path(args.genome_a)
    B = Path(args.genome_b)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    paf_path  = out_prefix.with_suffix(".paf")
    link_path = out_prefix.with_suffix(".link")

    threads = os.cpu_count() or 24

    run_minimap2(mm2, ref=str(B), qry=str(A), preset=args.preset,
                 paf_path=paf_path, threads=threads)

    paf_to_link(paf_path, link_path, min_match=args.min_match)

    print(f"[OK] Wrote PAF: {paf_path}")
    print(f"[OK] Wrote link: {link_path}")

if __name__ == "__main__":
    main()
