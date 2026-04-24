#!/usr/bin/env python3
import argparse
import os
import sys
import shlex
import subprocess
from pathlib import Path
import shutil
import re
import glob

def _run(cmd):
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

def _ensure_fai(fasta_path: str) -> str:
    fai_path = fasta_path + ".fai"
    if os.path.exists(fai_path):
        return fai_path
    samtools = shutil.which("samtools")
    if samtools:
        _run([samtools, "faidx", fasta_path])
        if os.path.exists(fai_path):
            return fai_path
    # minimal .fai generator
    with open(fasta_path, "rb") as f, open(fai_path, "w") as fo:
        name = None
        length = 0
        seq_offset = 0
        line_bases = 0
        line_width = 0
        pos = 0
        while True:
            line = f.readline()
            if not line:
                if name is not None:
                    fo.write(f"{name}\t{length}\t{seq_offset}\t{line_bases}\t{line_width}\n")
                break
            if line.startswith(b">"):
                if name is not None:
                    fo.write(f"{name}\t{length}\t{seq_offset}\t{line_bases}\t{line_width}\n")
                name = line[1:].split()[0].decode("utf-8")
                length = 0
                seq_offset = f.tell()
                line_bases = 0
                line_width = 0
            else:
                if line_bases == 0:
                    line_bases = len(line.rstrip(b"\r\n"))
                    line_width = len(line)
                length += len(line.rstrip(b"\r\n"))
            pos = f.tell()
    return fai_path

def _matrix_uri(path: str, res: int) -> str:
    if "::" in path:
        return path
    low = path.lower()
    if low.endswith(".mcool"):
        return f"{path}::resolutions/{res}"
    return path

def _extract_res_list(args_str: str):
    if not args_str:
        return []
    m = re.search(r"(?:^|\s)--resolution(?:[=\s]+)(\S+)", args_str)
    if not m:
        m = re.search(r"(?:^|\s)-res(?:[=\s]+)(\S+)", args_str)
    if not m:
        return []
    items = [x.strip() for x in m.group(1).split(",") if x.strip()]
    out = []
    for x in items:
        try:
            out.append(int(x))
        except ValueError:
            pass
    return out

def _has_multires_outputs(prefix: Path) -> bool:
    # prefix is like step1/A/A
    parent = prefix.parent
    pattern = f"{prefix.name}.r*.contacts.tsv"
    return any(parent.glob(pattern))

def _sanitize_extra_args(args_str: str, remove_flags):
    if not args_str:
        return []
    toks = shlex.split(args_str)
    out = []
    i = 0
    remove = set(remove_flags or [])
    while i < len(toks):
        t = toks[i]
        matched = False
        for flag in remove:
            if t == flag:
                matched = True
                i += 1
                if i < len(toks) and not toks[i].startswith("--"):
                    i += 1
                break
            if t.startswith(flag + "="):
                matched = True
                i += 1
                break
        if not matched:
            out.append(t)
            i += 1
    return out

def main():
    p = argparse.ArgumentParser(
        prog="run-all",
        description="Run Coolsecture end-to-end using FASTA and contact matrices (no plot-cross).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--genome-a", required=True, help="FASTA for species A")
    p.add_argument("--genome-b", required=True, help="FASTA for species B")
    p.add_argument("--matrix-a", required=True, help="Contact matrix for A (.cool/.mcool/.hic)")
    p.add_argument("--matrix-b", required=True, help="Contact matrix for B (.cool/.mcool/.hic)")
    p.add_argument("--resolution", type=int, required=True, help="Resolution (bp)")
    p.add_argument("--out-prefix", default="run_all", help="Output prefix directory")
    p.add_argument("--name-a", default="A", help="Label for species A")
    p.add_argument("--name-b", default="B", help="Label for species B")
    p.add_argument("--asm-preset", default="asm10", choices=["asm5","asm10","asm20"], help="minimap2 preset for asm2link")

    p.add_argument("--prepare-args", default="", help="Extra args for prepare")
    p.add_argument("--link2mark-args", default="", help="Extra args for link2mark")
    p.add_argument("--liftcontracts-args", default="", help="Extra args for liftcontracts liftover")
    p.add_argument("--contact-stat-args", default="", help="Extra args for contact-stat")
    p.add_argument("--metric-args", default="", help="Extra args for metric")
    p.add_argument("--lift2matrix-args", default="", help="Extra args for lift2matrix")
    args = p.parse_args()
    out = Path(args.out_prefix)
    step0 = out / "step0"
    step1 = out / "step1"
    step2 = out / "step2"
    step3 = out / "step3"
    step0.mkdir(parents=True, exist_ok=True)
    step1.mkdir(parents=True, exist_ok=True)
    step2.mkdir(parents=True, exist_ok=True)
    step3.mkdir(parents=True, exist_ok=True)

    fai_a = _ensure_fai(args.genome_a)
    fai_b = _ensure_fai(args.genome_b)
    res_list = _extract_res_list(args.prepare_args)
    # 1) asm2link
    link_prefix = step0 / f"{args.name_a}_{args.name_b}"
    _run([sys.executable, "-m", "coolsecture", "asm2link",
          "--genome-a", args.genome_a, "--genome-b", args.genome_b,
          "--out-prefix", str(link_prefix), "-x", args.asm_preset])

    # 2) link2mark
    _run([sys.executable, "-m", "coolsecture", "link2mark",
          "--link", str(link_prefix) + ".link",
          "--out-prefix", str(link_prefix)] + shlex.split(args.link2mark_args))

    # 3) prepare A/B
    use_prepare_res = bool(res_list)
    if use_prepare_res:
        for m in (args.matrix_a, args.matrix_b):
            low = m.lower()
            if "::" in m:
                raise SystemExit("When using --prepare-args with --resolution/-res, provide raw .mcool paths (no ::resolutions/RES).")
            if not low.endswith(".mcool"):
                raise SystemExit("When using --prepare-args with --resolution/-res, matrix inputs must be .mcool.")
        mat_a = args.matrix_a
        mat_b = args.matrix_b
    else:
        mat_a = _matrix_uri(args.matrix_a, args.resolution)
        mat_b = _matrix_uri(args.matrix_b, args.resolution)
    prep_a = step1 / args.name_a / args.name_a
    prep_b = step1 / args.name_b / args.name_b
    prep_a.parent.mkdir(parents=True, exist_ok=True)
    prep_b.parent.mkdir(parents=True, exist_ok=True)
    cmd_a = [sys.executable, "-m", "coolsecture", "prepare",
             "--matrix", mat_a,
             "--out-prefix", str(prep_a),
             "--nthreads", str(os.cpu_count() or 8)]
    if args.matrix_a.lower().endswith(".hic"):
        cmd_a += ["--resolution", str(args.resolution)]
    cmd_a += shlex.split(args.prepare_args)
    _run(cmd_a)

    cmd_b = [sys.executable, "-m", "coolsecture", "prepare",
             "--matrix", mat_b,
             "--out-prefix", str(prep_b),
             "--nthreads", str(os.cpu_count() or 8)]
    if args.matrix_b.lower().endswith(".hic"):
        cmd_b += ["--resolution", str(args.resolution)]
    cmd_b += shlex.split(args.prepare_args)
    _run(cmd_b)

    # 4) liftcontracts
    mcool_input = args.matrix_a.lower().endswith(".mcool") or args.matrix_b.lower().endswith(".mcool")
    multi_res = (bool(res_list) and mcool_input) or _has_multires_outputs(prep_a) or _has_multires_outputs(prep_b)
    if multi_res:
        print("[INFO] Detected multi-resolution mode. Switching liftcontracts/contact-stat/metric/lift2matrix to prefix-based processing.")
    lift_prefix = step2 / f"{args.name_a}_{args.name_b}" / f"{args.name_a}_{args.name_b}"
    lift_prefix.parent.mkdir(parents=True, exist_ok=True)
    downstream_prefix = str(lift_prefix) + ".Merged"
    downstream_liftover = downstream_prefix + ".liftContacts"
    if multi_res:
        extra_bidir = _sanitize_extra_args(
            args.liftcontracts_args,
            remove_flags=["--contact-a", "--contact-b", "--matrix-a-prefix", "--matrix-b-prefix", "--write-merged"],
        )
        _run([sys.executable, "-m", "coolsecture", "liftcontracts",
              "--matrix-a-prefix", str(prep_a),
              "--matrix-b-prefix", str(prep_b),
              "--fadix-a", fai_a, "--fadix-b", fai_b,
              "--mark-ab", str(link_prefix) + ".mark",
              "--out-prefix", str(lift_prefix)] + extra_bidir)
    else:
        extra_bidir = _sanitize_extra_args(
            args.liftcontracts_args,
            remove_flags=["--contact-a", "--contact-b", "--matrix-a-prefix", "--matrix-b-prefix", "--write-merged"],
        )
        _run([sys.executable, "-m", "coolsecture", "liftcontracts",
              "--contact-a", str(prep_a) + ".contacts.tsv",
              "--contact-b", str(prep_b) + ".contacts.tsv",
              "--fadix-a", fai_a, "--fadix-b", fai_b,
              "--mark-ab", str(link_prefix) + ".mark",
              "--out-prefix", str(lift_prefix)] + extra_bidir)

    # 5) contact-stat
    if multi_res:
        _run([sys.executable, "-m", "coolsecture", "contact-stat",
              "--liftover-prefix", downstream_prefix,
              "--fadix", fai_a,
              "--out-prefix", str(step3 / f"{args.name_a}_{args.name_b}.Merged")] + shlex.split(args.contact_stat_args))
    else:
        _run([sys.executable, "-m", "coolsecture", "contact-stat",
              "--liftover", downstream_liftover,
              "--fadix", fai_a,
              "--out-prefix", str(step3 / f"{args.name_a}_{args.name_b}")] + shlex.split(args.contact_stat_args))

    # 6) metric
    if multi_res:
        _run([sys.executable, "-m", "coolsecture", "metric",
              "--liftover-prefix", downstream_prefix,
              "--fadix", fai_a,
              "--out-prefix", str(step3 / f"{args.name_a}_{args.name_b}.Merged")] + shlex.split(args.metric_args))
    else:
        _run([sys.executable, "-m", "coolsecture", "metric",
              "--liftover", downstream_liftover,
              "--fadix", fai_a,
              "--out-prefix", str(step3 / f"{args.name_a}_{args.name_b}")] + shlex.split(args.metric_args))

    # 7) lift2matrix
    if multi_res:
        _run([sys.executable, "-m", "coolsecture", "lift2matrix",
              "--liftover-prefix", downstream_prefix,
              "--fadix", fai_a,
              "--out-prefix", str(step3 / f"{args.name_a}_{args.name_b}.Merged")] + shlex.split(args.lift2matrix_args))
    else:
        _run([sys.executable, "-m", "coolsecture", "lift2matrix",
              "--liftover", downstream_liftover,
              "--fadix", fai_a,
              "--out-prefix", str(step3 / f"{args.name_a}_{args.name_b}")] + shlex.split(args.lift2matrix_args))

if __name__ == "__main__":
    main()
