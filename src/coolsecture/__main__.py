import argparse, sys

def _call(modname):
    mod = __import__(f"coolsecture.{modname}", fromlist=["main"])
    return mod.main()

def _format_epilog(cmd_desc: dict) -> str:
    lines = ["Commands:"]
    pad = max(len(k) for k in cmd_desc) + 2
    for k in sorted(cmd_desc):
        lines.append(f"  {k.ljust(pad)}{cmd_desc[k]}")
    return "\n".join(lines)

def main():
    cmds = {
        "asm2link":     "asm2link",
        "link2mark":    "link2mark",
        "prepare":      "prepare",
        "roughlift":    "roughlift",
        "liftcontacts": "bidirectional",
        "liftcontracts": "bidirectional",
        "contact-stat": "contact_stat",
        "metric":       "metric",
        "lift2matrix":  "lift2matrix",
        "run-all":      "run_all",
        "plot-cross":   "plot_cross",
        "multiscale":   "multiscale",
        "cross-validate": "cross_validate",
    }

    cmd_desc = {
        "asm2link":     "Align two assemblies with minimap2 or mummer4; write 6-column .link.",
        "link2mark":    "Convert syntenic links into mark file for liftover.",
        "prepare":      "Preprocessing from cool/mcool into rich contacts.",
        "roughlift":    "Fast/rough liftover for sanity checks (coarse mapping).",
        "liftcontacts": "Liftover contacts and compute reciprocal metrics.",
        "liftcontracts": "[Deprecated alias for liftcontacts]",
        "contact-stat": "Compute per-chromosome stats and simple plots from liftContacts.",
        "metric":       "Compute P-BAD and other metrics; write bedGraph and summary plots.",
        "plot-cross":   "Plot split-triangle heatmap (UPPER & LOWER modes).",
        "lift2matrix":  "Convert liftContacts to matrices (.cool or .hic).",
        "run-all":      "End-to-end pipeline (no plot-cross).",
        "multiscale":   "Multi-scale PBAD summary from multi-resolution liftover.",
        "cross-validate": "Stratum-adjusted correlation (HiCRep-style) between matched source-coordinate matrices.",
    }

    parser = argparse.ArgumentParser(
        prog="coolsecture",
        description="Coolsecture v0.3.5",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=_format_epilog(cmd_desc),
    )

    parser.add_argument("cmd", nargs="?", choices=list(cmds.keys()))
    parser.add_argument("args", nargs=argparse.REMAINDER)

    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return 0

    ns = parser.parse_args(sys.argv[1:2])

    sys.argv = [f"coolsecture {ns.cmd}"] + sys.argv[2:]
    return _call(cmds[ns.cmd])

if __name__ == "__main__":
    raise SystemExit(main())
