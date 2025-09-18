import argparse, sys
import argparse, sys
from textwrap import dedent

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
        "liftcontacts": "liftcontacts",
        "contact-stat": "contact_stat",
        "metric":       "metric",
        "lift2cool":    "lift2cool",
        "plot-cross":   "plot_cross",
    }

    cmd_desc = {
        "asm2link":     "Run minimap2 and convert PAF to 6-column link.",
        "link2mark":    "Convert syntenic links into mark file for liftover.",
        "prepare":      "preprocessing from cool/mcool into rich contacts.",
        "liftcontacts": "Liftover contacts from species A to B",
        "roughlift":    "Fast/rough liftover for sanity checks (coarse mapping).",
        "contact-stat": "Compute per-chromosome stats and simple plots from liftContacts.",
        "metric":       "Compute P-BAD and other metrics; write bedGraph and summary plots.",
        "plot-cross":   "plot split-triangle heatmap (UPPER & LOWER modes).",
        "lift2cool":    "Convert liftContacts to .cool.",
    }

    parser = argparse.ArgumentParser(
        prog="coolsecture",
        description="Coolsecture v0.1.2",
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
