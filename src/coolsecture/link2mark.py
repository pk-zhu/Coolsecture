#!/usr/bin/env python3
import argparse

def _iter_link_rows(path):
    with open(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            a = ln.split()
            if len(a) < 6:
                continue
            ca, sa, ea, cb, sb, eb = a[:6]
            try:
                sa, ea, sb, eb = int(sa), int(ea), int(sb), int(eb)
            except ValueError:
                continue
            yield ca, sa, ea, cb, sb, eb

def _densify_segment(seg, thr_len=300, step_len=150):
    ca, sa, ea, cb, sb, eb, d, tag = seg
    la = ea - sa
    lb = abs(eb - sb)
    if la <= thr_len:
        yield seg
        return
    k = max(1, la // step_len)
    if d > 0:
        for i in range(int(k)):
            xs = sa + (la // k) * i
            xe = sa + (la // k) * (i + 1) if i < k - 1 else ea
            ys = sb + (lb // k) * i
            ye = sb + (lb // k) * (i + 1) if i < k - 1 else eb
            yield (ca, xs, xe, cb, ys, ye, d, tag)
    else:
        for i in range(int(k)):
            xs = sa + (la // k) * i
            xe = sa + (la // k) * (i + 1) if i < k - 1 else ea
            ys = sb - (lb // k) * i
            ye = sb - (lb // k) * (i + 1) if i < k - 1 else eb
            yield (ca, xs, xe, cb, ye, ys, d, tag)

def link2mark(in_link, out_mark, densify=True, thr_len=300, step_len=150, add_gaps=True):
    with open(out_mark, 'w') as out:
        blkid = 0
        prev = None
        for ca, sa, ea, cb, sb, eb in _iter_link_rows(in_link):
            d = 1 if eb >= sb else -1
            blkid += 1
            seg = (ca, sa, ea, cb, sb, eb, d, blkid)
            if add_gaps and prev and prev[0] == ca and prev[3] == cb:
                gapA = sa - prev[2]
                if gapA > 0:
                    if prev[6] > 0:
                        out.write(f"{ca}\t{prev[2]}\t{sa}\t{cb}\t{prev[5]}\t{sb}\t{prev[6]}\t{blkid}_gap\n")
                    else:
                        out.write(f"{ca}\t{prev[2]}\t{sa}\t{cb}\t{sb}\t{prev[5]}\t{prev[6]}\t{blkid}_gap\n")
            if densify:
                for s in _densify_segment(seg, thr_len=thr_len, step_len=step_len):
                    out.write(f"{s[0]}\t{s[1]}\t{s[2]}\t{s[3]}\t{s[4]}\t{s[5]}\t{s[6]}\t{s[7]}\n")
            else:
                out.write(f"{ca}\t{sa}\t{ea}\t{cb}\t{sb}\t{eb}\t{d}\t{blkid}\n")
            prev = (ca, sa, ea, cb, sb, eb, d)

def main():
    p = argparse.ArgumentParser(
        prog="link2mark",
        description="Convert 6-column collinearity links (.link) into a .mark synteny map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--link", required=True, help="Path to .link file")
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    p.add_argument("--no-densify", action="store_true", help="Disable splitting of long segments (> thr-len bp)")
    p.add_argument("--no-gap", action="store_true", help="Do not emit gap segments between adjacent links")
    p.add_argument("--thr-len", type=int, default=300, help="Length threshold (bp) used for densify")
    p.add_argument("--step-len", type=int, default=150, help="Step length (bp) when splitting long segments")
    a = p.parse_args()
    out_mark = f"{a.out_prefix}.mark"
    link2mark(a.link, out_mark, densify=not a.no_densify, thr_len=a.thr_len, step_len=a.step_len, add_gaps=not a.no_gap)

if __name__ == '__main__':
    main()
