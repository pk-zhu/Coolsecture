#!/usr/bin/env python3
import argparse

def _iter_chain_blocks(path):
    with open(path) as f:
        line = None
        for line in f:
            if not line.strip():
                continue
            if line.startswith("chain"):
                parts = line.split()
                if len(parts) < 13:
                    continue
                _, score, tName, tSize, tStrand, tStart, tEnd, qName, qSize, qStrand, qStart, qEnd, chain_id = parts[:13]
                tSize = int(tSize); tStart = int(tStart); tEnd = int(tEnd)
                qSize = int(qSize); qStart = int(qStart); qEnd = int(qEnd)
                tPos = tStart
                if qStrand == "+":
                    qPos = qStart
                else:
                    qPos = qSize - qEnd
                while True:
                    block = next(f, None)
                    if block is None:
                        break
                    block = block.strip()
                    if not block:
                        break
                    nums = block.split()
                    size = int(nums[0])
                    dt = int(nums[1]) if len(nums) > 1 else 0
                    dq = int(nums[2]) if len(nums) > 2 else 0
                    t_s = tPos
                    t_e = tPos + size
                    if qStrand == "+":
                        q_s = qPos
                        q_e = qPos + size
                    else:
                        q_s = qPos + size
                        q_e = qPos
                    yield tName, t_s, t_e, qName, q_s, q_e
                    tPos += size + dt
                    if qStrand == "+":
                        qPos += size + dq
                    else:
                        qPos += size + dq
                continue

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

def link2mark(rows_iter, out_mark, densify=True, thr_len=300, step_len=150, add_gaps=True):
    with open(out_mark, 'w') as out:
        blkid = 0
        prev = None
        for ca, sa, ea, cb, sb, eb in rows_iter:
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
        description="Convert 6-column collinearity links (.link) or UCSC chain (.chain) into a .mark synteny map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--link", help="Path to .link file")
    p.add_argument("--chain", help="Path to UCSC .chain file")
    p.add_argument("-p", "--out-prefix", required=True, help="Output prefix")
    p.add_argument("--no-densify", action="store_true", help="Disable splitting of long segments (> thr-len bp)")
    p.add_argument("--no-gap", action="store_true", help="Do not emit gap segments between adjacent links")
    p.add_argument("--thr-len", type=int, default=300, help="Length threshold (bp) used for densify")
    p.add_argument("--step-len", type=int, default=150, help="Step length (bp) when splitting long segments")
    a = p.parse_args()
    out_mark = f"{a.out_prefix}.mark"
    if (not a.link) and (not a.chain):
        raise SystemExit("Provide --link or --chain.")
    if a.link and a.chain:
        raise SystemExit("Provide only one of --link or --chain.")
    if a.chain:
        link2mark(_iter_chain_blocks(a.chain), out_mark, densify=not a.no_densify, thr_len=a.thr_len, step_len=a.step_len, add_gaps=not a.no_gap)
    else:
        link2mark(_iter_link_rows(a.link), out_mark, densify=not a.no_densify, thr_len=a.thr_len, step_len=a.step_len, add_gaps=not a.no_gap)

if __name__ == '__main__':
    main()

