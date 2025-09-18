#!/usr/bin/env python3
import argparse

def ChromIndexingFAI(path):
    ChrInd = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            name = line.split()[0]
            ChrInd[name] = i + 1
            ChrInd[i + 1] = name
    return ChrInd

def lftReadingMarkPoints(mark_path, ChrIdxs1, ChrIdxs2):
    ObjCoorMP = {}
    with open(mark_path, 'r') as f:
        for line in f:
            a = line.split()
            if len(a) < 6:
                continue
            try:
                N1 = ChrIdxs1.get(a[0], ChrIdxs1.get(a[0][3:]))
                N2 = ChrIdxs2.get(a[3], ChrIdxs2.get(a[3][3:]))
                c1 = (int(a[1]) + int(a[2])) // 400
                c2 = (int(a[4]) + int(a[5])) // 400
            except Exception:
                continue
            if N1 is None or N2 is None:
                continue
            ObjCoorMP.setdefault((N1, c1), set()).add((N2, c2))
    return ObjCoorMP

def lftRough(ObjCoorMP, ChrIdxs1, ChrIdxs2, bed, out_bed):
    with open(bed, 'r') as f, open(out_bed, 'w') as out:
        for line in f:
            if not line.strip():
                continue
            a = line.split()
            try:
                N1 = ChrIdxs1.get(a[0], ChrIdxs1.get(a[0][3:]))
                c1 = int(a[1]) // 200
                c2 = int(a[2]) // 200
            except Exception:
                continue
            if N1 is None:
                continue
            for j in range(c1, c2 + 1):
                lifted = ObjCoorMP.get((N1, j))
                if not lifted:
                    continue
                for (n2, b2) in lifted:
                    out.write(f"{ChrIdxs2[n2]}\t{b2*200}\t{b2*200+199}\t{a[3] if len(a)>3 else '.'}\n")

def main():
    p = argparse.ArgumentParser(
        description="Roughly liftover a BED track from genome A to genome B using a .mark synteny map (QA/visual checks)."
    )
    p.add_argument("--track", required=True, help="Input BED on genome A")
    p.add_argument("--fadix-a", required=True, help="FASTA index (.fai) for genome A")
    p.add_argument("--fadix-b", required=True, help="FASTA index (.fai) for genome B")
    p.add_argument("--remap-mark", required=True, help="A→B .mark file describing syntenic segments")
    p.add_argument("--out-bed", required=True, help="Output BED on genome B")
    a = p.parse_args()
    idxA = ChromIndexingFAI(a.fadix_a)
    idxB = ChromIndexingFAI(a.fadix_b)
    MP = lftReadingMarkPoints(a.remap_mark, idxA, idxB)
    lftRough(MP, idxA, idxB, a.track, a.out_bed)

if __name__ == '__main__':
    main()
