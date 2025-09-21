#!/usr/bin/env python3
import os, sys, argparse
from collections import OrderedDict
import numpy as np

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

def _is_rich(path):
    with open(path) as f:
        _ = f.readline()
        line = f.readline().split()
    if len(line) < 2:
        return False
    try:
        int(line[0]); int(line[1])
        return False
    except Exception:
        return True

def read_contacts_rich(contacts_path):
    out = {}
    bins = {}
    with open(contacts_path, 'r') as f:
        header = f.readline().strip().split()
        for line in f:
            if not line.strip():
                continue
            (chrom1, start1, end1, bin1,
             chrom2, start2, end2, bin2,
             rank, strict, weak, cov1, cov2, dist_bins) = line.split()[:14]
            start1 = int(start1); end1 = int(end1); bin1 = int(bin1)
            start2 = int(start2); end2 = int(end2); bin2 = int(bin2)
            bins[bin1] = (chrom1, start1, end1)
            bins[bin2] = (chrom2, start2, end2)
            key = ((chrom1, bin1), (chrom2, bin2)) if bin1 <= bin2 else ((chrom2, bin2), (chrom1, bin1))
            out[key] = (int(rank), int(strict), int(weak), float(cov1), float(cov2))
    return out, bins

def build_contact_hash_from_rich(rich_contacts, ChrIdxs):
    out = {}
    for ((chrom1, b1), (chrom2, b2)), (rank, strict, weak, cov1, cov2) in rich_contacts.items():
        N1 = ChrIdxs[chrom1]; N2 = ChrIdxs[chrom2]
        if (N2 < N1) or (N2 == N1 and b2 < b1):
            N1, N2 = N2, N1
            b1, b2 = b2, b1
            cov1, cov2 = cov2, cov1
        deviation = float(max(abs(rank - strict), abs(weak - rank)))
        out[(N1, b1, N2, b2)] = (int(rank), int(strict), int(weak), int(strict), int(weak), deviation, 1.0, cov1, cov2)
    return out

def label_from_keys(keyset, bins):
    ks = sorted(list(keyset))
    if not ks: return '-------'
    if len(ks) < 3:
        return ''.join([f"{bins[k[1]][0]}:{bins[k[1]][1]}-" for k in ks])
    first = ks[0]
    return f"{bins[first[1]][0]}:{bins[first[1]][1]}-"

def iDuplicateContact(Contact_disp_L, ObjCoorMP1, ObjCoorMP2):
    c = [0,0,0,0,0,0,0]
    base = ObjCoorMP1[0] * ObjCoorMP2[0] * ObjCoorMP1[1] * ObjCoorMP2[1]
    c[-1] = base
    c[0] = base * Contact_disp_L[0]
    c[1] = base * Contact_disp_L[1]
    c[2] = base * Contact_disp_L[2]
    c[3] = base * Contact_disp_L[3]
    c[4] = base * Contact_disp_L[4]
    c[-2] = base * Contact_disp_L[-1]
    return c

def choose_best(writed, dupled, crit):
    if crit == 'coverage':
        return (writed[-2] * writed[-3]) < (dupled[-2] * dupled[-3])
    elif crit == 'deviation':
        return writed[5] < dupled[5]
    elif crit == 'length':
        return (writed[-2] < dupled[-2]) or (dupled[-2] < 0)
    elif crit == 'none':
        return True
    else:
        return writed[-1] < dupled[-1]

def _infer_resolution_from_bins(bins_dict):
    widths = np.fromiter((v[2]-v[1] for v in bins_dict.values()), dtype=np.int64)
    if widths.size == 0:
        return 1
    vals, cnts = np.unique(widths, return_counts=True)
    return int(vals[int(np.argmax(cnts))])

def _build_chr_arrays(bins_dict, idx):
    chr_bins = {}
    chr_starts = {}
    chr_ends = {}
    for bid, (chrom, start, end) in bins_dict.items():
        N = idx[chrom]
        chr_bins.setdefault(N, []).append((start, end, bid))
    for N, arr in chr_bins.items():
        arr.sort(key=lambda x: x[0])
        starts = np.array([a[0] for a in arr], dtype=np.int64)
        ends = np.array([a[1] for a in arr], dtype=np.int64)
        bins = [a[2] for a in arr]
        chr_bins[N] = bins
        chr_starts[N] = starts
        chr_ends[N] = ends
    return chr_bins, chr_starts, chr_ends

def _locate_bin_index(center, starts_np, ends_np):
    pos = int(np.searchsorted(starts_np, center, side='right') - 1)
    if pos < 0 or pos >= starts_np.size:
        return -1
    if center >= int(ends_np[pos]):
        return -1
    return pos

def iReadingMarkPoints(mark_path, ChrIdxs1, ChrIdxs2, agg_bp, A_chr_bins, A_chr_starts, A_chr_ends, B_chr_bins, B_chr_starts, B_chr_ends, resB):
    frame = max(1, int(round(agg_bp / max(resB,1))))
    rev = {}
    with open(mark_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            a = line.split()
            N1 = ChrIdxs1.get(a[0])
            N2 = ChrIdxs2.get(a[3])
            if N1 is None or N2 is None:
                continue
            c1 = (int(a[1]) + int(a[2])) // 2
            c2 = (int(a[4]) + int(a[5])) // 2
            Astarts = A_chr_starts.get(N1)
            Aends = A_chr_ends.get(N1)
            Bstarts = B_chr_starts.get(N2)
            Bends = B_chr_ends.get(N2)
            if Astarts is None or Bstarts is None:
                continue
            idx1 = _locate_bin_index(c1, Astarts, Aends)
            idx2 = _locate_bin_index(c2, Bstarts, Bends)
            if idx1 < 0 or idx2 < 0:
                continue
            b1 = A_chr_bins[N1][idx1]
            b2 = B_chr_bins[N2][idx2]
            rev.setdefault((N2, b2), set()).add((N1, b1))
    fwd = {}
    for (N2, b2), src_set in rev.items():
        w = 1.0 / max(len(src_set), 1)
        for (N1, b1) in src_set:
            fwd.setdefault((N1, b1), []).append((N2, b2, 1.0, w))
    Obj = {}
    for src_key, triples in fwd.items():
        by_chr = {}
        for (N2, b2, cnt, w) in triples:
            by_chr.setdefault(N2, []).append((b2, cnt, w))
        groups = []
        for N2, arr in by_chr.items():
            arr.sort()
            cur = OrderedDict()
            last_b = None
            for (b2, cnt, w) in arr:
                if last_b is None or abs(b2 - last_b) <= frame:
                    cur[(N2, b2)] = [cnt, w]
                else:
                    groups.append(cur)
                    cur = OrderedDict({(N2, b2): [cnt, w]})
                last_b = b2
            if cur:
                groups.append(cur)
        normed = []
        for g in groups:
            tot = sum(v[0] for v in g.values())
            if tot <= 0:
                continue
            og = OrderedDict((k, (round(v[0] / tot, 2), round(v[1], 2))) for k, v in g.items())
            if og:
                normed.append(og)
        if normed:
            Obj[src_key] = normed
    return Obj

def iDifferContact(Contact_disp_0, Contact_disp_1, ObjCoorMP, model, criteria, ChrIdxs2, Bbins, stat_out_prefix):
    DifferContact = {}
    Dups = {}
    Statistic = (['all', 'remappable', 'processed', 'unique', 'duplicated', 'dropped'],
                 [0, 0, set(), set(), [], 0],
                 [set(), set(), set(), set(), set(), set()])
    for i, payload0 in Contact_disp_0.items():
        key1 = i[:2]; key2 = i[2:]
        Statistic[1][0] += 1
        Statistic[2][0] |= {key1, key2}
        if (key1 in ObjCoorMP) and (key2 in ObjCoorMP):
            end1 = len(ObjCoorMP[key1]); end2 = len(ObjCoorMP[key2])
            Statistic[1][1] += 1
            Statistic[2][1] |= {key1, key2}
            if end1 == 1: Statistic[2][3] |= {key1}
            else:         Statistic[2][4] |= {key1}
            if end2 == 1: Statistic[2][3] |= {key2}
            else:         Statistic[2][4] |= {key2}
            for j1 in range(end1):
                for j2 in range(end2):
                    c = [0,0,0,0,0,0,0, set(), set()]
                    k_hits = 0
                    for k1, w1 in ObjCoorMP[key1][j1].items():
                        for k2, w2 in ObjCoorMP[key2][j2].items():
                            if (k1 + k2) in Contact_disp_1:
                                dc = iDuplicateContact(Contact_disp_1[k1 + k2], w1, w2)
                                for t in range(5): c[t] += dc[t]
                                c[-4] += dc[-2]; c[-3] += dc[-1]
                                c[-2].add(k1);   c[-1].add(k2); k_hits += 1
                            elif (k2 + k1) in Contact_disp_1:
                                dc = iDuplicateContact(Contact_disp_1[k2 + k1], w1, w2)
                                for t in range(5): c[t] += dc[t]
                                c[-4] += dc[-2]; c[-3] += dc[-1]
                                c[-2].add(k2);   c[-1].add(k1); k_hits += 1
                            else:
                                Statistic[2][5] |= {key1, key2}
                    if k_hits == 0:
                        continue
                    Statistic[1][2] |= {i}; Statistic[2][2] |= {key1, key2}
                    norm = 1.0 if model != 'balanced' else (c[-3] if c[-3] != 0 else 1.0)
                    if c[-3] == 0:
                        continue
                    disp1 = max((payload0[2] - payload0[0]), (payload0[0] - payload0[1]))
                    disp2 = max((c[2] - c[0]), (c[0] - c[1]))
                    to_write = (
                        label_from_keys(c[-2], Bbins)[:-1],
                        label_from_keys(c[-1], Bbins)[:-1],
                        int(round(payload0[0])),
                        int(round(c[0] / norm)),
                        int(round(disp1)),
                        int(round(disp2)),
                        int(round(payload0[3])),
                        int(round(payload0[4])),
                        int(round(c[3] / norm)),
                        int(round(c[4] / norm)),
                        float(round(c[-4] / norm, 2)),
                        float(round(c[-3], 5)),
                    )
                    if i not in DifferContact:
                        DifferContact[i] = to_write
                        Statistic[1][3] |= {i}
                    else:
                        Statistic[1][4].append(i)
                        if choose_best(DifferContact[i], to_write, criteria):
                            Dups.setdefault(i, []).append(to_write)
                        else:
                            Dups.setdefault(i, []).append(DifferContact[i])
                            DifferContact[i] = to_write
    Statistic_list = list(Statistic)
    Statistic_list[1][2] = len(Statistic_list[1][2])
    Statistic_list[1][5] = len(Statistic_list[1][4])
    Statistic_list[1][4] = len(set(Statistic_list[1][4]))
    Statistic_list[1][3] = len(Statistic_list[1][3]) - Statistic_list[1][4]
    Statistic_list[2][5] = Statistic_list[2][5] - Statistic_list[2][2]
    Statistic_list[2][3] = Statistic_list[2][3] - Statistic_list[2][5]
    Statistic_list[2][4] = Statistic_list[2][4] - Statistic_list[2][5]
    with open(stat_out_prefix + '.stat', 'w') as f:
        for i in range(6):
            Statistic_list[2][i] = len(Statistic_list[2][i])
            f.write(f"{Statistic_list[0][i]} {Statistic_list[1][i]} {Statistic_list[2][i]}\n")
    return DifferContact, Dups

def iPrintDifferContact(data, Abins, out_prefix):
    out_path = out_prefix + '.liftContacts'
    with open(out_path, 'w') as f:
        header = ('chr1_observed\tpos1_observed\tchr2_observed\tpos2_observed\t'
                  'remap1_control\tremap2_control\tobserved_contacts\tcontrol_contacts\t'
                  'observed_deviations\tcontrol_deviations\t'
                  'observed_coverages_pos1\tobserved_coverages_pos2\t'
                  'control_coverages_pos1\tcontrol_coverages_pos2\t'
                  'control_contact_distances\tremapping_coverages')
        f.write(header + "\n")
        Keys = sorted(list(data.keys()), key=lambda x: (x[0], x[2], x[1], x[3]))
        for key in Keys:
            i = data[key]
            c1name, c1start = Abins[key[1]][0], Abins[key[1]][1]
            c2name, c2start = Abins[key[3]][0], Abins[key[3]][1]
            line = (f"{c1name}\t{c1start}\t{c2name}\t{c2start}\t"
                    f"{i[0]}\t{i[1]}\t{i[2]}\t{i[3]}\t{i[4]}\t{i[5]}\t"
                    f"{i[6]}\t{i[7]}\t{i[8]}\t{i[9]}\t{float(i[10]):.2f}\t{float(i[11]):.5f}\n")
            f.write(line)

def _build_chr_bins(bins_dict, idx):
    chr_bins = {}
    for bid, (chrom, start, end) in bins_dict.items():
        N = idx[chrom]
        chr_bins.setdefault(N, []).append((start, bid))
    for N in chr_bins:
        chr_bins[N].sort(key=lambda x: x[0])
        chr_bins[N] = [bid for _, bid in chr_bins[N]]
    return chr_bins

def main():
    p = argparse.ArgumentParser(
        prog='liftcontacts',
        description='Liftover and compare rich contacts A→B',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--contact-a', required=True, help='A: .contacts.rich.tsv')
    p.add_argument('--contact-b', required=True, help='B: .contacts.rich.tsv')
    p.add_argument('--fadix-a', required=True, help='FASTA index (.fai) for A')
    p.add_argument('--fadix-b', required=True, help='FASTA index (.fai) for B')
    p.add_argument('--mark', required=True, help='Path to A->B .mark')
    p.add_argument('--agg-frame', type=int, default=150000, help='Aggregation frame on B (bp) to merge adjacent remapped bins')
    p.add_argument('--dups-filter', choices=['length','coverage','deviation','none','default'], default='default', help='Duplicate selection rule')
    p.add_argument('--model', choices=['balanced','raw'], default='raw', help='Normalization model')
    p.add_argument('--out-prefix', required=True, help='Output prefix')
    a = p.parse_args()
    if not _is_rich(a.contact_a) or not _is_rich(a.contact_b):
        sys.exit("Only RICH contacts (*.contacts.rich.tsv) are accepted.")
    CA_rich, Abins = read_contacts_rich(a.contact_a)
    CB_rich, Bbins = read_contacts_rich(a.contact_b)
    idxA = ChromIndexingFAI(a.fadix_a)
    idxB = ChromIndexingFAI(a.fadix_b)
    A_chr_bins = _build_chr_bins(Abins, idxA)
    B_chr_bins = _build_chr_bins(Bbins, idxB)
    A_chr_bins2, A_chr_starts, A_chr_ends = _build_chr_arrays(Abins, idxA)
    B_chr_bins2, B_chr_starts, B_chr_ends = _build_chr_arrays(Bbins, idxB)
    CA = build_contact_hash_from_rich(CA_rich, idxA)
    CB = build_contact_hash_from_rich(CB_rich, idxB)
    mark_path = a.mark if a.mark.endswith('.mark') else (a.mark + '.mark')
    if not os.path.exists(mark_path):
        sys.exit(f".mark not found: {mark_path}")
    resB = _infer_resolution_from_bins(Bbins)
    MP_A_to_B = iReadingMarkPoints(mark_path, idxA, idxB, a.agg_frame, A_chr_bins2, A_chr_starts, A_chr_ends, B_chr_bins2, B_chr_starts, B_chr_ends, resB)
    DifferA, DupsA = iDifferContact(CA, CB, MP_A_to_B, a.model, a.dups_filter, idxB, Bbins, a.out_prefix)
    iPrintDifferContact(DifferA, Abins, a.out_prefix)

if __name__ == '__main__':
    main()
