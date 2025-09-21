#!/usr/bin/env python3
import os, sys, time, math
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
try:
    import scipy.stats as st
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
try:
    from sklearn.neighbors import KernelDensity
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False
import pandas as pd

def infer_resolution_from_liftover(path: str, sample_lines: int = 200000) -> int:
    steps, widths, last_pos = [], [], {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            a = line.split()
            if len(a) < 6:
                continue
            try:
                c1, p1 = a[0], int(a[1])
                c2, p2 = a[2], int(a[3])
                for (c, p) in ((c1, p1), (c2, p2)):
                    if c in last_pos:
                        d = abs(p - last_pos[c])
                        if d > 0:
                            steps.append(d)
                    last_pos[c] = p
            except Exception:
                pass
            for tok in (a[4], a[5]):
                if tok and ':' in tok and '-' in tok:
                    try:
                        lhs, rhs = tok.split('-')[:2]
                        s = int(lhs.split(':')[1]); e = int(rhs.split(':')[1])
                        w = abs(e - s)
                        if w > 0:
                            widths.append(w)
                    except Exception:
                        pass
    vals = [v for v in (steps + widths) if v > 0]
    if not vals:
        return 40000
    g = vals[0]
    for v in vals[1:]:
        g = math.gcd(g, v)
        if g == 1:
            break
    if g < 1000 and widths:
        from collections import Counter
        g = Counter(widths).most_common(1)[0][0]
    if g < 1000:
        g = max(10000, g)
    return int(g)

def ChromIndexingFAI(path: str) -> Dict[str,int]:
    d: Dict[str,int] = {}
    with open(path) as f:
        for i, line in enumerate(f, 1):
            toks = line.split()
            if not toks:
                continue
            name = toks[0]
            d[name] = i
            d[i] = name
    return d

def read_chr_sizes_from_fai(path: str) -> Dict[str,int]:
    sizes: Dict[str,int] = {}
    with open(path) as f:
        for line in f:
            toks = line.split()
            if len(toks) < 2:
                continue
            sizes[toks[0]] = int(toks[1])
    return sizes

def read_contacts(file_path: str, Order: Dict, resolution: int, short: bool=False):
    start = time.time()
    with open(file_path) as f:
        lines = f.readlines()
    ln = len(lines)
    if not short:
        Contacts: Dict[Tuple[str,int,str,int], List[float]] = {}
        for i in range(ln-1, 0, -1):
            a = lines[i].split()
            try:
                b1 = int(a[1]) // resolution
                b2 = int(a[3]) // resolution
                if (Order[a[0]] < Order[a[2]]) or (Order[a[0]] == Order[a[2]] and b1 <= b2):
                    key = (a[0], b1, a[2], b2)
                else:
                    key = (a[2], b2, a[0], b1)
                c  = float(a[6])
                q  = float(a[7])
                dr = float(a[8])
                dq = float(a[9])
                l  = float(a[-2])
            except Exception:
                continue
            if key not in Contacts or l < Contacts[key][-1]:
                Contacts[key] = [c,q,dr,dq,l]
            if (ln - i) % 1000000 == 0:
                print(f"contact reading progress: {ln-i}, elapsed: {time.time()-start:.2f}s")
                sys.stdout.flush()
        print("total time elapsed: %.2f" % (time.time()-start))
        return Contacts
    else:
        out: List[Tuple[float,float,int,float]] = []
        for i in range(ln-1, 0, -1):
            a = lines[i].split()
            try:
                b1 = int(a[1]) // resolution
                b2 = int(a[3]) // resolution
                lr = abs(b2 - b1) if (Order[a[0]] == Order[a[2]]) else -1000
                c  = float(a[6]); q = float(a[7]); lq = float(a[-2])
                out.append((c,q,lr,lq))
            except Exception:
                continue
        print("total time elapsed: %.2f" % (time.time()-start))
        return out

def _parse_remap_token(tok: str) -> Optional[Tuple[str,int]]:
    if not tok or tok == '-':
        return None
    if '-' in tok:
        lhs, rhs = tok.split('-')[:2]
        c1, p1 = lhs.split(':')[0], lhs.split(':')[1]
        c2, p2 = rhs.split(':')[0], rhs.split(':')[1]
        if c1 != c2:
            return None
        try:
            m = (int(p1) + int(p2)) // 2
            return (c1, m)
        except Exception:
            return None
    else:
        if ':' not in tok:
            return None
        c, p = tok.split(':')[:2]
        try:
            return (c, int(p))
        except Exception:
            return None

def extract_arrays_for_contact_stat(lift_path: str, resolution: int, Order: Dict):
    pA, pB, dA, dB, ok = [], [], [], [], []
    with open(lift_path) as f:
        header = f.readline()
        for line in f:
            a = line.split()
            if len(a) < 16:
                continue
            try:
                ca1, xa1 = a[0], int(a[1])
                ca2, xa2 = a[2], int(a[3])
                pa = float(a[6]); pb = float(a[7])
            except Exception:
                continue
            if ca1 == ca2:
                dA_Mb = abs(xa2 - xa1) / 1e6
            else:
                dA_Mb = np.nan
            rm1 = _parse_remap_token(a[4])
            rm2 = _parse_remap_token(a[5])
            if rm1 and rm2 and (rm1[0] == rm2[0]):
                dB_Mb = abs(rm2[1] - rm1[1]) / 1e6
            else:
                try:
                    dB_Mb = float(a[-2])
                except Exception:
                    dB_Mb = np.nan
            pA.append(pa); pB.append(pb)
            dA.append(dA_Mb); dB.append(dB_Mb)
            ok.append((not np.isnan(dA_Mb)) and (not np.isnan(dB_Mb)))
    return (np.array(pA), np.array(pB), np.array(dA), np.array(dB), np.array(ok, dtype=bool))

def randomize_vector(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    np.random.shuffle(y)
    return y

def randomize_contacts(Contacts, idx: int = 1):
    if isinstance(Contacts, list):
        pool = [row[idx] for row in Contacts]
        np.random.shuffle(pool)
        out = []
        for i, row in enumerate(Contacts):
            tmp = list(row)
            tmp[idx] = pool[i]
            out.append(tuple(tmp))
        return out
    elif isinstance(Contacts, dict):
        keys = list(Contacts.keys())
        pool = [Contacts[k][idx] for k in keys]
        np.random.shuffle(pool)
        out = {}
        for i, k in enumerate(keys):
            v = list(Contacts[k])
            v[idx] = pool[i]
            out[k] = v
        return out
    else:
        raise TypeError("Contacts must be list or dict.")

def _metric_pbad(contacts, locus, locusKeys, max_dist, frame):
    I = 0.0; n = 0; threshold = frame**2
    for j in locusKeys:
        for k in locusKeys:
            if abs(j[1]-k[1]) > max_dist:
                continue
            key = j + k
            if key in contacts:
                p1, p2 = contacts[key][0], contacts[key][1]
                dp = abs(p1-p2) / 100.0
                ds1 = 1.0 - abs(p1-50)/50.0
                ds2 = 1.0 - abs(p2-50)/50.0
                ds1 = min(max(ds1, 0.01), 0.99)
                ds2 = min(max(ds2, 0.01), 0.99)
                I += -1.0*dp*np.log10(ds1*ds2)
                n += 1
    if n > threshold: return (locus[0], locus[1], locus[2], I/n)

def _metric_log(contacts, locus, locusKeys, max_dist, frame):
    I = 0.0; n = 0; threshold = frame**2
    for j in locusKeys:
        for k in locusKeys:
            if abs(j[1]-k[1]) > max_dist:
                continue
            key = j + k
            if key in contacts:
                p1, p2 = contacts[key][0], contacts[key][1]
                I += np.log10(max(p1,1e-6)/max(p2,1e-6))
                n += 1
    if n > threshold: return (locus[0], locus[1], locus[2], I/n)

def _metric_stripe(contacts, locus, locusKeys, max_dist, frame):
    I = 0.0; n = 0; threshold = 0.4*frame
    for j in locusKeys:
        key = j + locus[:2]
        if key in contacts:
            p1, p2 = contacts[key][0], contacts[key][1]
            I += np.log10(max(p1,1e-6)/max(p2,1e-6))
            n += 1
    if n > threshold: return (locus[0], locus[1], locus[2], I/n)

def _metric_pearson(contacts, locus, locusKeys, max_dist, frame):
    X, Y = [], []; threshold = frame**2
    for j in locusKeys:
        for k in locusKeys:
            if abs(j[1]-k[1]) > max_dist:
                continue
            key = j + k
            if key in contacts:
                p1, p2 = contacts[key][0], contacts[key][1]
                X.append(p1); Y.append(p2)
    if len(X) > threshold:
        return (locus[0], locus[1], locus[2], float(np.corrcoef(X,Y)[0,1]))

def _metric_spearman(contacts, locus, locusKeys, max_dist, frame):
    if not HAVE_SCIPY:
        raise RuntimeError("spearman requires scipy")
    X, Y = [], []; threshold = frame**2
    for j in locusKeys:
        for k in locusKeys:
            if abs(j[1]-k[1]) > max_dist:
                continue
            key = j + k
            if key in contacts:
                p1, p2 = contacts[key][0], contacts[key][1]
                X.append(p1); Y.append(p2)
    if len(X) > threshold:
        return (locus[0], locus[1], locus[2], float(st.spearmanr(X,Y)[0]))

def metricCalc(contacts: dict, resolution: int, frame: int=8, metric: str='pbad', max_dist_mb: float=100.0, loci=None):
    max_dist = int(max_dist_mb * 1e6 / resolution)
    func = {'pbad':_metric_pbad,'log':_metric_log,'stripe':_metric_stripe,'pearsone':_metric_pearson,'spearman':_metric_spearman}.get(metric, _metric_pbad)
    results = []
    if loci is None:
        keys = set([])
        for k in contacts.keys():
            keys.add(k[:2]); keys.add(k[2:])
        anchors = sorted(keys)
        for key in anchors:
            locus = (key[0], key[1], key[1]+1)
            locusKeys = [(key[0], i) for i in range(key[1]-frame, key[1]+frame+1)]
            r = func(contacts, locus, locusKeys, max_dist, frame)
            if r is not None: results.append(r)
    else:
        for locus in loci:
            locusKeys = [(locus[0], i) for i in range(locus[1]//resolution, locus[2]//resolution + 1)]
            r = func(contacts, locus, locusKeys, max_dist, 0)
            if r is not None: results.append(r)
    return results

def _log2_ratio_safe(a: np.ndarray, b: np.ndarray, eps: float=1e-9) -> np.ndarray:
    return np.log2((np.maximum(a, 0) + eps) / (np.maximum(b, 0) + eps))

def _save_fig(fig, path, fmt='pdf', dpi=300):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    if fmt.lower() == 'pdf':
        fig.savefig(path, format='pdf', bbox_inches='tight')
    else:
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')
    print(f"[OK] wrote {path}")

def make_pinkblue_cmap():
    return LinearSegmentedColormap.from_list("pinkblue", ["#2b6cb0", "#66b2ff", "#ffffff", "#fbb6ce", "#e53e6a"], N=256)

def make_diverging_norm(data2d, vcenter=0.0, vmax=None):
    finite = np.isfinite(data2d)
    if vmax is None:
        vmax = np.quantile(np.abs(data2d[finite]), 0.99) if finite.any() else 1.0
        vmax = max(vmax, 1e-6)
    return TwoSlopeNorm(vmin=-vmax, vcenter=vcenter, vmax=vmax)

def _read_bed_positions(bed_path: str, chrom: str, start: int, end: int, use_midpoint: bool=True):
    pos = []
    with open(bed_path) as f:
        for ln in f:
            if not ln or ln[0] in "#tb":
                continue
            a = ln.split()
            if len(a) < 3:
                continue
            if a[0] != chrom:
                continue
            s = int(a[1]); e = int(a[2])
            x = (s + e) // 2 if use_midpoint else s
            if start <= x < end:
                pos.append(x)
    pos.sort()
    return pos

def _drawMap(allCon: dict, locus: Tuple[str,int,int], resolution: int):
    Keys = sorted(allCon.keys())
    Y1=[]; X1=[]; S1=[]; C1=[]; S2=[]; C2=[]
    for key in Keys:
        X1.append(key[3] - locus[1]//resolution)
        Y1.append(key[1] - locus[1]//resolution)
        s1 = (.1*(100 - allCon[key][2]))**2
        s1 = min(100, max(4, s1))
        S1.append(int(s1))
        C1.append(allCon[key][0])
        s2 = .1*(100 - allCon[key][2] - allCon[key][3])
        s2 = max(2, s2); s2 = min(100, s2**2)
        S2.append(int(s2))
        c2 = allCon[key][0] - allCon[key][1]
        C2.append(c2)
    return (Y1, X1, S1, C1, S2, C2, [])

def read_contacts_full(lift_path: str, Order: Dict, resolution: int):
    return read_contacts(lift_path, Order, resolution, short=False)

def _parse_locus(locus: str, sizes: dict):
    if ':' in locus:
        chrom, span = locus.split(':', 1)
        s, e = span.replace(',', '').split('-', 1)
        start, end = int(s), int(e)
    else:
        chrom = locus
        if chrom not in sizes:
            raise ValueError(f"chrom '{chrom}' not in .fai")
        start, end = 0, sizes[chrom]
    if end <= start:
        raise ValueError("invalid region: end must be > start")
    return chrom, start, end

def _parse_borders_arg(arg: Optional[str]):
    if not arg:
        return []
    items = []
    for tok in arg.split(','):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(':')
        path  = parts[0]
        label = parts[1] if len(parts) > 1 and parts[1] else None
        color = parts[2] if len(parts) > 2 and parts[2] else None
        items.append((path, label, color))
    return items

def _parse_region(spec: str):
    n, span = spec.split(':', 1)
    s, e = span.replace(',', '').split('-', 1)
    return (n, int(s), int(e))

def _bins_for_region(region, resolution):
    chrom, s, e = region
    bs = s // resolution
    be = e // resolution
    return chrom, bs, be, be - bs

def _matrix_for_regions(contacts: dict, resolution: int, reg_y, reg_x, mode: str = 'obs'):
    cy, by_s, by_e, ny = _bins_for_region(reg_y, resolution)
    cx, bx_s, bx_e, nx = _bins_for_region(reg_x, resolution)
    M = np.full((ny, nx), np.nan, dtype=float)
    eps = 1e-6
    for (c1, b1, c2, b2), vals in contacts.items():
        if c1 == cy and c2 == cx:
            iy = b1 - by_s
            ix = b2 - bx_s
            if 0 <= iy < ny and 0 <= ix < nx:
                obs = float(vals[0]); ctrl = float(vals[1])
                if   mode == 'obs':      v = obs
                elif mode == 'ctrl':     v = ctrl
                elif mode == 'diff':     v = obs - ctrl
                else:                    v = np.log2((obs + eps) / (ctrl + eps))
                M[iy, ix] = v
                if cy == cx and 0 <= ix < ny and 0 <= iy < nx:
                    M[ix, iy] = v
    return M

def _read_bedgraph_track(bg_path: str, region, resolution: int):
    chrom, bs, be, nx = _bins_for_region(region, resolution)
    v = np.zeros(nx, dtype=float)
    with open(bg_path) as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            a = line.split()
            if len(a) < 4:
                continue
            if a[0] != chrom:
                continue
            s = int(a[1]) // resolution
            e = (int(a[2]) + (resolution - 1)) // resolution
            val = float(a[3])
            s = max(s, bs); e = min(e, be)
            if e <= s:
                continue
            v[s-bs:e-bs] = val
    return v

def _build_bins_from_sizes(chr_sizes: Dict[str,int], resolution: int):
    chroms=[]; starts=[]; ends=[]
    for chrom, size_bp in chr_sizes.items():
        nbin = size_bp // resolution
        for i in range(nbin):
            chroms.append(chrom); starts.append(i*resolution); ends.append((i+1)*resolution)
    return pd.DataFrame({'chrom':chroms,'start':starts,'end':ends})

def _pixels_from_contacts(contacts: dict, resolution: int, chr_sizes: dict, which: str='observed'):
    offset = {}
    nbin_by_chr = {}
    cum = 0
    for chrom, size_bp in chr_sizes.items():
        nbin = size_bp // resolution
        offset[chrom] = cum
        nbin_by_chr[chrom] = nbin
        cum += nbin
    val_idx = 0 if which == 'observed' else 1
    bin1 = []; bin2 = []; count = []
    for (c1, b1, c2, b2), vals in contacts.items():
        if (c1 not in offset) or (c2 not in offset):
            continue
        if b1 < 0 or b2 < 0:
            continue
        if b1 >= nbin_by_chr[c1] or b2 >= nbin_by_chr[c2]:
            continue
        i = offset[c1] + b1
        j = offset[c2] + b2
        if j < i:
            i, j = j, i
        bin1.append(i); bin2.append(j); count.append(float(vals[val_idx]))
    if not bin1:
        return pd.DataFrame({'bin1_id': [], 'bin2_id': [], 'count': []}, dtype='int64')
    return pd.DataFrame({'bin1_id': bin1, 'bin2_id': bin2, 'count': count})

def _write_cool_minimal(path, bins_df, px_df, assembly=None):
    import cooler
    import h5py
    try:
        cooler.create_cooler(path, bins=bins_df, pixels=px_df, dtypes={'count': 'float32'})
    except TypeError:
        cooler.create_cooler(path, bins_df, px_df)
    if assembly:
        with h5py.File(path, 'r+') as hf:
            hf['/'].attrs['assembly'] = str(assembly)
