#!/usr/bin/env python3
import os, re, sys, time, math
from typing import Dict, Tuple, List, Optional, Iterable
import numpy as np
import matplotlib

_FONT_WARNED = False

def configure_matplotlib_for_publication(font_family: str = "Carlito"):
    global _FONT_WARNED
    matplotlib.use('Agg', force=True)
    fallback_fonts = [font_family, "Arial", "DejaVu Sans"]
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": fallback_fonts,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
        "axes.unicode_minus": False,
    })
    if not _FONT_WARNED:
        try:
            from matplotlib import font_manager
            font_manager.findfont(font_family, fallback_to_default=False)
        except Exception:
            print(f"[WARN] Font '{font_family}' not found; falling back to Arial/DejaVu Sans.")
        _FONT_WARNED = True

configure_matplotlib_for_publication()
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
            for alias in chromosome_aliases(name):
                d.setdefault(alias, i)
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

def chromosome_aliases(name: str) -> List[str]:
    aliases = [name]
    if name.startswith("chr"):
        alt = name[3:]
        if alt:
            aliases.append(alt)
    else:
        aliases.append(f"chr{name}")
    seen = set()
    out: List[str] = []
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            out.append(alias)
    return out

def build_chrom_alias_map(names: Iterable[str]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    collisions = set()
    for name in names:
        for alias in chromosome_aliases(name):
            prev = alias_map.get(alias)
            if prev is None:
                alias_map[alias] = name
            elif prev != name:
                collisions.add(alias)
    for alias in collisions:
        alias_map.pop(alias, None)
    return alias_map

def canonicalize_chrom_name(name: str, alias_map: Dict[str, str]) -> Optional[str]:
    return alias_map.get(name)

def read_contacts(file_path: str, Order: Dict, resolution: int, short: bool=False):
    start = time.time()
    open_kwargs = {'buffering': 1024 * 1024} if sys.platform.startswith('linux') else {}
    if not short:
        Contacts: Dict[Tuple[str,int,str,int], List[float]] = {}
        processed = 0
        with open(file_path, **open_kwargs) as f:
            _ = f.readline()
            for line in f:
                processed += 1
                a = line.split()
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
                    Contacts[key] = [c, q, dr, dq, l]
                if processed % 1000000 == 0:
                    print(f"contact reading progress: {processed}, elapsed: {time.time()-start:.2f}s")
                    sys.stdout.flush()
        print("total time elapsed: %.2f" % (time.time()-start))
        return Contacts
    else:
        out: List[Tuple[float,float,int,float]] = []
        processed = 0
        with open(file_path, **open_kwargs) as f:
            _ = f.readline()
            for line in f:
                processed += 1
                a = line.split()
                try:
                    b1 = int(a[1]) // resolution
                    b2 = int(a[3]) // resolution
                    lr = abs(b2 - b1) if (Order[a[0]] == Order[a[2]]) else -1000
                    c  = float(a[6]); q = float(a[7]); lq = float(a[-2])
                    out.append((c, q, lr, lq))
                except Exception:
                    continue
                if processed % 1000000 == 0:
                    print(f"contact reading progress: {processed}, elapsed: {time.time()-start:.2f}s")
                    sys.stdout.flush()
        print("total time elapsed: %.2f" % (time.time()-start))
        return out

def _parse_remap_token(tok: str) -> Optional[Tuple[str,int]]:
    if not tok or tok == '-':
        return None
    # C-InterSecture-style labels may contain multiple loci concatenated by '-':
    # e.g. "chr1:1000-chr1:2000-chr1:4000". Use one representative midpoint.
    hits = re.findall(r'([A-Za-z0-9_.]+):(-?\d+)', tok)
    if hits:
        chroms = {c for c, _ in hits}
        if len(chroms) != 1:
            return None
        chrom = next(iter(chroms))
        try:
            coords = [int(p) for _, p in hits]
        except Exception:
            return None
        if not coords:
            return None
        m = (min(coords) + max(coords)) // 2
        return (chrom, m)
    if ':' not in tok:
        return None
    c, p = tok.split(':')[:2]
    try:
        return (c, int(p))
    except Exception:
        return None

def _read_lift_as_map(path: str, res: int):
    out = {}
    with open(path) as f:
        _ = f.readline()
        for line in f:
            a = line.split()
            if len(a) < 16:
                continue
            c1, p1 = a[0], int(a[1])
            c2, p2 = a[2], int(a[3])
            rm1 = _parse_remap_token(a[4])
            rm2 = _parse_remap_token(a[5])
            if not rm1 or not rm2:
                continue
            b1 = (c1, p1 // res)
            b2 = (c2, p2 // res)
            key = (b1, b2) if b1 <= b2 else (b2, b1)
            rb1 = (rm1[0], rm1[1] // res)
            rb2 = (rm2[0], rm2[1] // res)
            val = (rb1, rb2) if rb1 <= rb2 else (rb2, rb1)
            out[key] = val
    return out

def _reciprocal_stats(map_ab: Dict, map_ba: Dict):
    keys_ab = set(map_ab.keys())
    keys_ba = set(map_ba.keys())
    inter = 0
    for k in keys_ab:
        v = map_ab.get(k)
        if v in map_ba:
            inter += 1
    denom = len(keys_ab)
    reciprocal = inter / denom if denom else 0.0
    symmetry = (2 * inter / (len(keys_ab) + len(keys_ba))) if (keys_ab or keys_ba) else 0.0
    return reciprocal, symmetry, inter, len(keys_ab), len(keys_ba)

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
                    v = float(a[-2])
                    # Legacy liftContacts stores target distance as bins; convert using resolution.
                    # If value looks already in bp, convert directly to Mb.
                    dB_Mb = (v / 1e6) if v >= 1e6 else (v * float(resolution) / 1e6)
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
    configure_matplotlib_for_publication()
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    if fmt.lower() == 'pdf':
        fig.savefig(path, format='pdf', bbox_inches='tight')
    else:
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')
    print(f"[OK] wrote {path}")

def _import_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
        return go, pio, make_subplots
    except Exception:
        return None, None, None

def _resolve_interactive(mode: str) -> bool:
    mode = (mode or "auto").lower()
    if mode not in ("auto", "on", "off"):
        raise RuntimeError("Invalid --interactive value. Use auto, on, or off.")
    if mode == "off":
        return False
    go, _, _ = _import_plotly()
    if mode == "on" and go is None:
        raise RuntimeError("Plotly is not installed. Install with: pip install plotly")
    return go is not None

def _write_plotly_html(fig, path: str, title: Optional[str]=None):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        font=dict(family="Carlito, Arial, sans-serif"),
        margin=dict(l=60, r=20, t=60, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
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
    be = (e + resolution - 1) // resolution
    return chrom, bs, be, be - bs

def _matrix_for_regions(contacts: dict, resolution: int, reg_y, reg_x, mode: str = 'obs'):
    cy, by_s, by_e, ny = _bins_for_region(reg_y, resolution)
    cx, bx_s, bx_e, nx = _bins_for_region(reg_x, resolution)
    M = np.full((ny, nx), np.nan, dtype=float)
    eps = 1e-6

    # Build chromosome alias map for both region chromosomes
    from typing import Set
    chroms_to_map: Set[str] = set()
    chroms_to_map.add(cy)
    chroms_to_map.add(cx)
    # Also add aliases for these chromosomes
    alias_map = build_chrom_alias_map(chroms_to_map)

    # Get canonical forms for region chromosomes
    cy_can = canonicalize_chrom_name(cy, alias_map) or cy
    cx_can = canonicalize_chrom_name(cx, alias_map) or cx

    for (c1, b1, c2, b2), vals in contacts.items():
        # Canonicalize contact chromosomes
        c1_can = canonicalize_chrom_name(c1, alias_map) or c1
        c2_can = canonicalize_chrom_name(c2, alias_map) or c2

        if c1_can == cy_can and c2_can == cx_can:
            iy = b1 - by_s
            ix = b2 - bx_s
            if 0 <= iy < ny and 0 <= ix < nx:
                obs = float(vals[0]); tgt = float(vals[1])
                if   mode == 'obs':      v = obs
                elif mode == 'tgt':      v = tgt
                elif mode == 'diff':     v = obs - tgt
                else:                    v = np.log2((obs + eps) / (tgt + eps))
                M[iy, ix] = v
                if cy_can == cx_can and 0 <= ix < ny and 0 <= iy < nx:
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
        nbin = (size_bp + resolution - 1) // resolution
        for i in range(nbin):
            chroms.append(chrom); starts.append(i*resolution); ends.append(min((i+1)*resolution, size_bp))
    return pd.DataFrame({'chrom':chroms,'start':starts,'end':ends})

def _pixels_from_contacts(contacts: dict, resolution: int, chr_sizes: dict, which: str='observed'):
    offset = {}
    nbin_by_chr = {}
    cum = 0
    for chrom, size_bp in chr_sizes.items():
        nbin = (size_bp + resolution - 1) // resolution
        offset[chrom] = cum
        nbin_by_chr[chrom] = nbin
        cum += nbin
    # Build chromosome alias map
    alias_map = build_chrom_alias_map(chr_sizes.keys())
    val_idx = 0 if which == 'observed' else 1
    bin1 = []; bin2 = []; count = []
    for (c1, b1, c2, b2), vals in contacts.items():
        # Canonicalize chromosome names
        c1_can = canonicalize_chrom_name(c1, alias_map)
        c2_can = canonicalize_chrom_name(c2, alias_map)
        if c1_can is None or c2_can is None:
            continue
        if (c1_can not in offset) or (c2_can not in offset):
            continue
        if b1 < 0 or b2 < 0:
            continue
        if b1 >= nbin_by_chr[c1_can] or b2 >= nbin_by_chr[c2_can]:
            continue
        i = offset[c1_can] + b1
        j = offset[c2_can] + b2
        if j < i:
            i, j = j, i
        bin1.append(i); bin2.append(j); count.append(float(vals[val_idx]))
    if not bin1:
        return pd.DataFrame({'bin1_id': [], 'bin2_id': [], 'count': []}, dtype='int64')
    return pd.DataFrame({'bin1_id': bin1, 'bin2_id': bin2, 'count': count})

def _write_cool_minimal(path, bins_df, px_df, assembly=None):
    import cooler
    import h5py
    import numpy as np
    # Try a simple approach without dask
    # First create the file structure
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Create cooler file without dask
    from cooler import create_cooler
    # Convert pixels to numpy arrays for faster processing
    if not px_df.empty:
        # Create cooler file using direct numpy arrays
        create_cooler(
            path,
            bins=bins_df,
            pixels=px_df,
            dtypes={'count': 'float32'},
            ordered=True
        )
    else:
        # Create empty cooler file
        create_cooler(
            path,
            bins=bins_df,
            pixels=px_df,
            dtypes={'count': 'float32'}
        )
    if assembly:
        with h5py.File(path, 'r+') as hf:
            hf['/'].attrs['assembly'] = str(assembly)
