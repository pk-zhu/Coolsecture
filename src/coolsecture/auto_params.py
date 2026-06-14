#!/usr/bin/env python3
from pathlib import Path


def _parse_res_from_uri(path: str):
    if "::resolutions/" not in path:
        return None
    try:
        return int(path.rsplit("::resolutions/", 1)[1].split("/", 1)[0])
    except Exception:
        return None


def available_resolutions(matrix_path: str):
    uri_res = _parse_res_from_uri(matrix_path)
    if uri_res:
        return [uri_res]
    low = matrix_path.lower()
    if low.endswith(".cool") and not low.endswith(".mcool"):
        import cooler
        clr = cooler.Cooler(matrix_path)
        return [int(clr.binsize)] if clr.binsize else []
    if low.endswith(".mcool"):
        import cooler
        out = []
        for group in cooler.fileops.list_coolers(matrix_path):
            if group.startswith("/resolutions/"):
                try:
                    out.append(int(group.rsplit("/", 1)[1]))
                except Exception:
                    pass
        return sorted(set(out))
    if low.endswith(".hic"):
        try:
            import hicstraw
            hic = hicstraw.HiCFile(matrix_path)
            return sorted(int(x) for x in hic.getResolutions())
        except Exception:
            return [10000, 40000, 100000]
    return []


def pick_resolution(matrix_a: str, matrix_b: str):
    res_a = set(available_resolutions(matrix_a))
    res_b = set(available_resolutions(matrix_b))
    shared = sorted(res_a & res_b)
    if not shared:
        raise RuntimeError("Could not find a shared resolution between matrix-a and matrix-b")
    preferred = [10000, 25000, 40000, 50000, 100000]
    for res in preferred:
        if res in shared:
            return res, f"shared preferred resolution {res}"
    return shared[0], f"smallest shared resolution {shared[0]}"


def pick_metric_frames(resolution: int):
    frame = max(4, int(round(200000 / float(resolution))))
    return [frame]


def pick_max_dist_mb(fai_path: str, resolution: int):
    sizes = []
    with open(fai_path) as f:
        for line in f:
            a = line.split()
            if len(a) >= 2:
                try:
                    sizes.append(int(a[1]))
                except Exception:
                    pass
    if not sizes:
        return 10.0
    mean_chr_mb = (sum(sizes) / len(sizes)) / 1e6
    return max(5.0, min(50.0, mean_chr_mb * 0.05))


def write_auto_params(path: str, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("parameter\tvalue\treason\n")
        for key, value, reason in rows:
            f.write(f"{key}\t{value}\t{reason}\n")
    print(f"[OK] wrote {path}")
