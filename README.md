# Coolsecture

Coolsecture compares Hi-C contact maps across species. Given two genome assemblies, a synteny map, and one Hi-C matrix per species, it lifts contacts between coordinate systems and measures cross-species agreement.

Coolsecture builds on [C-InterSecture](https://github.com/NuriddinovMA/C-InterSecture) with a CLI-based workflow, native `.cool`/`.mcool`/`.hic` support, multi-resolution analysis, and lower-memory contact liftover. The `liftcontacts` and `lift2matrix` steps can spill intermediate tables to disk with `--spill-threshold-mb` and `--tmp-dir`; other commands still load their main inputs into memory.

## Main features

- Assembly alignment with `minimap2` or `mummer4`.
- Conversion of `.link` or UCSC `.chain` files to Coolsecture `.mark` format.
- Contact ranking from `.cool`, `.mcool`, and `.hic` matrices.
- Bidirectional contact liftover and reciprocal consistency summaries.
- Reconstruction of observed/target `.cool` or `.hic` matrices.
- PBAD and related metrics, diagnostic plots, and split-triangle cross-species heatmaps.
- HiCRep-inspired SCC-like similarity scoring.
- Command-by-command execution, `run-all`, and example Snakemake workflows.

## Installation

Requires Python 3.8+.

```bash
git clone https://github.com/pk-zhu/Coolsecture.git
cd Coolsecture
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[hic]"    # .hic input
python -m pip install -e ".[viz]"    # Plotly HTML
python -m pip install -e ".[stats]"  # Spearman metric + KDE plotting
```

External tools:

- `minimap2` — default aligner for `asm2link` and `run-all`.
- `mummer4` (`nucmer`, `delta-filter`, `show-coords`) — required only with `-a mummer4` / `--asm-aligner mummer4`.
- `samtools` — optional; `run-all` can create a minimal `.fai` if `samtools faidx` is unavailable.
- `juicer_tools` — required only for `lift2matrix --format hic` or `--format both`.
- `snakemake` — required only for the example workflows.

## Command overview

```bash
coolsecture -h
coolsecture <command> -h
```

| Command | Purpose |
| --- | --- |
| `asm2link` | Align two assemblies and write `.paf` plus six-column `.link`. |
| `link2mark` | Convert `.link` or UCSC `.chain` files to `.mark`. |
| `prepare` | Convert `.cool`, `.mcool`, or `.hic` to ranked contact tables. |
| `roughlift` | Roughly lift a BED track for quick synteny QA. |
| `liftcontacts` | Run A→B and B→A contact liftover and reciprocal summaries. (`liftcontracts` is a deprecated alias.) |
| `contact-stat` | Plot percentile, distance, and ratio diagnostics. |
| `metric` | Compute PBAD and related metrics as bedGraph plus figures. |
| `lift2matrix` | Convert lifted contacts to observed/target `.cool` or `.hic` matrices. |
| `plot-cross` | Draw split-triangle cross-species heatmaps. |
| `multiscale` | Summarize divergence stability across resolutions (`--metric pbad`/`log`/`stripe`/`pearsone`/`spearman`). |
| `similarity` | Compute a HiCRep-inspired SCC-like similarity score. |
| `run-all` | Run the main end-to-end workflow. |

> In the current CLI, FASTA index arguments are named `--fadix`, `--fadix-a`, and `--fadix-b`.

## Inputs

You need:

- FASTA and `.fai` files for both species.
- One Hi-C matrix per species in `.cool`, `.mcool`, or `.hic` format.
- A synteny file: six-column `.link` or UCSC `.chain`.

`.link` format (0-based, half-open):

```text
chromA  startA  endA  chromB  startB  endB
```

For `.cool` / `.mcool`, the bins table must contain at least one normalization vector among `KR`, `VC_SQRT`, `VC`, or `weight`.

## Quick start

`run-all` performs assembly alignment, mark generation, contact preparation, bidirectional liftover, diagnostics, metric calculation, matrix reconstruction, similarity scoring, multi-resolution stability summary, and automatic `plot-cross` region selection.

```bash
coolsecture run-all \
  --genome-a Asu.fa \
  --genome-b Ath.fa \
  --matrix-a Asu.mcool \
  --matrix-b Ath.mcool \
  --resolution 40000 \
  --name-a Asu \
  --name-b Ath \
  --out-prefix run_all
```

Arguments can be forwarded to individual steps:

```bash
coolsecture run-all \
  --genome-a Asu.fa \
  --genome-b Ath.fa \
  --matrix-a Asu.mcool \
  --matrix-b Ath.mcool \
  --resolution 40000 \
  --name-a Asu \
  --name-b Ath \
  --prepare-args "--max-distance 5000000000 --inter" \
  --liftcontacts-args "--model balanced --dups-filter coverage --nthreads 8" \
  --metric-args "--frames 8 --metric pbad" \
  --out-prefix run_all
```

For `.hic` input:

```bash
python -m pip install -e ".[hic]"

coolsecture run-all \
  --genome-a hg38.fa \
  --genome-b mm10.fa \
  --matrix-a GM12878.hic \
  --matrix-b mESC.hic \
  --resolution 100000 \
  --name-a GM12878 \
  --name-b mESC \
  --out-prefix run_all_hic
```

`run-all` accepts one resolution. Standalone `prepare` also supports comma-separated resolutions for `.hic`, for example `--resolution 40000,100000`.

## Learn more

For the full command-by-command walkthrough (all nine steps, options, and outputs), see [docs/workflow.md](docs/workflow.md).

- [docs/workflow.md](docs/workflow.md) — detailed `asm2link` → `link2mark` → `prepare` → `liftcontacts` → `contact-stat` → `metric` → `lift2matrix` → `similarity` → `plot-cross` workflow.
- [docs/file-formats.md](docs/file-formats.md) — `.link`, `.mark`, `.contacts.tsv`, `.liftContacts` columns and the normalization conventions.
- [docs/reference.md](docs/reference.md) — Snakemake examples, interactive Plotly HTML, releases/CI/tests.
- [docs/VISUALIZATIONS.md](docs/VISUALIZATIONS.md) — which TSV tables get interactive charts.
- `example1/` (`.mcool` + `.link`) and `example2/` (`.hic` + UCSC `.chain`) are runnable end-to-end examples.

## Changelog

### v0.4.1 — 2026-09-10

- `similarity`: added optional HiCRep-style 2-D stratum smoothing. `--smooth-h h` smooths over a `(2h+1)×(2h+1)` bin window before the per-stratum correlations (default `0` keeps the unsmoothed SCC-like value); `--smooth-h auto` scans `h` and selects the SCC plateau onset, writing the SCC(h) curve to `*.scc-h.tsv`.
- `multiscale`: added `--metric {pbad,log,stripe,pearsone,spearman}` (default `pbad`) and `--metric-thr`; all metrics are mapped to a common divergence score (`1-r` for correlations, `|log ratio|` for log metrics). Outputs are named per metric (`*.multiscale.<metric>.tsv`/`.summary.tsv`).

### v0.3.5 — 2026-07-06

- Added mummer4 support to `asm2link` (`-a mummer4`; `nucmer` + `delta-filter` + `show-coords`). minimap2 remains the default.
- Added `--mummer-filter {1-to-1,mutual-best,none}` (default `1-to-1`), `--mummer-min-idy`, and `--mummer-min-len`.
- Added `run-all --asm-aligner` and `--asm-mummer-filter`.
- Fixed several CLI help strings, including `prepare`, `liftcontacts --contact-a`, `--dups-filter`, `--model`, and duplicated `(default: auto)` text.

### v0.3.2 — 2026-06-14

- Switched static plots to editable PDF/SVG text with Carlito-preferred fonts.
- Added multi-resolution `.hic` support to `prepare` and `run-all`.
- Added chromosome-name mapping output when aliases are used during liftover matrix generation.
- Added automatic parameter selection with `run-all --auto` and `auto_params.tsv`.
- Added automatic `plot-cross` region selection for top differential/conserved 2 Mb PBAD-ranked regions.

## Citation

If you use Coolsecture in your research, please cite the paper.

```text
# Peer-reviewed paper 
Zhu P. et al.Coolsecture: an easy to use and improved framework for cross species Hi C contact map comparison. Bioinformatics (Accepted). doi: <DOI>

```
