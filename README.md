# Coolsecture

Coolsecture is a Python toolkit for cross-species Hi-C contact-map comparison. It
starts from genome assemblies, synteny mappings, and Hi-C contact matrices, then
builds source-coordinate contact tables, performs contact liftover, summarizes
reciprocal consistency, and generates downstream matrices and diagnostic plots.

The project is inspired by and modernizes ideas from
[C-InterSecture](https://github.com/NuriddinovMA/C-InterSecture), with a stronger
focus on reproducible command-line workflows, cooler/mcool/hic input support,
multi-resolution processing, disk-spill paths for larger runs, and publication
ready summary outputs.

## Changelog

### v0.3.5 - 2026-07-06

- Added mummer4 aligner support in `asm2link` via `-a mummer4` (nucmer +
  delta-filter + show-coords pipeline). Default aligner remains minimap2.
- Added `--mummer-filter {1-to-1,mutual-best,none}` (default `1-to-1`) and
  `--mummer-min-idy` / `--mummer-min-len` for mummer4 alignment filtering.
- Wired `--asm-aligner` and `--asm-mummer-filter` through `run-all`.
- Fixed misleading help text: `prepare` prog name, `liftcontacts --contact-a`
  file extension, missing `--dups-filter` / `--model` descriptions, and
  duplicate `(default: auto)` on several `--interactive` flags.

### v0.3.2 - 2026-06-14

- Set publication-ready static plots to use editable PDF/SVG text with Carlito-preferred fonts.
- Added multi-resolution `.hic` support in `prepare` and `run-all`.
- Added chromosome-name mapping output when liftover matrix generation uses aliases.
- Added `run-all --auto` parameter selection with `auto_params.tsv` reporting.
- Added automatic `plot-cross` region selection for top differential/conserved 2 Mb PBAD-ranked regions.

## Main Features

- Convert assemblies to syntenic links with `minimap2` or `mummer4`.
- Convert `.link` or UCSC `.chain` files to Coolsecture `.mark` maps.
- Convert `.cool`, `.mcool`, or `.hic` matrices to percentile-ranked contact
  tables.
- Lift contacts in both directions and report reciprocal consistency metrics.
- Generate observed/target `.cool` or `.hic` matrices from lifted contacts.
- Compute PBAD-style metrics, contact-stat plots, split-triangle cross plots, and
  HiCRep-style SCC similarity.
- Run either command-by-command, through `run-all`, or through the example
  Snakemake workflows.

## Installation

Coolsecture requires Python 3.8 or newer.

```bash
git clone https://github.com/pk-zhu/Coolsecture.git
cd Coolsecture
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[hic]"
python -m pip install -e ".[viz]"
python -m pip install -e ".[stats]"
```

External tools:

- `minimap2` is required by `asm2link` and `run-all` (default aligner).
- `mummer4` (`nucmer`, `delta-filter`, `show-coords`) is optional; needed only
  when `asm2link -a mummer4` or `run-all --asm-aligner mummer4` is used.
- `samtools` is optional; `run-all` can create simple `.fai` files itself if
  `samtools faidx` is unavailable.
- `juicer_tools` is required only when `lift2matrix --format hic` or
  `--format both` is used.
- `snakemake` is required only for the example workflow directories.

## Command Overview

```bash
coolsecture -h
coolsecture <command> -h
```

Available commands:

| Command | Purpose |
| --- | --- |
| `asm2link` | Align two assemblies with minimap2 or mummer4 and write `.paf` plus six-column `.link`. |
| `link2mark` | Convert `.link` or UCSC `.chain` synteny files to `.mark`. |
| `prepare` | Convert `.cool`, `.mcool`, or `.hic` to ranked contact tables. |
| `roughlift` | Roughly lift a BED track for quick synteny QA. |
| `liftcontracts` | Run A->B and B->A contact liftover and reciprocal summaries. |
| `contact-stat` | Plot percentile, distance, and ratio diagnostics from lifted contacts. |
| `metric` | Compute PBAD and related metrics as bedGraph plus figures. |
| `lift2matrix` | Convert lifted contacts to observed/target `.cool` or `.hic` matrices. |
| `plot-cross` | Draw split-triangle cross-species heatmaps for a locus. |
| `multiscale` | Summarize PBAD stability across multiple resolutions. |
| `cross-validate` | Compute stratum-adjusted correlation (HiCRep-style SCC) between matched matrices. |
| `run-all` | Run the main end-to-end pipeline from FASTA and matrices. |

Note: in the current CLI, FASTA index arguments are named `--fadix`,
`--fadix-a`, and `--fadix-b`.

## Inputs

Typical inputs are:

- Genome FASTA files for species A and species B.
- FASTA index files (`.fai`) for each genome.
- Hi-C matrices in `.cool`, `.mcool`, or `.hic` format.
- A synteny file, either a six-column `.link` file or a UCSC `.chain` file.

Six-column `.link` format:

```text
chromA  startA  endA  chromB  startB  endB
```

Coolsecture accepts normalized cooler files. For `.cool` and `.mcool`, the bins
table should contain at least one normalization vector among `KR`, `VC_SQRT`,
`VC`, or `weight`.

## Quick Start: End-to-End

Use `run-all` when you have both genome FASTA files and both contact matrices.
This command runs assembly alignment, mark generation, contact preparation,
bidirectional liftover, contact statistics, metrics, and matrix reconstruction.
It does not run `plot-cross`.

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

Extra arguments can be passed through to individual steps:

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
  --liftcontracts-args "--model balanced --dups-filter coverage --nthreads 8" \
  --metric-args "--frames 8 --metric pbad" \
  --out-prefix run_all
```

For `.hic` input, install the `hic` extra and provide a single resolution:

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

## Step-by-Step Workflow

### 1. Build a link file from assemblies

Using minimap2 (default):

```bash
coolsecture asm2link \
  --genome-a Asu.fa \
  --genome-b Ath.fa \
  -x asm10 \
  --out-prefix step0/Asu_Ath
```

Using mummer4 (more precise for divergent assemblies; produces 1-to-1
syntenic alignments by default):

```bash
coolsecture asm2link \
  --genome-a Asu.fa \
  --genome-b Ath.fa \
  -a mummer4 \
  --mummer-filter 1-to-1 \
  --out-prefix step0/Asu_Ath
```

Outputs (minimap2 path):

- `step0/Asu_Ath.paf`
- `step0/Asu_Ath.link`

Outputs (mummer4 path):

- `step0/Asu_Ath.delta` (raw nucmer output)
- `step0/Asu_Ath.filter.delta` (filtered; omitted when `--mummer-filter none`)
- `step0/Asu_Ath.coords.tsv` (show-coords tabular output)
- `step0/Asu_Ath.link`

If you already have a UCSC `.chain` file, skip this step and pass it directly to
`link2mark`.

### 2. Convert synteny to a mark file

From `.link`:

```bash
coolsecture link2mark \
  --link step0/Asu_Ath.link \
  --thr-len 300 \
  --step-len 150 \
  --out-prefix step0/Asu_Ath
```

From UCSC `.chain`:

```bash
coolsecture link2mark \
  --chain hg38ToMm10.over.chain \
  --thr-len 300 \
  --step-len 150 \
  --out-prefix step0/hg38_mm10
```

Output:

- `*.mark`

### 3. Prepare contact tables

Single-resolution `.cool` or `.mcool::resolutions/RES`:

```bash
coolsecture prepare \
  --matrix Asu.mcool::resolutions/40000 \
  --max-distance 5000000000 \
  --inter \
  --nthreads 8 \
  --out-prefix step1/Asu/Asu
```

Multi-resolution `.mcool`:

```bash
coolsecture prepare \
  --matrix Asu.mcool \
  --resolution 40000,100000 \
  --max-distance 5000000000 \
  --inter \
  --summary \
  --nthreads 8 \
  --out-prefix step1/Asu/Asu
```

`.hic` input:

```bash
coolsecture prepare \
  --matrix GM12878.hic \
  --resolution 100000 \
  --max-distance 5000000000 \
  --inter \
  --out-prefix step1/GM12878/GM12878
```

Main outputs:

- `*.contacts.tsv`
- `*.stats.tsv`
- `*.r<resolution>.contacts.tsv` and `*.r<resolution>.stats.tsv` in
  multi-resolution mode
- `*.multi_resolution.summary.tsv` and `.pdf` when `--summary` is used

### 4. Run bidirectional contact liftover

```bash
coolsecture liftcontracts \
  --contact-a step1/Asu/Asu.r40000.contacts.tsv \
  --contact-b step1/Ath/Ath.r40000.contacts.tsv \
  --fadix-a step0/Asu.fa.fai \
  --fadix-b step0/Ath.fa.fai \
  --mark-ab step0/Asu_Ath.mark \
  --model balanced \
  --dups-filter coverage \
  --agg-frame 400000 \
  --nthreads 8 \
  --out-prefix step2/Asu_Ath/Asu_Ath.r40000
```

Useful options for larger files:

```bash
--tmp-dir .snakemake/tmp/liftcontracts
--spill-threshold-mb 256
--hash-shards 64
--pbad-mode auto
--pbad-auto-threshold-mb 1024
```

Main outputs:

- `*.AtoB.liftContacts`
- `*.BtoA.liftContacts`
- `*.Merged.liftContacts`
- `*.bidirectional.summary.tsv`
- `*.bidirectional.tags.tsv` unless `--no-tags` is used

### 5. Generate diagnostic plots

```bash
coolsecture contact-stat \
  --liftover step2/Asu_Ath/Asu_Ath.r40000.Merged.liftContacts \
  --fadix step0/Asu.fa.fai \
  --stats-a step1/Asu/Asu.r40000.stats.tsv \
  --stats-b step1/Ath/Ath.r40000.stats.tsv \
  --bins 400 \
  --cmap RdBu_r \
  --max-dist-mb 5000 \
  --format pdf \
  --out-prefix step3/Asu_Ath.r40000
```

Outputs:

- `*.percentile_heatmap.pdf`
- `*.distance_heatmap.pdf`
- `*.ratio_scatter.pdf`

### 6. Compute metrics

```bash
coolsecture metric \
  --liftover step2/Asu_Ath/Asu_Ath.r40000.Merged.liftContacts \
  --fadix step0/Asu.fa.fai \
  --frames 8 \
  --metric pbad \
  --format pdf \
  --out-prefix step3/Asu_Ath.r40000
```

Outputs:

- `*.pbad.8frame.bedGraph`
- `*.pbad.8frame.stat.pdf`

Supported `--metric` values are `pbad`, `log`, `stripe`, `pearsone`, and
`spearman`.

### 7. Reconstruct observed/target matrices

```bash
coolsecture lift2matrix \
  --liftover step2/Asu_Ath/Asu_Ath.r40000.Merged.liftContacts \
  --fadix step0/Asu.fa.fai \
  --format cool \
  --out-prefix step3/Asu_Ath.r40000
```

Outputs:

- `*.Observed.cool`
- `*.Target.cool`

Use `--format hic` or `--format both` if `juicer_tools` is available.

### 8. Compute matrix similarity

```bash
coolsecture cross-validate \
  --matrix-a step3/Asu_Ath.r40000.Observed.cool \
  --matrix-b step3/Asu_Ath.r40000.Target.cool \
  --max-dist-mb 10 \
  --format pdf \
  --out-prefix step3/Asu_Ath.r40000
```

Outputs:

- `*.scc.tsv`
- `*.scc.summary.tsv`
- `*.scc.pdf`

### 9. Plot a cross-species locus

```bash
coolsecture plot-cross \
  --liftover step2/Asu_Ath/Asu_Ath.r40000.Merged.liftContacts \
  --fadix step0/Asu.fa.fai \
  --locus chr1:0-10000000 \
  --heat obs-tgt \
  --format pdf \
  --out-prefix step3/Asu_Ath.r40000
```

## Snakemake Examples

The repository includes two workflow templates:

- `example1/`: plant example using `.mcool` matrices and a `.link` file.
- `example2/`: mammalian example using `.hic` matrices and a UCSC `.chain` file.

Each example is configured through `config.yaml` and run with:

```bash
cd example1
snakemake -n -s Snakefile --cores 1
snakemake -s Snakefile --cores 8
```

Before running, edit `config.yaml` so that the matrix, synteny, and `.fai` paths
point to files available on your machine. Large Hi-C matrices and generated
`step1/`, `step2/`, and `step3/` outputs are not intended to be committed to the
Git repository.

## Interactive Outputs

Several commands support optional Plotly HTML summaries:

- `prepare --summary --interactive auto|on|off`
- `liftcontracts --interactive auto|on|off`
- `multiscale --interactive auto|on|off`

Install Plotly with:

```bash
python -m pip install -e ".[viz]"
```

The Snakemake examples are designed around PDF outputs and may remove Plotly HTML
artifacts to keep workflow outputs predictable.

## Repository Hygiene

Recommended files to keep in the GitHub repository:

- `src/`
- `pyproject.toml`
- `README.md`
- `LICENSE`
- lightweight example `Snakefile` and `config.yaml` files
- small synthetic or metadata-only example inputs, if needed

Recommended files to exclude from Git:

- `.snakemake/`
- `.codex_tmp/`
- `__pycache__/`
- `src/*.egg-info/`
- `step0/`, `step1/`, `step2/`, `step3/`
- large `.hic`, `.cool`, `.mcool`, `.tsv`, `.bedGraph`, and generated figure files
- local backup or benchmark directories such as `bak/`

For reproducible public releases, put large data on Zenodo, Figshare, SRA/GEO, or
another data repository, then link to it from this README or from a separate data
availability document.

## Citation

If you use Coolsecture in your research, please cite this repository:

```text
Coolsecture: an easy-to-use framework for cross-species Hi-C contact map comparison.
```
