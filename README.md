# Coolsecture

Coolsecture compares Hi-C contact maps across species. Given two genome
assemblies, a synteny map between them, and a Hi-C matrix for each, it lifts
contacts from one species' coordinate system into the other's and reports how
well they agree.

It builds on [C-InterSecture](https://github.com/NuriddinovMA/C-InterSecture)
but is rebuilt around a CLI workflow, native `.cool`/`.mcool`/`.hic` input,
multi-resolution runs, and a disk-spill path for matrices too large to fit
in memory.

## Changelog

### v0.3.5 - 2026-07-06

- `asm2link` can now use mummer4 (`-a mummer4`; nucmer + delta-filter +
  show-coords) in addition to minimap2. minimap2 stays the default.
- New `--mummer-filter {1-to-1,mutual-best,none}` (default `1-to-1`) and
  `--mummer-min-idy` / `--mummer-min-len` for filtering mummer4 alignments.
- `run-all` gained `--asm-aligner` and `--asm-mummer-filter` to forward.
- Help text fixes: `prepare` prog name, `liftcontacts --contact-a` extension,
  missing descriptions for `--dups-filter` / `--model`, duplicated
  `(default: auto)` on several `--interactive` flags.

### v0.3.2 - 2026-06-14

- Set publication-ready static plots to use editable PDF/SVG text with Carlito-preferred fonts.
- Added multi-resolution `.hic` support in `prepare` and `run-all`.
- Added chromosome-name mapping output when liftover matrix generation uses aliases.
- Added `run-all --auto` parameter selection with `auto_params.tsv` reporting.
- Added automatic `plot-cross` region selection for top differential/conserved 2 Mb PBAD-ranked regions.

## Main Features

- Align assemblies into syntenic links with `minimap2` or `mummer4`.
- Convert `.link` or UCSC `.chain` into Coolsecture's `.mark` synteny format.
- Turn `.cool` / `.mcool` / `.hic` matrices into distance-stratified
  percentile-ranked contact tables.
- Lift contacts A→B and B→A, then summarize reciprocal consistency.
- Reconstruct observed/target `.cool` or `.hic` matrices from lifted contacts.
- PBAD and related metrics, diagnostic plots, split-triangle cross plots,
  HiCRep-style SCC.
- Run command-by-command, through `run-all`, or via the example Snakemake
  workflows.

## Installation

Needs Python 3.8+.

```bash
git clone https://github.com/pk-zhu/Coolsecture.git
cd Coolsecture
python -m pip install -e .
```

Optional extras (`.hic` reading, Plotly HTML, SCC stats):

```bash
python -m pip install -e ".[hic]"
python -m pip install -e ".[viz]"
python -m pip install -e ".[stats]"
```

External tools:

- `minimap2` — used by `asm2link` and `run-all` (default aligner).
- `mummer4` (`nucmer`, `delta-filter`, `show-coords`) — only if you pass
  `-a mummer4` / `--asm-aligner mummer4`.
- `samtools` — optional; `run-all` can write a minimal `.fai` itself if
  `samtools faidx` is missing.
- `juicer_tools` — only for `lift2matrix --format hic` or `--format both`.
- `snakemake` — only for the example workflows.

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

You need:

- A FASTA and `.fai` for each species (A and B).
- A Hi-C matrix for each, in `.cool`, `.mcool`, or `.hic`.
- A synteny file: a six-column `.link` (from `asm2link`) or a UCSC `.chain`.

`.link` format (0-based half-open, same as PAF):

```text
chromA  startA  endA  chromB  startB  endB
```

For `.cool` / `.mcool`, the bins table must carry at least one normalization
vector among `KR`, `VC_SQRT`, `VC`, or `weight`.

## Quick Start: End-to-End

`run-all` chains the full pipeline — alignment, mark generation, contact
preparation, bidirectional liftover, statistics, metrics, and matrix
reconstruction. It does not run `plot-cross`.

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

minimap2 (default):

```bash
coolsecture asm2link \
  --genome-a Asu.fa \
  --genome-b Ath.fa \
  -x asm10 \
  --out-prefix step0/Asu_Ath
```

mummer4 (more precise on divergent assemblies; emits 1-to-1 syntenic
alignments by default):

```bash
coolsecture asm2link \
  --genome-a Asu.fa \
  --genome-b Ath.fa \
  -a mummer4 \
  --mummer-filter 1-to-1 \
  --out-prefix step0/Asu_Ath
```

minimap2 produces:

- `step0/Asu_Ath.paf`
- `step0/Asu_Ath.link`

mummer4 produces:

- `step0/Asu_Ath.delta` — raw nucmer output
- `step0/Asu_Ath.filter.delta` — after `delta-filter` (skipped when
  `--mummer-filter none`)
- `step0/Asu_Ath.coords.tsv` — `show-coords -T -H` tabular output
- `step0/Asu_Ath.link`

If you already have a UCSC `.chain`, skip this step and feed it to `link2mark`.

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

Two workflow templates ship with the repo:

- `example1/` — plant example, `.mcool` + `.link`.
- `example2/` — mammalian example, `.hic` + UCSC `.chain`.

Each is driven by `config.yaml`:

```bash
cd example1
snakemake -n -s Snakefile --cores 1   # dry-run
snakemake -s Snakefile --cores 8
```

Edit `config.yaml` so the matrix, synteny, and `.fai` paths point at files on
your machine. Don't commit large matrices or generated `step1..3/` outputs —
see `.gitignore`.

## Interactive Outputs

A few commands can emit Plotly HTML alongside the static plots:

- `prepare --summary --interactive auto|on|off`
- `liftcontracts --interactive auto|on|off`
- `multiscale --interactive auto|on|off`

```bash
python -m pip install -e ".[viz]"
```

The Snakemake examples target PDF outputs and may delete Plotly HTML to keep
workflow outputs predictable.

## Citation

If you use Coolsecture in your research, please cite this repository:

```text
Coolsecture: an easy-to-use framework for cross-species Hi-C contact map comparison.
```
