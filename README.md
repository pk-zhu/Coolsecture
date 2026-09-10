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

`run-all` performs assembly alignment, mark generation, contact preparation, bidirectional liftover, diagnostics, metric calculation, matrix reconstruction, similarity scoring, and automatic `plot-cross` region selection.

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

## Step-by-step workflow

### 1. Build a link file from assemblies

minimap2 (default):

```bash
coolsecture asm2link \
  --genome-a Asu.fa \
  --genome-b Ath.fa \
  -x asm10 \
  --out-prefix step0/Asu_Ath
```

mummer4:

```bash
coolsecture asm2link \
  --genome-a Asu.fa \
  --genome-b Ath.fa \
  -a mummer4 \
  --mummer-filter 1-to-1 \
  --out-prefix step0/Asu_Ath
```

minimap2 outputs:

- `step0/Asu_Ath.paf`
- `step0/Asu_Ath.link`

mummer4 outputs:

- `step0/Asu_Ath.delta`
- `step0/Asu_Ath.filter.delta` (omitted with `--mummer-filter none`)
- `step0/Asu_Ath.coords.tsv`
- `step0/Asu_Ath.link`

If you already have a UCSC `.chain`, skip this step.

### 2. Convert synteny to `.mark`

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
- `*.r<resolution>.contacts.tsv` and `*.r<resolution>.stats.tsv` in multi-resolution mode
- `*.multi_resolution.summary.tsv` and `.pdf` with `--summary`

Before ranking, counts are normalized using the first available vector in this order:

`KR > VC_SQRT > VC > weight`

`KR`/`VC`-type vectors are divisive; Cooler `weight` is multiplicative. Contacts touching bins with missing or non-positive weights are removed. Matrices without a supported weight column are rejected.

### 4. Run bidirectional contact liftover

```bash
coolsecture liftcontacts \
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

For larger files:

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

`--model balanced` divides aggregated values by summed synteny remapping weight. `--model raw` skips this adjustment. When one source contact maps to multiple target positions, `--dups-filter` controls which mapping is retained; discarded mappings are written to `.discarded_dups.tsv`.

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

The randomized null is generated by shuffling target percentile ranks. Use `--seed <int>` for reproducibility.

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

Supported metrics: `pbad`, `log`, `stripe`, `pearsone`, and `spearman`. `spearman` requires the `[stats]` extra.

P-BAD is a windowed divergence between source and target percentile ranks. The PDF compares the observed distribution with the randomized null.

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

Use `--format hic` or `--format both` to also write `.hic`. This requires Juicer Tools on `PATH`, or:

```bash
--juicer-tools "java -jar /path/to/juicer_tools.jar"
```

Matrix values are 0–99 percentile ranks in source coordinates.

### 8. Compute matrix similarity

```bash
coolsecture similarity \
  --matrix-a step3/Asu_Ath.r40000.Observed.cool \
  --matrix-b step3/Asu_Ath.r40000.Target.cool \
  --max-dist-mb 10 \
  --format pdf \
  --out-prefix step3/Asu_Ath.r40000
```

Outputs:

- `*.scc-like.tsv`
- `*.scc-like.summary.tsv`
- `*.scc-like.pdf`

The score is a Pearson correlation computed within genomic-distance strata and averaged using stratum pixel counts as weights. By default (`--smooth-h 0`) there is no 2-D smoothing, so the result is reported as **SCC-like** rather than SCC. Pass `--smooth-h h` to enable HiCRep-style 2-D stratum smoothing over a `(2h+1)×(2h+1)` bin window before the per-stratum correlations; the window mean uses only pixels observed in each matrix, so unmapped positions are not zero-filled. The score depends on h, which is recorded in the summary. `--smooth-h auto` scans h (up to `--auto-h-max`, default 10) and selects the smallest h at which SCC reaches a plateau (`|SCC(h)−SCC(h−1)| < --auto-h-tol`, default 0.01); the full SCC(h) curve is written to `*.scc-h.tsv` and the chosen h to the summary.

`multiscale` separately summarizes divergence stability across resolutions; `--metric` accepts `pbad` (default), `log`, `stripe`, `pearsone`, or `spearman`. Correlation metrics are converted to `1 - r` and log-ratio metrics to `|log ratio|`, so high values always mean divergence.

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

## Snakemake examples

Two workflow templates are included:

- `example1/` — plant example using `.mcool` + `.link`
- `example2/` — mammalian example using `.hic` + UCSC `.chain`

Each uses `config.yaml`:

```bash
cd example1
snakemake -n -s Snakefile --cores 1
snakemake -s Snakefile --cores 8
```

Edit `config.yaml` to point to your local matrix, synteny, and `.fai` files. Large matrices and generated `step1..3/` outputs should not be committed; see `.gitignore`.

## Interactive outputs

The following commands can emit Plotly HTML in addition to static plots:

- `prepare --summary --interactive auto|on|off`
- `liftcontacts --interactive auto|on|off`
- `multiscale --interactive auto|on|off`

Install the visualization extra first:

```bash
python -m pip install -e ".[viz]"
```

The example Snakemake workflows target PDF outputs and may remove Plotly HTML to keep outputs predictable.

## File formats

### `.link`

Six columns, 0-based half-open:

```text
chromA  startA  endA  chromB  startB  endB
```

Columns 1–3 are ascending on genome A. For reverse alignments, genome B coordinates are written in reverse order (`startB > endB`), so orientation is encoded by the sign of `endB - startB`. There is no separate strand column.

### `.mark`

Eight columns:

```text
chromA  startA  endA  chromB  startB  endB  direction  block_id
```

`link2mark` generates `.mark` from `.link` or `.chain`. Long collinear blocks are densified at `--step-len` spacing (default 150 bp); blocks shorter than `--thr-len` (default 300 bp) are kept intact. `direction` is `+1` or `-1`. Rows with `<id>_gap` mark gaps between adjacent source blocks.

### `.contacts.tsv`

Fourteen columns:

```text
chrom1 start1 end1 bin1  chrom2 start2 end2 bin2  rank strict weak  cov1 cov2  dist_bins
```

- `rank`, `strict`, `weak` — normalized contact percentile ranks within each distance stratum, on a 0–99 integer scale.
- `cov1`, `cov2` — total raw Hi-C read coverage incident on each bin.
- `dist_bins` — genomic distance in bins; `-1` indicates inter-chromosomal contacts.

The companion `.stats.tsv` reports `n`, `p05`, `p50`, `p95`, and percentile quantiles `p001`–`p100` for each distance bin.

### `.liftContacts`

Sixteen columns:

```text
chr1_observed pos1_observed chr2_observed pos2_observed
remap1_target remap2_target
observed_contacts target_contacts
observed_deviations target_deviations
observed_coverages_pos1 observed_coverages_pos2
target_coverages_pos1 target_coverages_pos2
target_contact_distances remapping_coverages
```

- `observed_*` — source-genome coordinates and values.
- `target_*` / `remap*` — corresponding values after liftover.
- `observed_contacts`, `target_contacts` — 0–99 percentile ranks.
- `observed_deviations`, `target_deviations` — percentile-rank uncertainty, calculated as `max(weak-rank, rank-strict)`.
- `target_contact_distances` — target-genome span in bins; `-1` indicates inter-chromosomal contacts.
- `remapping_coverages` — summed synteny weight; a one-to-one mapping is approximately 1.0.

## Releases, CI, and tests

- Current software version: **0.3.5** (see `pyproject.toml`).
- CI (`.github/workflows/ci.yml`) runs on each push/PR, installs the package on Python 3.10, runs `python -m compileall src`, and checks `python -m coolsecture -h`.
- No formal test suite is included yet. The Snakemake examples (`example1/`, `example2/`) serve as end-to-end integration runs.

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
