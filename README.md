# Coolsecture

Coolsecture compares Hi-C contact maps across species. Given two genome
assemblies, a synteny map between them, and a Hi-C matrix for each, it lifts
contacts from one species' coordinate system into the other's and reports how
well they agree.

It builds on [C-InterSecture](https://github.com/NuriddinovMA/C-InterSecture)
but is rebuilt around a CLI workflow, native `.cool`/`.mcool`/`.hic` input, and
multi-resolution runs. The memory-heavy contact-liftover and matrix-reconstruction
stages (`liftcontacts`, `lift2matrix`) can spill intermediate tables to disk via
`--spill-threshold-mb`/`--tmp-dir`; other stages still load their inputs normally,
so a matrix that does not fit in RAM cannot be processed by every command.

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
  HiCRep-inspired SCC-like similarity score.
- Run command-by-command, through `run-all`, or via the example Snakemake
  workflows.

## Installation

Needs Python 3.8+.

```bash
git clone https://github.com/pk-zhu/Coolsecture.git
cd Coolsecture
python -m pip install -e .
```

Optional extras (`.hic` reading, Plotly HTML, Spearman metric + KDE plotting):

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
| `liftcontacts` | Run A->B and B->A contact liftover and reciprocal summaries. (`liftcontracts` is a deprecated alias.) |
| `contact-stat` | Plot percentile, distance, and ratio diagnostics from lifted contacts. |
| `metric` | Compute PBAD and related metrics as bedGraph plus figures. |
| `lift2matrix` | Convert lifted contacts to observed/target `.cool` or `.hic` matrices. |
| `plot-cross` | Draw split-triangle cross-species heatmaps for a locus. |
| `multiscale` | Summarize PBAD stability across multiple resolutions. |
| `similarity` | Compute stratum-adjusted correlation (HiCRep-inspired SCC-like similarity score) between matched matrices. |
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

`run-all` chains the full pipeline, including alignment, mark generation, contact preparation, bidirectional lift-over, diagnostic statistics, metrics, matrix reconstruction, similarity scoring, and automatic regional plot-cross visualization.

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
  --liftcontacts-args "--model balanced --dups-filter coverage --nthreads 8" \
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
- `liftcontacts --interactive auto|on|off`
- `multiscale --interactive auto|on|off`

```bash
python -m pip install -e ".[viz]"
```

The Snakemake examples target PDF outputs and may delete Plotly HTML to keep
workflow outputs predictable.

## File formats

### `.link` (synteny, 6 columns; 0-based half-open)

```text
chromA  startA  endA  chromB  startB  endB
```

Columns 1-3 are always ascending on genome A. Columns 4-6 give the aligned
interval on genome B: for a forward alignment `startB < endB`; for a reverse
(strand `-`) alignment the B coordinates are emitted **swapped**, so
`startB > endB` encodes the orientation. There is no separate strand column —
the sign of `endB - startB` is the direction. (`asm2link` writes both the
minimap2 PAF path and the mummer4 path this way; UCSC `.chain` carries strand
explicitly and is converted by `link2mark`.)

### `.mark` (densified synteny map, 8 columns)

```text
chromA  startA  endA  chromB  startB  endB  direction  block_id
```

Produced by `link2mark` from `.link`/`.chain`. Long collinear blocks are
densified to `--step-len` spacing (default 150 bp segments, blocks shorter than
`--thr-len` 300 bp kept whole). `direction` is `+1`/`-1` (orientation of the B
interval); `-1` segments walk B backwards. `block_id` is the source block, with
`<id>_gap` rows marking the spanned gap between adjacent blocks.

### `.contacts.tsv` (prepared contacts, 14 columns)

Written by `prepare` (one row per retained contact pixel):

```text
chrom1 start1 end1 bin1  chrom2 start2 end2 bin2  rank strict weak  cov1 cov2  dist_bins
```

- `rank`/`strict`/`weak` — percentile rank of the contact's normalized signal
  within its distance stratum, on a **0-99** integer scale (`rank` is the
  midpoint rank, `strict`/`weak` bracket it); 99 = strongest.
- `cov1`/`cov2` — per-bin Hi-C read coverage (total normalized signal incident
  on each bin); higher = better sampled.
- `dist_bins` — genomic distance in bins; `-1` marks inter-chromosomal contacts.

The companion `.stats.tsv` holds, per distance bin, `n`, `p05/p50/p95`, and 100
tab-separated percentile quantiles (`p001`..`p100`).

### `.liftContacts` (lifted contacts, 16 columns)

```text
chr1_observed pos1_observed chr2_observed pos2_observed
remap1_target remap2_target
observed_contacts target_contacts
observed_deviations target_deviations
observed_coverages_pos1 observed_coverages_pos2
target_coverages_pos1 target_coverages_pos2
target_contact_distances remapping_coverages
```

- `observed_*` are in the source genome; `target_*`/`remap*` are the liftover
  onto the other genome.
- `observed_contacts`/`target_contacts` are the 0-99 percentile ranks of the
  source and target contacts.
- `*_deviations` are coordinate spread (placement uncertainty) in bins.
- `target_contact_distances` is the target-genome span in bins (`-1` =
  inter-chromosomal).
- `remapping_coverages` is the summed synteny weight (a 1:1 map is ~1.0; a
  1-to-many map is diluted toward 0).

## Normalization and the `balanced` model

Two normalizations happen at **different stages** — do not confuse them:

1. **Cooler matrix balancing** — in `prepare`. Pixel counts are balanced before
   percentile ranking. Two weight conventions coexist and are handled
   differently (hic2cool: *"cooler uses multiplicative weights and hic uses
   divisive weights"*):
   - `KR` / `VC_SQRT` / `VC` (Juicer `.hic`, hic2cool ≥ 0.5, 4DN) are **divisive**:
     `val = count / (w1 * w2)`.
   - `weight` (the standard `cooler balance` vector) is **multiplicative**:
     `val = count * w1 * w2`; Coolsecture internally inverts it before the
     division.

   The weight column is picked by preference `KR > VC_SQRT > VC > weight`. Files
   with no supported normalization vector cause `prepare` to raise an error
   (there is no raw-count fallback).
2. **Liftover `balanced` model** — in `liftcontacts`. When a source contact
   maps through several synteny paths, the aggregated target quantities are
   divided by the summed remapping weight `c[-3]` (= the
   `remapping_coverages` column) to normalize for liftover ambiguity. `raw`
   skips this division. **No Cooler weights are applied here** — they were
   already consumed in `prepare`.

### Invalid / NaN weights

In `prepare`, any non-finite or non-positive balancing weight
(`NaN`, `Inf`, `w <= 0`, i.e. the "bad bins" a balancing pass marks) is replaced
with `1.0`, so that `count / (w1*w2)` for those bins falls back to the raw count
rather than exploding. Bins that are unalignable / gap-covered can additionally
be masked upstream. If the `.cool`/`.mcool` carries **no** weight column among
`KR/VC_SQRT/VC/weight`, `prepare` raises a `RuntimeError` instead of silently
using raw counts.

## Metrics

All scores compare the source percentile rank against the target rank for
lifted contacts; the randomized null shuffles the target percentile rank across
contacts (no cross-species correlation).

- **P-BAD** (percentile-based Bhattacharyya-like divergence), evaluated in a
  `±frame` bin window around each locus. For contact pair with source rank
  `p1` and target rank `p2`:

  ```text
  dp      = |p1 - p2| / 100
  ds(p)   = clamp(1 - |p - 50|/50, 0.01, 0.99)        # rank confidence
  P-BAD   = mean over window pairs of [ -dp * log10(ds(p1) * ds(p2)) ]
  ```

  The window is scored only when it contains more than `frame²` contacts.
  Higher P-BAD = stronger divergence. `metric --metric` also offers `log`,
  `stripe`, `pearsone`, `spearman` (the last needs the `[stats]` extra).

- **SCC-like** (`similarity`) — a HiCRep-inspired stratum-adjusted
  correlation. Matrices are split into genomic-distance strata; a Pearson
  correlation is computed per stratum and aggregated as a weighted mean with
  stratum pixel counts as weights:

  ```text
  SCC-like = Σ_d (n_d · r_d) / Σ_d n_d
  ```

  This is a **simplified** SCC-like statistic: it does not apply HiCRep's 2-D
  stratum smoothing or inverse-variance weights. Outputs are `.scc-like.tsv`
  (per-stratum `r`, `n`), `.scc-like.summary.tsv`, and `.scc-like.<fmt>`.

- **Multiscale stability** (`multiscale`) — P-BAD is recomputed across
  resolutions; for per-resolution mean P-BAD `m_r`:

  ```text
  stability = 1 - std(m_r) / (|mean(m_r)| + 1e-9)
  ```

  near 1 = stable across resolutions. It also reports the fraction of contacts
  with P-BAD above `--pbad-thr` and coarse/fine conservation flags.

## Reproducing the randomized figures

The observed-vs-randomized diagnostics shuffle the target percentile ranks.
Pass an explicit RNG seed to make a run reproducible:

```bash
coolsecture contact-stat ... --seed 20260329
```

(`contact-stat` exposes `--seed`; the null uses `np.random.seed`. The `metric`
observed-vs-random panel uses the same shuffle but currently has no `--seed`
flag, so its null varies run to run.)

## Writing `.hic` output with Juicer Tools

`lift2matrix --format cool` needs no external tool. `--format hic` (or `both`)
shells out to a `juicer_tools` executable on `PATH`:

```bash
# obtain Juicer Tools (requires Java)
wget https://s3.amazonaws.com/hicfiles.tc4ga.com/public/juicer/juicer_tools_1.22.01.jar
echo 'exec java -jar /path/to/juicer_tools_1.22.01.jar "$@"' > juicer_tools
chmod +x juicer_tools && export PATH="$PWD:$PATH"

coolsecture lift2matrix --liftover x.Merged.liftContacts --fadix a.fa.fai \
    --format hic --out-prefix step3/x
```

Coolsecture writes a temporary `<chrom> <size>` `chrom.sizes` and a
`chrom pos chrom pos value` contact list, then runs
`juicer_tools pre -r <resolution> <in.txt> <out.hic> <chrom.sizes>`. Missing
`juicer_tools` now exits non-zero with an explanatory error instead of failing
silently.

## Releases, CI, and tests

- Software version: see `pyproject.toml` (currently **0.3.5**); released tags
  are published on the GitHub repository.
- CI (`.github/workflows/ci.yml`) runs on every push/PR: installs the package
  on Python 3.10, runs `python -m compileall src`, and checks
  `python -m coolsecture -h`.
- There is no formal test suite checked in yet; the Snakemake examples
  (`example1/`, `example2/`) double as end-to-end integration runs.

## Citation

If you use Coolsecture in your research, please cite the paper and the software
release. Fill in the bibliographic details from the published version:

```text
# Peer-reviewed paper (TODO: confirm authors / title / journal / year / DOI)
<Authors>. Coolsecture: <full paper title>. <Journal> (<Year>). doi: <DOI>

# Software release (version + archive DOI)
Coolsecture v0.3.5, <Authors>. Zenodo/Figshare archive, doi: <archive DOI>
```

The repository itself may also be cited as:

```text
Coolsecture: an easy-to-use framework for cross-species Hi-C contact map comparison.
https://github.com/pk-zhu/Coolsecture
```
