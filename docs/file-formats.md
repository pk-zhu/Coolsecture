# File formats

## `.link`

Six columns, 0-based half-open:

```text
chromA  startA  endA  chromB  startB  endB
```

Columns 1–3 are ascending on genome A. For reverse alignments, genome B coordinates are written in reverse order (`startB > endB`), so orientation is encoded by the sign of `endB - startB`. There is no separate strand column.

## `.mark`

Eight columns:

```text
chromA  startA  endA  chromB  startB  endB  direction  block_id
```

`link2mark` generates `.mark` from `.link` or `.chain`. Long collinear blocks are densified at `--step-len` spacing (default 150 bp); blocks shorter than `--thr-len` (default 300 bp) are kept intact. `direction` is `+1` or `-1`. Rows with `<id>_gap` mark gaps between adjacent source blocks.

## `.contacts.tsv`

Fourteen columns:

```text
chrom1 start1 end1 bin1  chrom2 start2 end2 bin2  rank strict weak  cov1 cov2  dist_bins
```

- `rank`, `strict`, `weak` — normalized contact percentile ranks within each distance stratum, on a 0–99 integer scale.
- `cov1`, `cov2` — total raw Hi-C read coverage incident on each bin.
- `dist_bins` — genomic distance in bins; `-1` indicates inter-chromosomal contacts.

The companion `.stats.tsv` reports `n`, `p05`, `p50`, `p95`, and percentile quantiles `p001`–`p100` for each distance bin.

### Normalization

Before ranking, counts are normalized using the first available weight vector in this order:

```text
KR > VC_SQRT > VC > weight
```

`KR`/`VC`-type vectors are divisive (`count / (w1*w2)`); the Cooler `weight` column is multiplicative (`count * w1 * w2`). Contacts touching bins with missing or non-positive weights are removed. Matrices without a supported weight column are rejected.

## `.liftContacts`

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
