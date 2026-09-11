# Visualization Implementation Notes

This project generates interactive Plotly HTML visualizations for the majority of its summary tables. The goal is to keep TSV outputs as the source of truth while providing lightweight, responsive charts for quick inspection and reporting. The example Snakemake pipeline enforces PDF-only outputs and removes HTML artifacts after each step.

## What Gets Visualized

- `prepare --summary`
  - `*.multi_resolution.summary.tsv` -> `*.multi_resolution.summary.plotly.html`
- `bidirectional`
  - `*.bidirectional.summary.tsv` -> `*.bidirectional.summary.plotly.html`
  - `*.bidirectional.tags.tsv` -> `*.bidirectional.tags.plotly.html`
- `multiscale` (per `--metric` `<tag>`: pbad/log/stripe/pearsone/spearman)
  - `*.multiscale.<tag>.tsv` -> `*.multiscale.<tag>.plotly.html`
  - `*.multiscale.<tag>.summary.tsv` -> `*.multiscale.<tag>.summary.plotly.html`
- `similarity` emits only static `.scc-like.<fmt>` plots (matplotlib); it has no Plotly HTML output.

## Implementation Pattern

1. Each command writes its canonical TSV output first.
2. If Plotly is available and `--interactive` is `auto` or `on`, the command creates a Plotly figure and writes a responsive HTML file using the same output prefix.
3. HTML files are self-contained and viewable in any modern browser.

## Updating for New Data

- Re-run the command that generates the TSV file. The HTML visualization is regenerated automatically (unless `--interactive off` is used).
- Keep the same `--out-prefix` to overwrite the existing HTML, or change it to keep a historical snapshot.
- For batch workflows (Snakemake), the example pipeline keeps `--interactive auto` for compatibility but **deletes** the `.plotly.html` files to guarantee PDF-only outputs.

## Dependency

Install Plotly via:

```bash
pip install -e .[viz]
```

If Plotly is not installed, `--interactive auto` will skip HTML generation, while `--interactive on` will raise a clear error.
