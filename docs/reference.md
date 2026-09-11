# Reference

## Snakemake examples

Two workflow templates are included:

- `example1/` — plant example using `.mcool` + `.link`
- `example2/` — mammalian example using `.hic` + UCSC `.chain`

Each uses `config.yaml`:

```bash
cd example1
snakemake -n -s Snakefile --cores 1   # dry-run
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

The example Snakemake workflows target PDF outputs and may remove Plotly HTML to keep outputs predictable. See [VISUALIZATIONS.md](VISUALIZATIONS.md).

## Releases, CI, and tests

- Software version: **0.4.1** (see `pyproject.toml`); released tags are published on the GitHub repository.
- CI (`.github/workflows/ci.yml`) runs on each push/PR, installs the package on Python 3.10, runs `python -m compileall src`, and checks `python -m coolsecture -h`.
- There is no formal test suite checked in yet; the Snakemake examples (`example1/`, `example2/`) serve as end-to-end integration runs.
