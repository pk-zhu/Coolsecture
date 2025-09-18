# Coolsecture: An Easy-to-Use and Improved Framework for Cross-Species Hi-C Contact Map Comparison

Coolsecture provides a modernized, user-friendly framework for comparative Hi-C analysis across species. By adopting standardized inputs and outputs, improving the statistical treatment of contact scores, implementing vectorized and parallel algorithms, and supplying comprehensive diagnostic visualizations, Coolsecture addresses the limitations of the original C-InterSecture pipeline.

---

## Requirements

Coolsecture requires **Python 3.8+** and the following dependencies:

- [minimap2](https://github.com/lh3/minimap2)  
- numpy  
- pandas  
- matplotlib  
- cooler  
- h5py  

---

## Installation

Clone this repository and install in editable mode:

```bash
git clone https://github.com/pk-zhu/Coolsecture.git && cd Coolsecture
pip install -e .
```

---

## Usage

To view the help message:

```bash
coolsecture -h
coolsecture <subcommand> -h
```

---

## Example Workflow

1. Navigate to the example directory:

```bash
cd example
```

2. Download example data and place it in the cools directory. For instance, obtain the .mcool files from the [Figureshare](https://github.com/pk-zhu/Coolsecture) and put them under cools:

```bash
tree .
example/
├── config.yaml
├── cools
│   ├── Asu.mcool
│   └── Ath.mcool
├── Snakefile
└── step0
    ├── Asu_Ath.link
    ├── Asu.fa.fai
    └── Ath.fa.fai
```

3. Run a dry-run to preview the workflow and Execute the full workflow

```bash
snakemake -np
snakemake -j 8
```

---

## Citation

If you use Coolsecture in your research, please cite this repository:
```
Coolsecture: An easy-to-use and improved framework for cross-species Hi-C contact map comparison.
```
