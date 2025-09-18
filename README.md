# Coolsecture: An Easy-to-Use and Improved Framework for Cross-Species Hi-C Contact Map Comparison

Coolsecture provides a modernized, user-friendly framework for comparative Hi-C analysis across species. By adopting standardized inputs and outputs, improving the statistical treatment of contact scores, implementing vectorized and parallel algorithms, and supplying comprehensive diagnostic visualizations, Coolsecture addresses the limitations of the original C InterSecture pipeline.

---

## Features
- **Cross-species Hi-C analysis** with standardized inputs and outputs  
- **Improved statistical treatment** of contact scores  
- **Efficient algorithms** with vectorized and parallel implementations  
- **Comprehensive diagnostic visualizations** to aid interpretation  
- **Easy-to-use interface** that streamlines comparative genomics workflows  

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
## Usage

To view the help message:

```bash
coolsecture -h
```
