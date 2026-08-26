# EpiCast: leveraging virtual epigenomic features to predict episomal regulatory activity across cell types

## Introduction

Designing regulatory sequences for synthetic biology and gene therapy requires understanding how DNA sequences drive cell-type-specific expression, yet existing MPRA-based models typically fail to generalize beyond the cell types used for training.

To address this challenge, we present **EpiCast**, a deep learning framework for predicting episomal cis-regulatory element (CRE) activity across diverse human cell types. We integrate DNA sequence with **virtual epigenomic features (VEFs)**: cell-type-specific regulatory proxies inferred from large-scale genomic sequence-to-function models such as Sei and AlphaGenome. Although episomal DNA lacks native chromatin structure, these model-derived features capture how different cell types are predicted to interpret a given sequence, enabling EpiCast to incorporate contextual regulatory information without requiring MPRA data from every cell type.

Trained on MPRA datasets, EpiCast learns both sequence grammar and cell-type-dependent regulatory logic. As a result, it achieves strong performance within training cell types and generalizes to previously unseen ones, which we evaluate both on held-out cell types of the training MPRA and zero-shot on an independent MPRA.

## Repository layout

```
src/epicast/       model, dataset and training library (installable package)
scripts/           command-line entry points for training and inference
paper/             everything needed to reproduce the figures of the paper
  config.py        single source of paths, cell types, model registry and colours
  utils.py         helpers shared by the analysis scripts
  analysis/        numbered pipeline, from raw MPRA data to metric tables
  plot/            one script per figure panel
data/              model track metadata; all other data is distributed separately
```

`paper/README.md` documents the pipeline and the figure-to-script mapping. `data/README.md` lists what has to be downloaded and where it goes.

## Installation

```bash
git clone https://github.com/maplecai/EpiCast.git
cd EpiCast
conda create -n epicast python=3.10
conda activate epicast
pip install -r requirements.txt
pip install -e .
```

After installation the library is importable from any working directory:

```python
import epicast
```

## Data

The MPRA datasets, the derived VEF matrices and the trained checkpoints are deposited at
<https://zenodo.org/records/17669741>. Sei model weights are available at
<https://zenodo.org/records/4906997>. See `data/README.md` for the expected directory
layout.

## Reproducing the paper

```bash
python paper/analysis/01_prepare_gosai_data.py   # and the remaining numbered scripts
python paper/plot/fig1c_vef_activity_correlation.py
```

Every script resolves its paths through `paper/config.py`, takes no positional arguments,
and can be run from either the repository root or `paper/`. The full order, the two steps
that need a GPU and the figure-to-script mapping are in `paper/README.md`.

## Citation

Under review (RECOMB 2026).

## License

MIT License.

## Contact

maplecai142857@gmail.com
