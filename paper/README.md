# paper/ — reproducing the figures

Two stages: `analysis/` turns data into metric tables under `results/`, `plot/` turns those
tables into one PDF per figure panel under `figures/`. Both stages read all of their paths,
cell-type names, model registry and colours from `config.py`, so no script takes positional
arguments and every script runs from either the repository root or from `paper/`.

```bash
conda activate epicast
python paper/analysis/07_eval_regression.py
python paper/plot/fig2bc_activity_metrics.py
```

Only `06_infer_trained_model.py` and `10_predict_castillo_mpra.sh` need a GPU; everything
else is CPU only. `results/` and `figures/` are created on demand and are not version
controlled.

## Pipeline

The numbering follows the dependency order, but the pipeline is not strictly serial.

```
01_prepare_gosai_data ──────────────────────────────────┐
01_parse_model_track_metadata                           │
                                                        │
02_extract_sei_vef ─┐                                   │
02_extract_ag_vef ──┼─→ 03_normalize_vef ───────────────┤  VEF matrices
02_extract_castillo_{ag,sei}_vef ─┘                     │
                                                        ├─→ 11_vef_partial_correlation
                                                        ├─→ 11_vef_pairwise_correlation
                                                        ▼
                                     04_vef_activity_specificity   (prints only)
                                     05_train_vef_only_models
                                     06_infer_trained_model        [GPU]
                                     10_predict_castillo_mpra.sh   [GPU]
                                                        │
        ┌───────────────────────────┬───────────────────┴──┐
        ▼                           ▼                      ▼
07_eval_regression        08_eval_classification     09_eval_retrieval
        └───────────────────────────┴──────────────────────┘
                                    ▼
                        14_export_prediction_tables  →  results/predictions/
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
     15_export_figure_metrics                    12_eval_castillo
        → results/figure_metrics/                  → results/castillo/
```

Steps in one line each:

| Step | What it does |
|---|---|
| `01_prepare_gosai_data` | Malinois-style preprocessing of the Gosai MPRA into a 760,679-row label table; z-scores are estimated on training chromosomes only |
| `01_parse_model_track_metadata` | parses the Sei, Enformer, Borzoi and AlphaGenome track metadata onto one naming scheme |
| `02_extract_*_vef` | reads the sequence-to-function model predictions and extracts the four-assay VEF matrix per dataset |
| `03_normalize_vef` | log1p transform for Enformer and Borzoi; Sei and AlphaGenome are normalized during extraction |
| `04_vef_activity_specificity` | sanity check of VEF-activity and VEF-specificity correlations, prints to stdout |
| `05_train_vef_only_models` | fits the VEF-only baselines that never see sequence |
| `06_infer_trained_model` | runs an EpiCast checkpoint, optionally against another dataset config, which is how the zero-shot predictions are produced |
| `07`, `08`, `09` | regression, CTS classification and top-k retrieval metrics |
| `11_vef_pairwise_correlation` | correlations among the four VEFs |
| `11_vef_partial_correlation` | each VEF against activity after conditioning on the other three, plus standardized OLS coefficients |
| `12_eval_castillo` | zero-shot evaluation on the independent Castillo-Hair MPRA |
| `14_export_prediction_tables` | measured and predicted activity side by side in one self-describing table per model |
| `15_export_figure_metrics` | aggregates the metric tables the figure scripts read |

Two conventions are worth knowing before reading the numbers. **Residual activity** always
means the activity of a CRE minus its mean over the three training cell types, which is the
quantity that carries cell-type specificity. **CTS CREs** are defined on the Gosai MPRA by
percentile tails of that residual, and on the Castillo MPRA by an absolute activity gap
against the other evaluated cell types; the two datasets are never mixed.

## Figures

One PDF per panel; colorbars and legends are written as separate files. Composing panels
into a figure, and the panel letters themselves, is done by hand.

| Panel | Script |
|---|---|
| 1C | `plot/fig1c_vef_activity_correlation.py` |
| 1D, 1F | `plot/fig1df_activity_correlation_heatmap.py` |
| 1E | `plot/fig1e_dnase_residual_specificity.py` |
| 2A | `plot/fig2a_epicast_scatter.py` |
| 2B, 2C | `plot/fig2bc_activity_metrics.py` |
| 3A | `plot/fig3a_residual_metrics.py` |
| 3B, 3C | `plot/fig3bc_cts_prioritization.py` |
| 3D, 3E | `plot/fig3de_topk_retrieval.py` |
| 3F, 3G | `plot/fig3fg_topk_activity_profile.py` |
| 4A, 4C | `plot/fig4ac_vef_correlation_heatmap.py` |
| 4B, 4D, 4E | `plot/fig4bde_vef_partial_correlation.py` |
| 5A-5E | `plot/fig5_castillo_metrics.py` |
| S1 | `plot/figs1_vef_assay_selection.py` |

A few conventions are shared by all of them, and changing one figure should not require
touching another: font sizes come from seaborn's `talk` context and are never set per
script, panel height is 6 inches so that panels can be composed without rescaling, a cell
type or a model keeps the same colour in every figure it appears in, and Sei is always
drawn before AlphaGenome.
