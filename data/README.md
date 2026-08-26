# data/

Only the track metadata of the four sequence-to-function models is version controlled here,
because the analysis scripts parse it directly and it is small. Everything else — the MPRA
tables, the model predictions they are derived from, the VEF matrices and the trained
checkpoints — is distributed through Zenodo (<https://zenodo.org/records/17669741>) and has
to be unpacked into this directory.

## Tracked files

| File | Read by | Origin |
|---|---|---|
| `Sei/Sei_tracks_info.csv` | `analysis/01`, `analysis/02_extract_*_sei_vef` | track table of Sei, <https://github.com/FunctionLab/sei-framework> |
| `AlphaGenome/metadata.csv` | `analysis/01` | output metadata of AlphaGenome, <https://github.com/google-deepmind/alphagenome> |
| `Enformer/model_track_info.tsv` | `analysis/01` | target table of Enformer, <https://github.com/google-deepmind/deepmind-research/tree/master/enformer> |
| `Borzoi/targets_human.txt` | `analysis/01` | human target table of Borzoi, <https://github.com/calico/borzoi> |

These four files define which output track of each model becomes which VEF, so they are the
one piece of metadata the analysis cannot be reproduced without.

## Expected layout after downloading

```
data/
├── Gosai_MPRA/            raw ENCODE files, input of analysis/01_prepare_gosai_data.py
├── gosai_mpra/            the 760,679-row label table and the VEF matrices derived from it
├── castillo_mpra/         the independent MPRA used for the zero-shot evaluation
├── Sei/                   Sei_tracks_info.csv (tracked) and the Sei predictions
├── AlphaGenome/           metadata.csv (tracked) and the AlphaGenome predictions
├── Enformer/              model_track_info.tsv (tracked)
└── Borzoi/                targets_human.txt (tracked)
```

Trained EpiCast checkpoints go under `saved/`, which `paper/config.py` resolves relative to
the repository root.
