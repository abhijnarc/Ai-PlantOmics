# Ai-PlantOmics: WallOmics Multi-Omics Workshop

This workshop notebook demonstrates PCA, sample-similarity graphs, GCNs, and MOGONET-style VCDN fusion on real, matched *Arabidopsis thaliana* WallOmics data. The prediction target is the real experimental temperature label (15°C versus 22°C); no synthetic data or labels are used.

## Quick start

1. Clone or download this repository.
2. Obtain the frozen WallOmics data by either method below.
3. Place the extracted CSV files in `data/processed/`.
4. Open and run `plant_multiomics.ipynb` from top to bottom.

The workshop notebook does not download raw WallOmics files or preprocess data. It uses only the frozen inputs and works with relative paths on Windows, Linux, and macOS.

## Obtain frozen data

- **Option A — GitHub:** download the files directly from [`data/processed/`](data/processed/).
- **Option B — Google Drive:** download the frozen-data ZIP from `<GOOGLE_DRIVE_DATA_LINK>` and extract its contents into `data/processed/`.

The MOGONET demonstration requires exactly these files:

- `data/processed/rna_processed.csv`
- `data/processed/proteomics_processed.csv`
- `data/processed/metabolomics_processed.csv`
- `data/processed/sample_metadata.csv`

## Workshop flow

```
Load frozen WallOmics data
→ inspect modalities
→ RNA / proteomics / metabolomics
→ PCA representations
→ sample-similarity graphs
→ GCNs
→ VCDN fusion
→ temperature prediction
→ baseline comparison
→ modality ablation
→ take-home message
```

## Evaluation protocol

The MOGONET section preserves a leakage-free protocol: the stratified train/test split happens first; PCA is fit on training samples only; training graphs use training samples only; and each held-out sample connects only to training neighbors at inference.

With 30 samples, accuracy and macro-F1 are illustrative architecture demonstrations, not biological performance benchmarks.

## Dependencies

Use Python 3.10+ with `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `networkx`, and CPU PyTorch. UMAP is optional; the notebook falls back to t-SNE when it is unavailable.

The scripts prefixed `generate_` are development utilities and are not part of the participant workflow.
