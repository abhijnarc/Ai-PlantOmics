# WallOmics Plant Multi-Omics Integration Demo

A workshop-friendly, end-to-end Jupyter notebook demonstrating multi-omics data
integration on a **real** *Arabidopsis thaliana* dataset — from raw data download through
a classical statistical baseline to a graph-based deep learning architecture
([MOGONET](https://www.nature.com/articles/s41467-021-23774-w)).

No synthetic data or synthetic labels are used anywhere in this project.

## Dataset

**WallomicsData** (Durufle et al., 2020, *Cells* 9(10):2249,
[doi:10.3390/cells9102249](https://doi.org/10.3390/cells9102249)) — rosette-organ
transcriptomics, proteomics, and metabolomics profiling of 5 *Arabidopsis* ecotypes grown
at two temperatures (15°C and 22°C), 3 replicates each (30 samples total). The raw `.rda`
data and metadata are downloaded programmatically from the dataset's public source and
cached locally in `wallomics_raw/`.

## Notebook workflow

`plant_multiomics_wallomics_preprocessing.ipynb` runs top-to-bottom with no manual
intervention and implements three stages:

```
WallOmics raw data (.rda)
        ↓
1. Preprocessing   → matched RNA / Proteomics / Metabolomics matrices + real metadata
        ↓
2. Statistical baseline   → per-modality PCA, early fusion, UMAP visualization
        ↓
3. MOGONET-style integration   → per-modality sample graphs, GCNs, VCDN fusion,
                                   Temperature (15°C vs 22°C) prediction
```

### 1. Preprocessing (`generate_wallomics_notebook.py`)

- Downloads the real WallomicsData `.rda` files programmatically (with local caching in
  `wallomics_raw/`) and loads them via `pyreadr`.
- Loads transcriptomics, proteomics, and metabolomics data and identifies the sample IDs
  present in each modality.
- Computes the intersection of samples shared across all three modalities.
- Builds three matched DataFrames (samples × features) and reports sample counts, feature
  counts, and missing-value percentages for each modality.
- Applies modality-appropriate preprocessing: missing-value handling, near-zero-variance
  feature removal, and standardization.
- Recovers the **real** experimental metadata (Temperature, Ecotype) from the original
  WallomicsData source and matches it to the processed sample IDs — no metadata is
  inferred, randomized, or reconstructed from the molecular data.
- Saves:
  - `processed_data/rna_processed.csv`
  - `processed_data/proteomics_processed.csv`
  - `processed_data/metabolomics_processed.csv`
  - `processed_data/sample_metadata.csv`
  - `processed_data/matched_sample_ids.txt`

### 2. Statistical baseline (`generate_baseline_integration_section.py`)

- Reloads the processed matrices and verifies identical sample ordering across modalities.
- Applies PCA separately to each modality (10–20 components).
- Builds an **early-fusion** representation by concatenating the per-modality PCA scores.
- Embeds each PCA representation (and the fused representation) in 2D with UMAP (falls
  back to t-SNE if UMAP is unavailable), colored by a metadata variable.
- Explains in Markdown what PCA does for each modality, why it's useful for
  high-dimensional omics data, what early fusion means, and what information can be lost
  by simply concatenating modalities.
- This is the classical baseline that the MOGONET-style section is compared against — no
  neural network is involved here.

### 3. MOGONET-style graph-based integration (`generate_mogonet_section.py`)

Demonstrates (not benchmarks) how an established multi-omics AI architecture integrates
heterogeneous data using per-modality sample-similarity graphs, graph convolutional
networks (GCNs), and a "VCDN" late-fusion layer, predicting the **real** Temperature
(15°C vs 22°C) experimental factor.

```
modality-specific PCA → sample similarity graphs → modality-specific GCNs
                                                            ↓
                                                    VCDN multimodal fusion
                                                            ↓
                                                  Temperature prediction
```

**Leakage-free protocol.** A stratified train/test split is performed *before* any other
step. PCA is fit on training samples only (test samples are transformed with the
already-fitted PCA). k-NN sample-similarity graphs used for training are built from
training samples only; at inference time, held-out test samples are connected only to
their nearest *training* neighbors (never to each other), so no test sample or test label
can influence PCA, feature selection, scaling, or graph construction.

The section includes:
- Markdown explanations of what the sample graphs represent, why each modality gets its
  own graph, what one GCN message-passing step does, and what VCDN contributes.
- Four figures: an example modality-specific sample graph, a simple architecture diagram,
  baseline-vs-MOGONET performance, and a modality-ablation experiment (removing each omics
  layer in turn).
- Comparisons against an RNA-only classifier and a concatenated-PCA classifier baseline.

> **Note on sample size.** With only 30 real samples, all reported accuracy/F1 numbers are
> **illustrative** of the architecture running end-to-end on real data with a
> leakage-free protocol — they are **not** a rigorous performance benchmark, and no method
> is claimed to be biologically superior on this basis.

## Repository structure

```
plant_multiomics_wallomics_preprocessing.ipynb   # the final notebook (runs top-to-bottom)
generate_wallomics_notebook.py                   # builds the preprocessing section
generate_baseline_integration_section.py         # appends the statistical baseline section
generate_mogonet_section.py                      # appends the MOGONET-style section
wallomics_raw/                                   # cached raw WallomicsData .rda files
processed_data/
  rna_processed.csv                              # processed RNA matrix (samples x genes)
  proteomics_processed.csv                        # processed proteomics matrix
  metabolomics_processed.csv                      # processed metabolomics matrix
  sample_metadata.csv                             # real Temperature/Ecotype metadata
  matched_sample_ids.txt                          # final matched sample IDs
  baseline_integration_umap.png                   # baseline UMAP figure
  mogonet_rna_sample_graph.png                    # MOGONET figure 1
  mogonet_architecture_diagram.png                # MOGONET figure 2
  mogonet_vs_baselines.png                        # MOGONET figure 3
  mogonet_modality_ablation.png                   # MOGONET figure 4
```

## Reproducing the notebook

```powershell
python generate_wallomics_notebook.py
python generate_baseline_integration_section.py
python generate_mogonet_section.py
python -m jupyter nbconvert --to notebook --execute --inplace `
    plant_multiomics_wallomics_preprocessing.ipynb --ExecutePreprocessor.timeout=1200
```

Each generator script is idempotent: re-running it against a notebook that already
contains its section is a no-op (it will print `[SKIP]` and leave the notebook unchanged).
To regenerate from scratch, delete `plant_multiomics_wallomics_preprocessing.ipynb` first.

## Requirements

Python 3.10+ with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `networkx`, `pyreadr`,
`torch` (CPU build), and `umap-learn` (optional; falls back to t-SNE). Missing packages
are installed automatically by the notebook cells on first run.
