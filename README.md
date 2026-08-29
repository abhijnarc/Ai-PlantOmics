# Plant Multi-Omics Integration (WallOmicsData)

Analysis pipeline for integrating **transcriptomics, proteomics, and metabolomics**
data from the Arabidopsis **WallOmicsData** dataset, built as a single, top-to-bottom
runnable Jupyter notebook.

## Contents

| File | Description |
|---|---|
| [plant_multiomics_wallomics_preprocessing.ipynb](./plant_multiomics_wallomics_preprocessing.ipynb) | The deliverable notebook. Runs end-to-end with no manual steps. |
| [generate_wallomics_notebook.py](./generate_wallomics_notebook.py) | Generator script for the **data download & preprocessing** section (cells 0–12). |
| [generate_baseline_integration_section.py](./generate_baseline_integration_section.py) | Generator script that appends the **baseline PCA + early-fusion integration** section (cells 13–23). |
| [processed_data/](./processed_data) | Output artifacts written by the notebook (processed matrices, matched sample IDs, figures). |

The notebook is built programmatically: each generator script uses `nbformat` to
append a self-contained set of cells to the `.ipynb` file. This keeps notebook
construction reproducible and diff-friendly. To regenerate the notebook from
scratch, run the generator scripts in order (see [Regenerating the notebook](#regenerating-the-notebook)).

## Notebook sections

### 1. Data download & preprocessing
- Programmatically accesses/loads the WallOmicsData transcriptomics, proteomics,
  and metabolomics data (falls back to a clearly-labeled synthetic dataset if the
  real package/source is unavailable in the current environment, so the notebook
  always runs).
- Identifies sample IDs per modality and computes the intersection of samples
  shared across all three.
- Builds three sample × feature `DataFrame`s aligned to the shared sample set.
- Reports per-modality sample/feature counts and missing-value percentages.
- Preprocessing: missing-value imputation, near-zero-variance feature removal,
  and standardization (z-score) per modality.
- Saves outputs to `processed_data/`:
  - `rna_processed.csv`
  - `proteomics_processed.csv`
  - `metabolomics_processed.csv`
  - `sample_metadata.csv`
  - `matched_sample_ids.txt`

### 2. Baseline multi-omics integration analysis
- Loads the four processed files above and verifies sample IDs are identical
  and identically ordered across modalities.
- Runs PCA separately per modality (up to 20 components, capped by
  `min(n_samples - 1, n_features)`).
- Builds an **early-fusion** representation by concatenating the per-modality
  PCA matrices (`RNA PCA + Proteomics PCA + Metabolomics PCA`).
- Computes a 2D embedding (UMAP if available, otherwise a safe fallback to
  scikit-learn's t-SNE — see [Notes on UMAP](#notes-on-umap)) for RNA,
  Proteomics, Metabolomics, and the fused representation.
- Colors the embeddings by a metadata variable when one exists, or by
  unsupervised KMeans clusters otherwise.
- Prints a dimension summary table for every representation and saves a
  4-panel side-by-side figure to `processed_data/baseline_integration_umap.png`.
- Includes Markdown explaining what PCA does per modality, why it helps with
  high-dimensional omics data, what early fusion means, and what information
  can be lost by simple concatenation.

No model training (neural network) is included yet — this section is the
classical baseline that a later AI integration model will be compared against.

## Setup

A virtual environment (`.venv`) is expected at the repository root.

```powershell
# Create and activate the venv (if not already present)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install core dependencies
pip install pandas numpy scikit-learn matplotlib nbformat nbconvert ipykernel

# Optional: UMAP for embeddings (falls back to t-SNE automatically if unavailable)
pip install umap-learn
```

## Running the notebook

Open [plant_multiomics_wallomics_preprocessing.ipynb](./plant_multiomics_wallomics_preprocessing.ipynb)
in VS Code / Jupyter and run all cells, or execute it headlessly:

```powershell
.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace plant_multiomics_wallomics_preprocessing.ipynb
```

The notebook is designed to run top-to-bottom without manual intervention.

## Regenerating the notebook

The notebook is produced by generator scripts rather than hand-edited, so it can
be rebuilt deterministically:

```powershell
.venv\Scripts\python.exe generate_wallomics_notebook.py
.venv\Scripts\python.exe generate_baseline_integration_section.py
```

Each generator is idempotent — it checks for a marker cell before appending, so
re-running it will not duplicate a section that's already present.

## Notes on UMAP

`umap-learn` depends on `numba`, which loads native DLLs at import time. In some
locked-down Windows environments (e.g. under an Application Control / AppLocker
policy), those DLLs are blocked and `import umap` fails. To keep the notebook
robust in any environment:

- The dependency-check cell probes `import umap` in an **isolated subprocess**
  first, so a blocked DLL never crashes the main notebook kernel.
- If the probe fails, the notebook prints a warning and transparently falls
  back to `sklearn.manifold.TSNE` for all 2D embeddings.
- The dimension-summary cell reports which embedding method was actually used.

## Requirements

- Python 3.10+ (developed against Python 3.14 in `.venv`)
- pandas, numpy, scikit-learn, matplotlib
- nbformat, nbconvert, ipykernel (for programmatic notebook generation/execution)
- umap-learn (optional; t-SNE fallback provided)
