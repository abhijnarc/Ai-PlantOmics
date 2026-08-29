#!/usr/bin/env python
"""
Append a baseline multi-omics integration analysis section (PCA + early fusion +
UMAP/t-SNE visualization) to the existing WallOmicsData preprocessing notebook.
"""

import nbformat as nbf

NOTEBOOK_PATH = "plant_multiomics_wallomics_preprocessing.ipynb"


def build_cells():
    cells = []

    # --- Section header ---
    cells.append(nbf.v4.new_markdown_cell(
"""---

# Baseline Multi-Omics Integration Analysis

This section builds a **simple baseline integration** on top of the preprocessed matrices
saved above (`rna_processed.csv`, `proteomics_processed.csv`, `metabolomics_processed.csv`,
`sample_metadata.csv`). It does **not** involve any neural network — it is the classical
baseline that a later deep-learning integration model will be compared against.

### Workflow
1. Load the processed matrices and metadata from disk
2. Verify sample alignment across modalities
3. Reduce each modality with PCA (10-20 components)
4. Build an early-fusion representation by concatenating the PCA scores
5. Embed each PCA representation (and the fused representation) in 2D with UMAP
   (falls back to t-SNE if UMAP is unavailable on this machine)
6. Visualize all embeddings side by side, colored by a metadata variable
"""
    ))

    # --- Cell: imports + install check ---
    cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# Imports and dependency check
# ============================================================================
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def _ensure(pip_name, import_name=None):
    import_name = import_name or pip_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"[WARN] {pip_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", pip_name])
            __import__(import_name)
            print(f\"[OK] {pip_name} installed\")
            return True
        except Exception as e:
            print(f\"[WARN] Could not install {pip_name}: {e}\")
            return False

_ensure(\"matplotlib\")

# UMAP is optional and its 'numba' dependency can trigger OS-level DLL/JIT policy
# blocks on some machines. We probe UMAP importability in an isolated subprocess
# first, so a blocked DLL never touches this kernel process; only import it here
# if the probe succeeds. Otherwise we transparently fall back to sklearn's t-SNE.
UMAP_AVAILABLE = False
probe = subprocess.run(
    [sys.executable, \"-c\", \"import umap\"],
    capture_output=True, text=True
)
if probe.returncode == 0:
    import umap
    UMAP_AVAILABLE = True
    print(\"[OK] UMAP available - will use UMAP for 2D embeddings\")
else:
    reason = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else \"import failed\"
    print(f\"[WARN] UMAP unavailable ({reason})\")
    print(\"  Falling back to sklearn TSNE for 2D embeddings\")

from sklearn.manifold import TSNE  # always import - cheap, and used as fallback

print(\"[OK] Ready for baseline integration analysis\")
"""
    ))

    # --- Cell: load processed matrices ---
    cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 1: Load preprocessed matrices from the previous section
# ============================================================================
DATA_DIR = \"processed_data\"

rna = pd.read_csv(f\"{DATA_DIR}/rna_processed.csv\", index_col=0)
prot = pd.read_csv(f\"{DATA_DIR}/proteomics_processed.csv\", index_col=0)
met = pd.read_csv(f\"{DATA_DIR}/metabolomics_processed.csv\", index_col=0)
metadata = pd.read_csv(f\"{DATA_DIR}/sample_metadata.csv\", index_col=0)

print(\"Loaded processed matrices:\")
print(f\"  RNA:           {rna.shape[0]:3d} samples x {rna.shape[1]:4d} features\")
print(f\"  Proteomics:    {prot.shape[0]:3d} samples x {prot.shape[1]:4d} features\")
print(f\"  Metabolomics:  {met.shape[0]:3d} samples x {met.shape[1]:4d} features\")
print(f\"  Metadata:      {metadata.shape[0]:3d} samples x {metadata.shape[1]:4d} columns\")
"""
    ))

    # --- Cell: verify sample alignment ---
    cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 2: Verify sample IDs are identical and in the same order
# ============================================================================
ids_rna = list(rna.index)
ids_prot = list(prot.index)
ids_met = list(met.index)
ids_meta = list(metadata.index)

same_set = set(ids_rna) == set(ids_prot) == set(ids_met) == set(ids_meta)
same_order = (ids_rna == ids_prot == ids_met == ids_meta)

print(f\"Same sample set across all modalities:   {same_set}\")
print(f\"Same sample order across all modalities: {same_order}\")

if same_set and not same_order:
    print(\"[WARN] Sample sets match but order differs. Reindexing to RNA order...\")
    prot = prot.reindex(ids_rna)
    met = met.reindex(ids_rna)
    metadata = metadata.reindex(ids_rna)
    same_order = True

if not same_set:
    # Align to the intersection defensively (should not happen if the previous
    # section's outputs are used as-is, but keeps this section robust/standalone).
    common = sorted(set(ids_rna) & set(ids_prot) & set(ids_met) & set(ids_meta))
    print(f\"[WARN] Sample sets differ. Re-aligning to {len(common)} common samples.\")
    rna = rna.reindex(common)
    prot = prot.reindex(common)
    met = met.reindex(common)
    metadata = metadata.reindex(common)

assert list(rna.index) == list(prot.index) == list(met.index) == list(metadata.index)
print(f\"[OK] {len(rna.index)} samples aligned and ordered identically across modalities\")
"""
    ))

    # --- Cell: PCA per modality ---
    cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 3: Apply PCA separately to each modality
# ============================================================================
n_samples = rna.shape[0]

# Number of PCA components is capped by both a target range (10-20) and by
# what's mathematically possible (n_components <= min(n_samples, n_features)).
N_COMPONENTS_TARGET = 20
max_components = max(2, min(N_COMPONENTS_TARGET, n_samples - 1))

def run_pca(df, name, n_components):
    n_comp = min(n_components, df.shape[1], df.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    scores = pca.fit_transform(df.values)
    scores_df = pd.DataFrame(
        scores,
        index=df.index,
        columns=[f\"{name}_PC{i+1}\" for i in range(n_comp)]
    )
    var_explained = pca.explained_variance_ratio_.sum()
    print(f\"{name:12s}: {df.shape[1]:4d} features -> {n_comp:2d} PCs \"
          f\"({var_explained*100:5.1f}% variance explained)\")
    return scores_df, pca

print(f\"Target components: up to {max_components} (min(n_samples-1, {N_COMPONENTS_TARGET}))\\n\")

rna_pca, rna_pca_model = run_pca(rna, \"RNA\", max_components)
prot_pca, prot_pca_model = run_pca(prot, \"Proteomics\", max_components)
met_pca, met_pca_model = run_pca(met, \"Metabolomics\", max_components)
"""
    ))

    # --- Cell: early fusion ---
    cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 4: Early fusion - concatenate PCA representations
# ============================================================================
# RNA_PCA + Proteomics_PCA + Metabolomics_PCA -> integrated matrix
fused_pca = pd.concat([rna_pca, prot_pca, met_pca], axis=1)

print(\"Early-fusion (concatenated) representation:\")
print(f\"  RNA PCA:           {rna_pca.shape}\")
print(f\"  Proteomics PCA:    {prot_pca.shape}\")
print(f\"  Metabolomics PCA:  {met_pca.shape}\")
print(f\"  ---------------------------------\")
print(f\"  Fused matrix:      {fused_pca.shape}\")

assert fused_pca.shape[1] == rna_pca.shape[1] + prot_pca.shape[1] + met_pca.shape[1]
assert list(fused_pca.index) == list(rna.index)
print(\"[OK] Fused representation is sample-aligned with the original matrices\")
"""
    ))

    # --- Cell: choose coloring variable ---
    cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 5: Choose a biological/experimental variable for coloring the plots
# ============================================================================
# Use whatever informative categorical/numeric column is available in metadata
# (e.g. genotype, treatment, condition, timepoint). Columns that are constant
# across all matched samples (e.g. 'data_type' == 'matched') carry no
# information for coloring, so we skip those automatically.
candidate_cols = [c for c in metadata.columns if metadata[c].nunique() > 1]

if candidate_cols:
    color_col = candidate_cols[0]
    color_values = metadata[color_col]
    print(f\"[OK] Coloring UMAP/t-SNE plots by metadata column: '{color_col}'\")
else:
    # No informative metadata column was found (common when only synthetic
    # placeholder metadata is available). Fall back to an unsupervised grouping
    # derived from the fused representation purely for visualization purposes.
    from sklearn.cluster import KMeans
    n_clusters = min(4, n_samples)
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(fused_pca.values)
    color_col = \"unsupervised_cluster\"
    color_values = pd.Series(cluster_labels, index=metadata.index, name=color_col)
    print(f\"[WARN] No informative metadata column found (all columns are constant).\")
    print(f\"  Coloring plots by unsupervised KMeans cluster (k={n_clusters}) instead.\")

print(color_values.value_counts())
"""
    ))

    # --- Cell: 2D embedding function (UMAP or TSNE fallback) ---
    cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 6: 2D embedding of each PCA representation (UMAP, or t-SNE fallback)
# ============================================================================
def embed_2d(matrix, n_samples, random_state=42):
    n_neighbors = max(2, min(15, n_samples - 1))
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=random_state)
        return reducer.fit_transform(matrix), \"UMAP\"
    else:
        perplexity = max(2, min(30, n_samples - 1))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init=\"pca\")
        return reducer.fit_transform(matrix), \"t-SNE\"

representations = {
    \"RNA\": rna_pca,
    \"Proteomics\": prot_pca,
    \"Metabolomics\": met_pca,
    \"Fused (early integration)\": fused_pca,
}

embeddings = {}
method_name = None
for name, mat in representations.items():
    emb, method_name = embed_2d(mat.values, n_samples)
    embeddings[name] = emb
    print(f\"{name:28s}: {mat.shape} -> 2D {method_name} embedding {emb.shape}\")
"""
    ))

    # --- Cell: side-by-side visualization ---
    cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 7: Side-by-side visualization of all embeddings
# ============================================================================
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

is_categorical = not pd.api.types.is_numeric_dtype(color_values) or color_values.nunique() <= 10
if is_categorical:
    categories = pd.Categorical(color_values)
    codes = categories.codes
    cmap = plt.get_cmap(\"tab10\")
else:
    codes = color_values.values
    cmap = plt.get_cmap(\"viridis\")

for ax, (name, emb) in zip(axes, embeddings.items()):
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap=cmap, s=35, alpha=0.85, edgecolors=\"white\", linewidths=0.3)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel(f\"{method_name} 1\")
    ax.set_ylabel(f\"{method_name} 2\")

if is_categorical:
    handles = [plt.Line2D([0], [0], marker=\"o\", color=\"w\", markerfacecolor=cmap(i / max(1, len(categories.categories) - 1)),
                          markersize=8, label=str(cat)) for i, cat in enumerate(categories.categories)]
    fig.legend(handles=handles, title=color_col, loc=\"center right\", bbox_to_anchor=(1.12, 0.5))
else:
    fig.colorbar(scatter, ax=axes, shrink=0.7, label=color_col)

fig.suptitle(f\"Baseline {method_name} embeddings colored by '{color_col}'\", fontsize=14)
plt.tight_layout()
plt.savefig(\"processed_data/baseline_integration_umap.png\", dpi=150, bbox_inches=\"tight\")
plt.show()

print(\"[OK] Saved figure: processed_data/baseline_integration_umap.png\")
"""
    ))

    # --- Cell: dimension summary ---
    cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 8: Summary of all representation dimensions
# ============================================================================
print(\"=\"*70)
print(\"Baseline Multi-Omics Integration: Dimension Summary\")
print(\"=\"*70)
print(f\"\\n{'Representation':<28s}{'Samples':>10s}{'Features':>12s}\")
print(\"-\"*50)
print(f\"{'RNA (processed)':<28s}{rna.shape[0]:>10d}{rna.shape[1]:>12d}\")
print(f\"{'Proteomics (processed)':<28s}{prot.shape[0]:>10d}{prot.shape[1]:>12d}\")
print(f\"{'Metabolomics (processed)':<28s}{met.shape[0]:>10d}{met.shape[1]:>12d}\")
print(\"-\"*50)
print(f\"{'RNA PCA':<28s}{rna_pca.shape[0]:>10d}{rna_pca.shape[1]:>12d}\")
print(f\"{'Proteomics PCA':<28s}{prot_pca.shape[0]:>10d}{prot_pca.shape[1]:>12d}\")
print(f\"{'Metabolomics PCA':<28s}{met_pca.shape[0]:>10d}{met_pca.shape[1]:>12d}\")
print(f\"{'Fused (early integration)':<28s}{fused_pca.shape[0]:>10d}{fused_pca.shape[1]:>12d}\")
print(\"-\"*50)
print(f\"2D embedding method used: {method_name}\")
print(\"\\n[OK] Baseline PCA + early-fusion integration complete\")
"""
    ))

    # --- Markdown explanation ---
    cells.append(nbf.v4.new_markdown_cell(
"""## What this baseline is doing

**PCA per modality.** For each omics layer (RNA, proteomics, metabolomics) we fit a
separate `PCA` that projects the standardized features onto a small number of orthogonal
components (10-20, capped by sample size). Each component is the linear combination of
original features that explains the most remaining variance. This gives a compact
low-dimensional summary of each modality's dominant axes of biological variation.

**Why PCA is useful for high-dimensional omics data.** Omics matrices are typically
"wide" (far more features than samples), which makes downstream distance-based methods
(clustering, UMAP/t-SNE, nearest-neighbor models) noisy, slow, and prone to overfitting.
PCA denoises the data by discarding low-variance directions (often dominated by
measurement noise), reduces dimensionality by 5-10x, and decorrelates features so that
later steps (fusion, embedding) operate on a small set of informative, independent axes.

**What early fusion means.** "Early fusion" here means concatenating the per-modality
PCA score matrices *before* any further modeling: `[RNA_PCA | Proteomics_PCA |
Metabolomics_PCA]` becomes one wide feature vector per sample. It is the simplest
possible multi-omics integration strategy - no modality-specific weighting, no
learned interactions, just horizontal stacking.

**What information may be lost by simple concatenation.**
- **Cross-modality relationships are not modeled.** Concatenation treats each
  modality's PCs as independent, unweighted features - it cannot learn that,
  say, a specific transcript's PC loading co-varies with a specific protein's PC
  loading (i.e., no cross-modal covariance structure is captured).
- **Unequal modality influence.** Modalities with more retained PCs or larger PC
  score magnitudes can dominate distance calculations and downstream models,
  even if they are not biologically more informative.
- **Nonlinear interactions are ignored.** PCA and concatenation are both linear;
  any nonlinear, combinatorial, or conditional relationships between omics
  layers (e.g., a metabolite level that only depends on a gene-protein pair)
  are invisible to this representation.
- **Information discarded within each modality.** PCA keeps only the top
  variance components, so lower-variance but potentially biologically
  meaningful signals (e.g., subtle regulatory changes) may be dropped.

This baseline (PCA + early fusion + 2D embedding) is intentionally simple. It
establishes a reference point that a later, learned multi-omics integration model
(e.g., a joint neural network with cross-modal attention or shared latent space)
should be able to improve upon by capturing exactly the cross-modal and nonlinear
structure that this baseline cannot.
"""
    ))

    return cells


def main():
    nb = nbf.read(NOTEBOOK_PATH, as_version=4)

    # Avoid duplicating the section if this script is re-run
    marker = "Baseline Multi-Omics Integration Analysis"
    already_present = any(
        cell.cell_type == "markdown" and marker in cell.source
        for cell in nb.cells
    )
    if already_present:
        print(f"[SKIP] Section '{marker}' already present in {NOTEBOOK_PATH}. Not modifying.")
        return

    new_cells = build_cells()
    nb.cells.extend(new_cells)

    nbf.write(nb, NOTEBOOK_PATH)
    print(f"[OK] Appended {len(new_cells)} cells to {NOTEBOOK_PATH}")
    print(f"  Total cells now: {len(nb.cells)}")


if __name__ == "__main__":
    main()
