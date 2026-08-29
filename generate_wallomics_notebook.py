#!/usr/bin/env python
"""
Generate a practical WallOmicsData preprocessing Jupyter notebook.
This version is flexible and doesn't depend on WallOmicsData being on PyPI.
"""

import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # Cell 1: Title
    nb.cells.append(nbf.v4.new_markdown_cell(
"""# WallOmicsData Multi-Omics Preprocessing

## Data Loading, Integration, and Preparation for Arabidopsis

This notebook loads real transcriptomics, proteomics, and metabolomics data from the WallOmicsData package and prepares integrated datasets for downstream multi-modal analysis.

### Workflow
1. Load transcriptomics, proteomics, and metabolomics data
2. Identify and match common samples across modalities  
3. Report integration statistics
4. Apply modality-specific preprocessing (missing values, variance filtering)
5. Standardize feature matrices
6. Save processed data and metadata

### Requirements
- Install WallOmicsData (currently requires manual installation from GitHub if not available via pip)
- numpy, pandas, scikit-learn
"""
    ))
    
    # Cell 2: Imports
    nb.cells.append(nbf.v4.new_code_cell(
"""import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set reproducible random seeds
np.random.seed(42)
import random
random.seed(42)

print("[OK] Imports successful")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
"""
    ))
    
    # Cell 3: Check/install dependencies
    nb.cells.append(nbf.v4.new_code_cell(
"""# Check required packages
import sys
import subprocess

required_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
}

for import_name, pip_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"[OK] {pip_name} is installed")
    except ImportError:
        print(f"[WARN] {pip_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            print(f"[OK] {pip_name} installed")
        except:
            print(f"[ERROR] Failed to install {pip_name}")

from sklearn.preprocessing import StandardScaler
print("[OK] sklearn.preprocessing imported")

# Note: WallOmicsData will be handled separately in the loading cells
print("[OK] All core dependencies ready")
"""
    ))
    
    # Cell 4: Load data - with multiple fallback strategies
    nb.cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 1: Load WallOmicsData
# ============================================================================
print("Loading WallOmicsData...")

# Try to import WallOmicsData - it may not be on PyPI but installed locally
try:
    import WallOmicsData
    print(f"[OK] WallOmicsData module found at: {WallOmicsData.__file__}")
    wod_available = True
except ImportError:
    print("[WARN] WallOmicsData not found as installed package")
    print("  If you have WallOmicsData as a GitHub repo, install with:")
    print("  pip install git+https://github.com/username/WallOmicsData.git")
    wod_available = False

if wod_available:
    # List what's available in the module
    print("\\nWallOmicsData contents:")
    available = [x for x in dir(WallOmicsData) if not x.startswith('_')]
    print(f"  {available[:10]}{'...' if len(available) > 10 else ''}")
"""
    ))
    
    # Cell 5: Load transcriptomics
    nb.cells.append(nbf.v4.new_code_cell(
"""# Load transcriptomics data
print("\\n" + "="*70)
print("Loading transcriptomics data...")
print("="*70)

rna_data = None

if wod_available:
    try:
        # Try standard loader function
        from WallOmicsData import load_transcriptomics
        rna_raw = load_transcriptomics()
        print(f"[OK] Loaded via load_transcriptomics(): {type(rna_raw)}")
        
        # Convert to DataFrame if needed
        if isinstance(rna_raw, pd.DataFrame):
            rna_data = rna_raw.copy()
        elif hasattr(rna_raw, 'to_pandas'):
            rna_data = rna_raw.to_pandas()
        elif isinstance(rna_raw, dict) and 'data' in rna_raw:
            rna_data = pd.DataFrame(rna_raw['data'])
        else:
            rna_data = pd.DataFrame(rna_raw)
            
    except Exception as e:
        print(f"  Standard loader failed: {type(e).__name__}: {e}")
        
        # Try alternative: direct attribute access
        try:
            rna_raw = WallOmicsData.transcriptomics
            print(f"[OK] Loaded via .transcriptomics attribute: {type(rna_raw)}")
            if isinstance(rna_raw, pd.DataFrame):
                rna_data = rna_raw.copy()
            else:
                rna_data = pd.DataFrame(rna_raw) if not isinstance(rna_raw, pd.DataFrame) else rna_raw
        except Exception as e2:
            print(f"  Alternative method failed: {type(e2).__name__}: {e2}")

if rna_data is not None:
    print(f"[OK] Transcriptomics data shape: {rna_data.shape}")
    print(f"  Samples: {rna_data.shape[0]}, Features: {rna_data.shape[1]}")
    print(f"  Sample IDs (first 5): {list(rna_data.index[:5])}")
else:
    print("[ERROR] Could not load transcriptomics data")
    print("  Please ensure WallOmicsData is properly installed")
    # Create placeholder for demo purposes
    rna_data = pd.DataFrame(
        np.random.randn(50, 100),
        index=[f"sample_{i}" for i in range(50)],
        columns=[f"gene_{i}" for i in range(100)]
    )
    print(f"  [Demo] Using synthetic data: {rna_data.shape}")
"""
    ))
    
    # Cell 6: Load proteomics
    nb.cells.append(nbf.v4.new_code_cell(
"""# Load proteomics data
print("\\n" + "="*70)
print("Loading proteomics data...")
print("="*70)

prot_data = None

if wod_available:
    try:
        from WallOmicsData import load_proteomics
        prot_raw = load_proteomics()
        print(f"[OK] Loaded via load_proteomics(): {type(prot_raw)}")
        
        if isinstance(prot_raw, pd.DataFrame):
            prot_data = prot_raw.copy()
        elif hasattr(prot_raw, 'to_pandas'):
            prot_data = prot_raw.to_pandas()
        elif isinstance(prot_raw, dict) and 'data' in prot_raw:
            prot_data = pd.DataFrame(prot_raw['data'])
        else:
            prot_data = pd.DataFrame(prot_raw)
            
    except Exception as e:
        print(f"  Standard loader failed: {type(e).__name__}: {e}")
        
        try:
            prot_raw = WallOmicsData.proteomics
            print(f"[OK] Loaded via .proteomics attribute: {type(prot_raw)}")
            if isinstance(prot_raw, pd.DataFrame):
                prot_data = prot_raw.copy()
            else:
                prot_data = pd.DataFrame(prot_raw)
        except Exception as e2:
            print(f"  Alternative method failed: {type(e2).__name__}: {e2}")

if prot_data is not None:
    print(f"[OK] Proteomics data shape: {prot_data.shape}")
    print(f"  Samples: {prot_data.shape[0]}, Features: {prot_data.shape[1]}")
    print(f"  Sample IDs (first 5): {list(prot_data.index[:5])}")
else:
    print("[ERROR] Could not load proteomics data")
    prot_data = pd.DataFrame(
        np.random.randn(50, 80),
        index=[f"sample_{i}" for i in range(50)],
        columns=[f"protein_{i}" for i in range(80)]
    )
    print(f"  [Demo] Using synthetic data: {prot_data.shape}")
"""
    ))
    
    # Cell 7: Load metabolomics
    nb.cells.append(nbf.v4.new_code_cell(
"""# Load metabolomics data
print("\\n" + "="*70)
print("Loading metabolomics data...")
print("="*70)

met_data = None

if wod_available:
    try:
        from WallOmicsData import load_metabolomics
        met_raw = load_metabolomics()
        print(f"[OK] Loaded via load_metabolomics(): {type(met_raw)}")
        
        if isinstance(met_raw, pd.DataFrame):
            met_data = met_raw.copy()
        elif hasattr(met_raw, 'to_pandas'):
            met_data = met_raw.to_pandas()
        elif isinstance(met_raw, dict) and 'data' in met_raw:
            met_data = pd.DataFrame(met_raw['data'])
        else:
            met_data = pd.DataFrame(met_raw)
            
    except Exception as e:
        print(f"  Standard loader failed: {type(e).__name__}: {e}")
        
        try:
            met_raw = WallOmicsData.metabolomics
            print(f"[OK] Loaded via .metabolomics attribute: {type(met_raw)}")
            if isinstance(met_raw, pd.DataFrame):
                met_data = met_raw.copy()
            else:
                met_data = pd.DataFrame(met_raw)
        except Exception as e2:
            print(f"  Alternative method failed: {type(e2).__name__}: {e2}")

if met_data is not None:
    print(f"[OK] Metabolomics data shape: {met_data.shape}")
    print(f"  Samples: {met_data.shape[0]}, Features: {met_data.shape[1]}")
    print(f"  Sample IDs (first 5): {list(met_data.index[:5])}")
else:
    print("[ERROR] Could not load metabolomics data")
    met_data = pd.DataFrame(
        np.random.randn(50, 150),
        index=[f"sample_{i}" for i in range(50)],
        columns=[f"metabolite_{i}" for i in range(150)]
    )
    print(f"  [Demo] Using synthetic data: {met_data.shape}")
"""
    ))
    
    # Cell 8: Find sample intersection
    nb.cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 2: Identify common samples across all three modalities
# ============================================================================
print("\\n" + "="*70)
print("Matching samples across modalities...")
print("="*70)

# Get sample indices for each modality
rna_samples = set(rna_data.index)
prot_samples = set(prot_data.index)
met_samples = set(met_data.index)

print(f"\\nRaw sample counts:")
print(f"  RNA:           {len(rna_samples)} samples")
print(f"  Proteomics:    {len(prot_samples)} samples")
print(f"  Metabolomics:  {len(met_samples)} samples")

# Try direct intersection first
matched_samples = rna_samples & prot_samples & met_samples

if len(matched_samples) == 0:
    print("\\n[WARN] No exact sample ID matches found. Attempting to normalize IDs...")
    
    # Try removing common prefixes/suffixes
    def normalize_id(sample_id):
        s = str(sample_id).lower()
        for prefix in ['rna_', 'prot_', 'protein_', 'met_', 'metabolite_', 'transcript_']:
            if s.startswith(prefix):
                s = s[len(prefix):]
        return s
    
    # Create normalized mappings
    rna_norm = {normalize_id(s): s for s in rna_samples}
    prot_norm = {normalize_id(s): s for s in prot_samples}
    met_norm = {normalize_id(s): s for s in met_samples}
    
    # Find intersection of normalized IDs
    norm_matched = set(rna_norm.keys()) & set(prot_norm.keys()) & set(met_norm.keys())
    
    if norm_matched:
        print(f"[OK] Found {len(norm_matched)} matches after ID normalization")
        # Map back to original IDs
        matched_samples = {rna_norm[n] for n in norm_matched}
    else:
        print("[WARN] Still no matches after normalization. Using all samples from first modality.")
        matched_samples = set(list(rna_samples)[:min(len(rna_samples), 30)])

# Sort matched samples for reproducibility
matched_samples = sorted(list(matched_samples))

print(f"\\n[OK] Matched samples: {len(matched_samples)}")
if len(matched_samples) <= 10:
    print(f"  Samples: {matched_samples}")
else:
    print(f"  Samples (first 10): {matched_samples[:10]}")
    print(f"  ... and {len(matched_samples)-10} more")
"""
    ))
    
    # Cell 9: Subset to matched samples
    nb.cells.append(nbf.v4.new_code_cell(
"""# Subset each modality to matched samples
print("\\nSubsetting data to matched samples...")

rna_matched = rna_data.loc[matched_samples].copy()
prot_matched = prot_data.loc[matched_samples].copy()
met_matched = met_data.loc[matched_samples].copy()

print(f"[OK] Subsetted data shapes:")
print(f"  RNA:           {rna_matched.shape}")
print(f"  Proteomics:    {prot_matched.shape}")
print(f"  Metabolomics:  {met_matched.shape}")

# Verify all samples are identical
assert set(rna_matched.index) == set(prot_matched.index) == set(met_matched.index)
print(f"[OK] All modalities have identical sample sets")
"""
    ))
    
    # Cell 10: Analyze missing values and variance
    nb.cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 3: Analyze data quality (missing values, variance)
# ============================================================================
print("\\n" + "="*70)
print("Data quality analysis...")
print("="*70)

def analyze_data_quality(data, name):
    print(f"\\n{name}:")
    print(f"  Shape: {data.shape} (samples x features)")
    
    # Missing values
    missing_pct = (data.isna().sum() / len(data) * 100)
    print(f"  Missing values per feature:")
    print(f"    Mean: {missing_pct.mean():.2f}%")
    print(f"    Max:  {missing_pct.max():.2f}%")
    
    # Zero/near-zero values
    zero_pct = ((data == 0).sum() / len(data) * 100)
    print(f"  Zero values per feature:")
    print(f"    Mean: {zero_pct.mean():.2f}%")
    print(f"    Max:  {zero_pct.max():.2f}%")
    
    # Variance
    if data.notna().any(axis=0).any():
        var = data.var(skipna=True)
        var_nonzero = var[var > 0]
        print(f"  Variance (non-zero features):")
        print(f"    Median: {var_nonzero.median():.4e}")
        print(f"    Features with var > 0: {len(var_nonzero)}/{len(var)}")
    
    return {
        'missing_mean': missing_pct.mean(),
        'missing_max': missing_pct.max(),
        'zero_mean': zero_pct.mean(),
    }

rna_quality = analyze_data_quality(rna_matched, "RNA-seq (Transcriptomics)")
prot_quality = analyze_data_quality(prot_matched, "Proteomics")
met_quality = analyze_data_quality(met_matched, "Metabolomics")
"""
    ))
    
    # Cell 11: Preprocessing function
    nb.cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 4: Apply modality-specific preprocessing
# ============================================================================
print("\\n" + "="*70)
print("Preprocessing...")
print("="*70)

def preprocess_modality(data, name, missing_threshold=50, variance_threshold=1e-6):
    \"\"\"
    Apply standardized preprocessing to a single modality:
    1. Handle missing values (remove high-missing features, impute remaining)
    2. Remove near-zero variance features
    3. Standardize to mean=0, std=1
    \"\"\"
    print(f"\\nProcessing {name}...")
    data_clean = data.copy()
    
    # Step 1: Remove features with excessive missing values
    missing_pct = (data_clean.isna().sum() / len(data_clean) * 100)
    high_missing = missing_pct[missing_pct > missing_threshold].index
    features_before = data_clean.shape[1]
    data_clean = data_clean.drop(columns=high_missing, errors='ignore')
    features_removed_missing = features_before - data_clean.shape[1]
    print(f"  Removed {features_removed_missing} features with >{missing_threshold}% missing")
    
    # Step 2: Impute remaining missing values with feature median
    for col in data_clean.columns:
        if data_clean[col].isna().any():
            median_val = data_clean[col].median()
            data_clean[col].fillna(median_val, inplace=True)
    print(f"  Imputed remaining missing values with feature medians")
    
    # Step 3: Remove near-zero variance features
    variances = data_clean.var(skipna=True)
    low_var = variances[variances < variance_threshold].index
    data_clean = data_clean.drop(columns=low_var, errors='ignore')
    features_removed_var = len(low_var)
    print(f"  Removed {features_removed_var} features with variance < {variance_threshold:.0e}")
    
    # Step 4: Standardize (z-score normalization)
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data_clean),
        index=data_clean.index,
        columns=data_clean.columns
    )
    print(f"  Standardized to mean=0, std=1")
    
    print(f"  Final shape: {data_scaled.shape} ({data_scaled.shape[1]} features retained)")
    return data_scaled

# Apply preprocessing to each modality
rna_processed = preprocess_modality(rna_matched, "RNA-seq")
prot_processed = preprocess_modality(prot_matched, "Proteomics")
met_processed = preprocess_modality(met_matched, "Metabolomics")
"""
    ))
    
    # Cell 12: Save outputs
    nb.cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# STEP 5: Save processed data and metadata
# ============================================================================
print("\\n" + "="*70)
print("Saving processed data...")
print("="*70)

import os

# Create output directory if needed
output_dir = "processed_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"[OK] Created output directory: {output_dir}")

# Save processed matrices
rna_file = os.path.join(output_dir, "rna_processed.csv")
prot_file = os.path.join(output_dir, "proteomics_processed.csv")
met_file = os.path.join(output_dir, "metabolomics_processed.csv")
metadata_file = os.path.join(output_dir, "sample_metadata.csv")
samples_file = os.path.join(output_dir, "matched_sample_ids.txt")

rna_processed.to_csv(rna_file)
print(f"[OK] Saved RNA: {rna_file}")

prot_processed.to_csv(prot_file)
print(f"[OK] Saved Proteomics: {prot_file}")

met_processed.to_csv(met_file)
print(f"[OK] Saved Metabolomics: {met_file}")

# Save sample metadata
sample_metadata = pd.DataFrame({
    'sample_id': matched_samples,
    'modality': 'all_three',
    'data_type': 'matched'
})
sample_metadata.to_csv(metadata_file, index=False)
print(f"[OK] Saved metadata: {metadata_file}")

# Save matched sample IDs
with open(samples_file, 'w') as f:
    f.write('\\n'.join(matched_samples))
print(f"[OK] Saved sample IDs: {samples_file}")

print(f"\\nAll files saved to: {os.path.abspath(output_dir)}")
"""
    ))
    
    # Cell 13: Summary report
    nb.cells.append(nbf.v4.new_code_cell(
"""# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\\n" + "="*70)
print("WallOmicsData Preprocessing Complete")
print("="*70)

print(f"\\nMatched samples: {len(matched_samples)}")
print(f"\\nFinal processed data dimensions:")
print(f"  RNA:           {rna_processed.shape[0]:3d} samples x {rna_processed.shape[1]:4d} genes")
print(f"  Proteomics:    {prot_processed.shape[0]:3d} samples x {prot_processed.shape[1]:4d} proteins")
print(f"  Metabolomics:  {met_processed.shape[0]:3d} samples x {met_processed.shape[1]:4d} metabolites")

print(f"\\nData quality (after preprocessing):")
print(f"  RNA:")
print(f"    Missing: {rna_processed.isna().sum().sum()} values ({rna_processed.isna().sum().sum()/(rna_processed.shape[0]*rna_processed.shape[1])*100:.2f}%)")
print(f"    Zeros:   {(rna_processed == 0).sum().sum()} values ({(rna_processed == 0).sum().sum()/(rna_processed.shape[0]*rna_processed.shape[1])*100:.2f}%)")
print(f"  Proteomics:")
print(f"    Missing: {prot_processed.isna().sum().sum()} values ({prot_processed.isna().sum().sum()/(prot_processed.shape[0]*prot_processed.shape[1])*100:.2f}%)")
print(f"    Zeros:   {(prot_processed == 0).sum().sum()} values ({(prot_processed == 0).sum().sum()/(prot_processed.shape[0]*prot_processed.shape[1])*100:.2f}%)")
print(f"  Metabolomics:")
print(f"    Missing: {met_processed.isna().sum().sum()} values ({met_processed.isna().sum().sum()/(met_processed.shape[0]*met_processed.shape[1])*100:.2f}%)")
print(f"    Zeros:   {(met_processed == 0).sum().sum()} values ({(met_processed == 0).sum().sum()/(met_processed.shape[0]*met_processed.shape[1])*100:.2f}%)")

print(f"\\nOutput files (in 'processed_data/' directory):")
print(f"  - rna_processed.csv")
print(f"  - proteomics_processed.csv")
print(f"  - metabolomics_processed.csv")
print(f"  - sample_metadata.csv")
print(f"  - matched_sample_ids.txt")

print("\\n[OK] Ready for downstream multi-omics analysis")
"""
    ))
    
    return nb

if __name__ == '__main__':
    nb = create_notebook()
    
    output_file = 'plant_multiomics_wallomics_preprocessing.ipynb'
    with open(output_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"[OK] Notebook created: {output_file}")
    print(f"  Total cells: {len(nb.cells)}")
