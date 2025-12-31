"""
Minimal Working Example (MWE) for DSPCA on MNIST Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split


# Add project root to path so we can import dspca without installing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dspca import DSPCA


# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = "/Users/espoma/Desktop/espoma-ml/dspca/data/mnist/mnist.parquet"
N_COMPONENTS = 3
SPARSITY_LEVELS = [70, 30, 20]  # Nested sparsity
MAX_SENSORS = 100  # Maximum number of sensors to use
N_SAMPLES = 1000  # Set to None to use full dataset


# ============================================================================
# Load Data
# ============================================================================
print("Loading MNIST data...")
df = pd.read_parquet(DATA_PATH)

# Extract features and labels
if 'class' in df.columns:
    X = df.drop('class', axis=1).values
    y = df['class'].values
else:
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

# Convert labels to integers and normalize features
y = y.astype(int)
X = X.astype(np.float64) / 255.0

print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Use subset if specified
if N_SAMPLES is not None and N_SAMPLES < X.shape[0]:
    # Use stratified sampling to ensure all digits are represented proportionally
    X, _, y, _ = train_test_split(
        X, y, 
        train_size=N_SAMPLES, 
        stratify=y, 
        random_state=42
    )
    print(f"✓ Using stratified subset: {X.shape[0]} samples")


# ============================================================================
# Fit DSPCA
# ============================================================================
print(f"\nFitting DSPCA with {N_COMPONENTS} components...")
print(f"Sparsity levels: {SPARSITY_LEVELS}")
print(f"Max sensors: {MAX_SENSORS}")

dspca = DSPCA(
    n_components=N_COMPONENTS, 
    sparsity_levels=SPARSITY_LEVELS,
    max_sensors=MAX_SENSORS
)
start_time = time.time()
dspca.fit(X)
end_time = time.time()
print(f"✓ DSPCA fitted successfully! ({end_time - start_time:.2f} seconds)")


# ============================================================================
# Display Results
# ============================================================================
print("\n" + "="*70)
print(" "*25 + "DSPCA RESULTS")
print("="*70)

print(f"\nTotal variance in data: {dspca.total_variance_:.4f}")
print(f"Number of components: {N_COMPONENTS}")

print("\nComponent Details:")
print("-" * 70)
for i in range(N_COMPONENTS):
    print(f"\n  PC{i+1}:")
    print(f"    Features selected: {len(dspca.components_[i])}")
    print(f"    Explained variance: {dspca.explained_variance_[i]:.4f}")
    print(f"    Percentage of total variance: {dspca.explained_variance_ratio_[i]*100:.2f}%")

# Cumulative variance
cumulative_pct = np.cumsum([pct * 100 for pct in dspca.explained_variance_ratio_])
print("\nCumulative Variance:")
print("-" * 70)
for i, cum_pct in enumerate(cumulative_pct):
    print(f"  PC1-PC{i+1}: {cum_pct:.2f}%")

print("\n" + "="*70)


# ============================================================================
# Visualization
# ============================================================================
print("\nCreating visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Number of features per component
ax1 = axes[0]
components = [f'PC{i+1}' for i in range(N_COMPONENTS)]
n_features = [len(comp) for comp in dspca.components_]
bars1 = ax1.bar(components, n_features, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Component', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax1.set_title('Features Selected per Component', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars1, n_features):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + max(n_features)*0.02,
            str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Explained variance
ax2 = axes[1]
bars2 = ax2.bar(components, dspca.explained_variance_, 
               color='forestgreen', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Component', fontsize=12, fontweight='bold')
ax2.set_ylabel('Explained Variance', fontsize=12, fontweight='bold')
ax2.set_title('Explained Variance per Component', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars2, dspca.explained_variance_):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + max(dspca.explained_variance_)*0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Percentage of explained variance
ax3 = axes[2]
variance_pct = [pct * 100 for pct in dspca.explained_variance_ratio_]
bars3 = ax3.bar(components, variance_pct, 
               color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Component', fontsize=12, fontweight='bold')
ax3.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
ax3.set_title('Percentage of Explained Variance', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars3, variance_pct):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + max(variance_pct)*0.02,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add cumulative line
x_pos = np.arange(len(components))
ax3_twin = ax3.twinx()
ax3_twin.plot(x_pos, cumulative_pct, 'o-', color='darkred', 
             linewidth=2.5, markersize=8, label='Cumulative %')
ax3_twin.set_ylabel('Cumulative Variance (%)', fontsize=11, fontweight='bold', color='darkred')
ax3_twin.tick_params(axis='y', labelcolor='darkred')
ax3_twin.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/espoma/Desktop/espoma-ml/dspca/results/dspca_summary.png', 
           dpi=150, bbox_inches='tight')
print("✓ Visualization saved to: results/dspca_summary.png")
plt.show()

print("\n✅ Analysis complete!")
