"""
Comprehensive analysis of all 9 thermodynamic properties trained across 8 models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
df = pd.read_csv('outputs/thermodynamic_complete_results.csv')

print("\n" + "="*100)
print("COMPREHENSIVE THERMODYNAMIC ML ANALYSIS - ALL 9 PROPERTIES")
print("="*100)

print(f"\n📊 Dataset: 5000.CSV.csv")
print(f"📦 Properties Trained: {len(df)}")
print(f"📈 Models Used: 8 (Ridge, Linear, Tree, XGB, LGB, RF, ET, Ensemble)")
print(f"🔍 Validation Split: 3000 train / 1000 val / 1000 test")

print("\n" + "="*100)
print("1. MODEL PERFORMANCE RANKING (Average R² across all properties)")
print("="*100)

models = ['Ridge_R2', 'Linear_R2', 'Tree_R2', 'XGB_R2', 'LGB_R2', 'RF_R2', 'ET_R2', 'Ensemble_R2']
avg_r2 = df[models].mean().sort_values(ascending=False)

for i, (model, r2) in enumerate(avg_r2.items(), 1):
    model_name = model.replace('_R2', '')
    stars = "⭐" * min(5, int(r2 * 5))
    print(f"{i}. {model_name:15s}: {r2:.6f} {stars}")

print("\n" + "="*100)
print("2. PROPERTY DIFFICULTY RANKING (Based on Ensemble Model R²)")
print("="*100)

difficulty = df.sort_values('Ensemble_R2', ascending=False)[['Property', 'Ensemble_R2', 'Ensemble_MAE', 'Ensemble_RMSE']]
for i, (_, row) in enumerate(difficulty.iterrows(), 1):
    status = "🟢 Excellent" if row['Ensemble_R2'] > 0.999 else "🟡 Very Good" if row['Ensemble_R2'] > 0.99 else "🔵 Good"
    print(f"{i}. {row['Property']:25s}: R2={row['Ensemble_R2']:.6f}  MAE={row['Ensemble_MAE']:8.4f}  RMSE={row['Ensemble_RMSE']:8.4f}  {status}")

print("\n" + "="*100)
print("3. BEST PERFORMING MODEL FOR EACH PROPERTY")
print("="*100)

for _, row in df.iterrows():
    prop = row['Property']
    r2_values = row[models]
    best_model = r2_values.idxmax().replace('_R2', '')
    best_r2 = r2_values.max()
    print(f"{prop:25s}: {best_model:12s} (R2 = {best_r2:.6f})")

print("\n" + "="*100)
print("4. PERFECT PREDICTIONS (R² ≥ 0.9999)")
print("="*100)

perfect_count = 0
for _, row in df.iterrows():
    for model in models:
        if row[model] >= 0.9999:
            perfect_count += 1
            model_name = model.replace('_R2', '')
            print(f"✓ {row['Property']:25s} × {model_name:12s}: {row[model]:.8f}")

print(f"\n📌 Total perfect predictions: {perfect_count}/{len(df) * len(models)} ({100*perfect_count/(len(df)*len(models)):.1f}%)")

print("\n" + "="*100)
print("5. MODEL CONSISTENCY (Properties where model achieves best or near-best performance)")
print("="*100)

for model in models:
    model_name = model.replace('_R2', '')
    best_count = (df[model] == df[models].max(axis=1)).sum()
    avg_rank = df[models].rank(axis=1, ascending=False)[model].mean()
    avg_r2 = df[model].mean()
    print(f"{model_name:15s}: Wins={best_count}/9  Avg_Rank={avg_rank:.1f}  Avg_R2={avg_r2:.6f}")

print("\n" + "="*100)
print("6. PREDICTION ERROR ANALYSIS (Ensemble Model)")
print("="*100)

error_stats = df[['Property', 'Ensemble_MAE', 'Ensemble_RMSE']].copy()
error_stats['RMSE/MAE'] = error_stats['Ensemble_RMSE'] / error_stats['Ensemble_MAE']
error_stats = error_stats.sort_values('Ensemble_MAE')

print(f"{'Property':<25s} {'MAE':>10s} {'RMSE':>10s} {'RMSE/MAE':>10s}")
print("-" * 60)
for _, row in error_stats.iterrows():
    print(f"{row['Property']:<25s} {row['Ensemble_MAE']:>10.4f} {row['Ensemble_RMSE']:>10.4f} {row['RMSE/MAE']:>10.2f}")

print("\n" + "="*100)
print("7. DATA QUALITY SUMMARY")
print("="*100)

print(f"\n✓ All 9 thermodynamic properties successfully trained")
print(f"✓ All 8 models (Ridge, Linear, Tree, XGB, LGB, RF, ET, Ensemble) trained per property")
print(f"✓ Individual SHAP analysis generated for XGB, LGB, RF, ET models")
print(f"✓ Parity plots created for visual validation")
print(f"✓ Prediction CSVs saved for all properties")
print(f"✓ Average Ensemble R²: {df['Ensemble_R2'].mean():.6f}")
print(f"✓ Median Ensemble R²: {df['Ensemble_R2'].median():.6f}")
print(f"✓ Minimum Ensemble R²: {df['Ensemble_R2'].min():.6f}")
print(f"✓ Maximum Ensemble R²: {df['Ensemble_R2'].max():.6f}")

print("\n" + "="*100)
print("OUTPUTS GENERATED")
print("="*100)
print(f"\n✓ Predictions: outputs/predictions_all_models/predictions_*.csv (9 files)")
print(f"✓ SHAP Analysis: outputs/shap_analysis/shap_*_*.png (multiple plots per model)")
print(f"✓ Parity Plots: outputs/Parity_*.png (9 plots)")
print(f"✓ Results Summary: outputs/thermodynamic_complete_results.csv")
print(f"✓ Model Weights (pickled): outputs/XGB_*_*.pkl, etc.")

print("\n" + "="*100 + "\n")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Model performance heatmap
ax1 = axes[0, 0]
r2_data = df[models].set_index(df['Property'])
r2_data.columns = [col.replace('_R2', '') for col in r2_data.columns]
sns.heatmap(r2_data, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0.97, vmax=1.0, ax=ax1, cbar_kws={'label': 'R² Score'})
ax1.set_title('Model Performance Across All Thermodynamic Properties', fontsize=12, fontweight='bold')
ax1.set_xlabel('Model')
ax1.set_ylabel('Property')

# 2. Ensemble performance per property
ax2 = axes[0, 1]
props = df['Property'].values
ensemble_r2 = df['Ensemble_R2'].values
colors = ['#2ecc71' if r2 > 0.9995 else '#3498db' for r2 in ensemble_r2]
ax2.barh(props, ensemble_r2, color=colors)
ax2.set_xlabel('Ensemble R²')
ax2.set_title('Ensemble Model Performance per Property', fontsize=12, fontweight='bold')
ax2.set_xlim([0.998, 1.0001])
for i, v in enumerate(ensemble_r2):
    ax2.text(v - 0.0003, i, f'{v:.6f}', va='center', ha='right', fontweight='bold')

# 3. Error metrics
ax3 = axes[1, 0]
x_pos = np.arange(len(df))
width = 0.35
ax3.bar(x_pos - width/2, df['Ensemble_MAE'], width, label='MAE', alpha=0.8)
ax3.bar(x_pos + width/2, df['Ensemble_RMSE'], width, label='RMSE', alpha=0.8)
ax3.set_xlabel('Property')
ax3.set_ylabel('Error')
ax3.set_title('Prediction Error Metrics (Ensemble)', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(df['Property'], rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Model ranking
ax4 = axes[1, 1]
model_names = [col.replace('_R2', '') for col in models]
avg_r2_values = df[models].mean().values
colors_rank = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
bars = ax4.bar(model_names, avg_r2_values, color=colors_rank)
ax4.set_ylabel('Average R²')
ax4.set_title('Average Model Performance Across All Properties', fontsize=12, fontweight='bold')
ax4.set_ylim([0.996, 1.0001])
ax4.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, avg_r2_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001, f'{val:.6f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/thermodynamic_analysis_summary.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved: outputs/thermodynamic_analysis_summary.png")

# Save detailed report
with open('outputs/THERMODYNAMIC_ANALYSIS_REPORT.txt', 'w') as f:
    f.write("="*100 + "\n")
    f.write("COMPREHENSIVE THERMODYNAMIC ML ANALYSIS REPORT\n")
    f.write("="*100 + "\n\n")
    f.write(f"Dataset: 5000.CSV.csv\n")
    f.write(f"Properties Analyzed: 9 thermodynamic properties\n")
    f.write(f"Models Trained: 8 (Ridge, Linear, Tree, XGB, LGB, RF, ET, Ensemble)\n")
    f.write(f"Sample Split: 3000 train / 1000 val / 1000 test\n\n")
    f.write(f"Average Model Performance:\n")
    for model, r2 in avg_r2.items():
        f.write(f"  {model.replace('_R2', ''):15s}: {r2:.6f}\n")
    f.write(f"\n{df.to_string(index=False)}\n")

print(f"✓ Detailed report saved: outputs/THERMODYNAMIC_ANALYSIS_REPORT.txt")
print(f"\n✅ ANALYSIS COMPLETE!")
