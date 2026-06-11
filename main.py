import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

COLORS = {
    "Linear Regression": "#1565C0",   
    "Ridge Regression":  "#2E7D32",   
    "Lasso Regression":  "#C62828",   
}

OUT = "outputs/"
os.makedirs(OUT, exist_ok=True)

print("┌─────────────────────────────────────────────────────────┐")
print("│  Loading Diabetes Dataset                               │")
print("└─────────────────────────────────────────────────────────┘")

diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target

print(f"  Total samples   : {df.shape[0]}")
print(f"  Total features  : {len(diabetes.feature_names)}")
print(f"  Feature names   : {list(diabetes.feature_names)}")
print(f"\n  Target range    : [{df['target'].min():.1f}, {df['target'].max():.1f}]")

X_raw = df[["bmi"]].copy()
y_raw = df["target"].copy()

X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)
print(f"\n  Initial Train size : {len(X_tr)}  |  Test size : {len(X_te)}")

print("\n┌─────────────────────────────────────────────────────────┐")
print("│  Outlier Removal — Z-Score Method on Train (|Z| > 3)    │")
print("└─────────────────────────────────────────────────────────┘")

bmi_tr_flat = X_tr["bmi"].values
bmi_tr_mean = bmi_tr_flat.mean()
bmi_tr_std  = bmi_tr_flat.std(ddof=0)
z_scores_tr = np.abs((bmi_tr_flat - bmi_tr_mean) / bmi_tr_std)

THRESHOLD = 3
inlier_mask_tr = z_scores_tr <= THRESHOLD

X_tr_clean = X_tr[inlier_mask_tr]
y_tr_clean = y_tr[inlier_mask_tr]
n_outliers = int((~inlier_mask_tr).sum())

print(f"  μ(bmi_train)    : {bmi_tr_mean:+.6f}")
print(f"  σ(bmi_train)    : {bmi_tr_std:.6f}")
print(f"  Max |Z| on Train: {z_scores_tr.max():.4f}")
print(f"  Removed samples : {n_outliers}")
print(f"  Final Train size: {len(X_tr_clean)} (Outliers excluded)")

print("\n┌─────────────────────────────────────────────────────────┐")
print("│   Standardization (StandardScaler)                      │")
print("└─────────────────────────────────────────────────────────┘")

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_clean)    
X_te_sc = scaler.transform(X_te)  

print(f"  Scaler learned μ = {scaler.mean_[0]:.6f}  |  σ = {scaler.scale_[0]:.6f}")

def train_evaluate(X_train, X_test, y_train, y_test):
    configs = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression",  Ridge(alpha=1.0)),
        ("Lasso Regression",  Lasso(alpha=1.0, max_iter=20000)),
    ]
        
    out = {}
    for name, mdl in configs:
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        out[name] = {
            "model": mdl,
            "y_pred": y_pred,
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "coef": float(mdl.coef_[0]),
            "intercept": float(mdl.intercept_),
        }
    return out

print("\n┌─────────────────────────────────────────────────────────┐")
print("│   Training Models (Fixed Alpha=1.0 Before & After)      │")
print("└─────────────────────────────────────────────────────────┘")

res_before = train_evaluate(X_tr_clean.values, X_te.values, y_tr_clean.values, y_te.values)
res_after  = train_evaluate(X_tr_sc, X_te_sc, y_tr_clean.values, y_te.values)

model_keys = list(COLORS.keys())
print("\n  ╔═══════════════════╦══════════════════╦══════════════════╗")
print("  ║ Model            ║ Before Std        ║ After Std         ║")
print("  ║                  ║ MSE      R²       ║ MSE      R²       ║")
print("  ╠═══════════════════╬══════════════════╬══════════════════╣")
for name in model_keys:
    b = res_before[name]; a = res_after[name]
    print(f"  ║ {name:<16}  ║ {b['mse']:7.2f}  {b['r2']:6.4f}  ║ {a['mse']:7.2f}  {a['r2']:6.4f}  ║")
print("  ╚═══════════════════╩══════════════════╩══════════════════╝")

print("\n┌─────────────────────────────────────────────────────────┐")
print("│   Plotting & Saving Figures                             │")
print("└─────────────────────────────────────────────────────────┘")

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle("Outlier Detection — Z-Score Method (BMI Train Feature Only)", fontsize=13, fontweight="bold")

ax1a.scatter(
    X_tr_clean.values.flatten(), y_tr_clean,
    alpha=0.45, s=32, c="#546E7A", edgecolors="white", linewidths=0.3,
    label=f"Inliers Train (n={inlier_mask_tr.sum()})", zorder=2,
)
if n_outliers > 0:
    ax1a.scatter(
        X_tr.values[~inlier_mask_tr].flatten(), y_tr.values[~inlier_mask_tr],
        alpha=1.0, s=100, c="#E53935", marker="X", linewidths=1.5,
        label=f"Outliers |Z|>{THRESHOLD} (n={n_outliers})", zorder=3,
    )
ax1a.set_xlabel("BMI (original scale)")
ax1a.set_ylabel("Target")
ax1a.set_title(f"Train Scatter: BMI vs Target\n({n_outliers} outlier(s) removed)")
ax1a.legend(fontsize=9)

ax1b.hist(z_scores_tr, bins=30, color="#5C6BC0", alpha=0.82, edgecolor="white", linewidth=0.5)
ax1b.axvline(THRESHOLD, color="#E53935", linewidth=2.2, linestyle="--", label=f"Threshold |Z| = {THRESHOLD}")
ax1b.set_xlabel("|Z-Score|")
ax1b.set_ylabel("Frequency")
ax1b.set_title("Distribution of |Z-Scores| for Train Data")
ax1b.legend(fontsize=10)

fig1.tight_layout()
fig1.savefig(OUT + "fig1_outlier_detection.png", bbox_inches="tight")
print("  ✓ fig1_outlier_detection.png")

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
fig2.suptitle("Regression Prediction Lines — Before & After Standardization", fontsize=13, fontweight="bold")

def _add_lines(ax, X_scatter, y_scatter, results, x_range, xlabel, subtitle):
    ax.scatter(X_scatter.flatten(), y_scatter, alpha=0.40, s=32, c="#90A4AE", edgecolors="white", linewidths=0.3, label="Actual (test)", zorder=2)
    x_line = np.linspace(x_range[0], x_range[1], 500).reshape(-1, 1)
    for name, r in results.items():
        y_line = r["model"].predict(x_line)
        alpha_val = f", α={r['model'].alpha}" if hasattr(r["model"], "alpha") else ""
        
        is_standardized = "after" in subtitle.lower()
        coef_label = f"std coef={r['coef']:.2f}" if is_standardized else f"coef={r['coef']:.2f}"
        
        ax.plot(x_line, y_line, color=COLORS[name], linewidth=2.5, label=f"{name}{alpha_val}\n{coef_label}", zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Target")
    ax.set_title(subtitle, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

xmin_raw = min(X_tr_clean.values.min(), X_te.values.min())
xmax_raw = max(X_tr_clean.values.max(), X_te.values.max())
_add_lines(ax2a, X_te.values, y_te.values, res_before, (xmin_raw - 0.005, xmax_raw + 0.005), "BMI (original scale)", "BEFORE Standardization (α=1.0)")

xmin_sc = min(X_tr_sc.min(), X_te_sc.min())
xmax_sc = max(X_tr_sc.max(), X_te_sc.max())
_add_lines(ax2b, X_te_sc, y_te.values, res_after, (xmin_sc - 0.1, xmax_sc + 0.1), "BMI (standardized)", "AFTER Standardization (α=1.0)")

fig2.tight_layout()
fig2.savefig(OUT + "fig2_regression_lines.png", bbox_inches="tight")
print("  ✓ fig2_regression_lines.png")

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("Evaluation Metrics — Before vs After Standardization", fontsize=13, fontweight="bold")

short_names = ["Linear", "Ridge", "Lasso"]
x_idx = np.arange(3)
w = 0.38

mse_b = [res_before[n]["mse"] for n in model_keys]
mse_a = [res_after[n]["mse"]  for n in model_keys]
r2_b  = [res_before[n]["r2"]  for n in model_keys]
r2_a  = [res_after[n]["r2"]   for n in model_keys]

ax3a.bar(x_idx - w/2, mse_b, w, label="Before Std", color="#3949AB", alpha=0.87)
ax3a.bar(x_idx + w/2, mse_a, w, label="After Std",  color="#00897B", alpha=0.87)
ax3a.set_xticks(x_idx)
ax3a.set_xticklabels(short_names, fontsize=11)
ax3a.set_ylabel("MSE")
ax3a.set_title("Mean Squared Error (MSE)")
ax3a.legend(fontsize=10)

ax3b.bar(x_idx - w/2, r2_b, w, label="Before Std", color="#3949AB", alpha=0.87)
ax3b.bar(x_idx + w/2, r2_a, w, label="After Std",  color="#00897B", alpha=0.87)
ax3b.set_xticks(x_idx)
ax3b.set_xticklabels(short_names, fontsize=11)
ax3b.set_ylabel("R² Score")
ax3b.set_title("R² Score")
ax3b.legend(fontsize=10)

fig3.tight_layout()
fig3.savefig(OUT + "fig3_metrics_comparison.png", bbox_inches="tight")
print("  ✓ fig3_metrics_comparison.png")

plt.close("all")

print("\n┌─────────────────────────────────────────────────────────┐")
print("│  Coefficient Analysis                                   │")
print("└─────────────────────────────────────────────────────────┘")
print(f"  {'Model':<22}  {'Coef Before':>14}  {'Coef After':>14}")
print("  " + "─" * 54)
for name in model_keys:
    cb = res_before[name]["coef"]
    ca = res_after[name]["coef"]
    print(f"  {name:<22}  {cb:>14.4f}  {ca:>14.4f}")

corr = np.corrcoef(df["bmi"], df["target"])[0, 1]
print(f"\n  • Correlation between BMI and Target (Full Dataset): {corr:.4f}")

print("\n  [!] Note: Since only one feature (BMI) is used, the impact of standardization "
      "\n      is expected to be smaller compared to multi-feature problems.")

print(f"\n  ✓ Analysis complete. All outputs saved to: {OUT}")
print("═" * 65)
