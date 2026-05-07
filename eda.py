import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# EDA should be done on the ORIGINAL unscaled data
# so that axis values are interpretable (e.g. Age in
# years, Cholesterol in mg/dL — not z-scores).
# We re-run preprocessing up to (but not including)
# the scaling step.
# ─────────────────────────────────────────────

train = pd.read_csv("./Material/train_data.csv")
test  = pd.read_csv("./Material/test_data.csv")

# Quick preprocessing to get a clean, unscaled dataframe for EDA
def prep_for_eda(df, is_train=True, fit_vals={}):
    df = df.copy()
    df = df.drop(columns=['id'], errors='ignore')
    if is_train:
        df = df.drop_duplicates()
    df["work_type"] = df["work_type"].replace(["children", "Never_worked"], "Unknown")
    if is_train:
        fit_vals["age_median"]  = df["Age"].median()
        fit_vals["gender_mode"] = df["Gender"].mode()[0]
    df["Age"]            = df["Age"].fillna(round(fit_vals["age_median"])).astype(int)
    df["Gender"]         = df["Gender"].fillna(fit_vals["gender_mode"])
    df["work_type"]      = df["work_type"].fillna("Unknown")
    df["smoking_status"] = df["smoking_status"].fillna("Unknown")
    df["Heart Disease"]  = df["Heart Disease"].map({"Yes": 1, "No": 0})
    return df

fit_vals = {}
df = prep_for_eda(train, is_train=True, fit_vals=fit_vals)

# Colour palette — consistent throughout
PALETTE   = {0: "#4A90D9", 1: "#E05C5C"}   # blue = no disease, red = disease
LABELS    = {0: "No Disease", 1: "Disease"}
STYLE_KW  = dict(edgecolor='white', linewidth=0.6)
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":     11,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
})

SAVE = "./Working Data/EDA/"


# Plot 1: Class Balance
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["Heart Disease"].value_counts().sort_index()
bars = ax.bar(
    [LABELS[i] for i in counts.index],
    counts.values,
    color=[PALETTE[i] for i in counts.index],
    width=0.5,
    **STYLE_KW
)
for bar, count in zip(bars, counts.values):
    pct = count / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 2,
            f"{count} ({pct:.1f}%)",
            ha='center', va='bottom', fontsize=9)

ax.set_title("Target Class Distribution", fontweight='bold', pad=12)
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.2)
plt.tight_layout()
plt.savefig(f"{SAVE}eda_1_class_balance.png", dpi=150)
plt.close()
print("Saved: eda_1_class_balance.png")


# Plot 2: Numeric Feature Distribution by Class
# Shows how each continuous feature separates the
# two classes — the more separated the peaks, the
# more predictive the feature.
numeric_cols = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    for label, group in df.groupby("Heart Disease"):
        ax.hist(group[col].dropna(), bins=20, alpha=0.6,
                color=PALETTE[label], label=LABELS[label],
                **STYLE_KW)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

# Hide unused subplot
axes[-1].set_visible(False)

fig.suptitle("Numeric Feature Distributions by Class", fontsize=13,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{SAVE}eda_2_numeric_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_2_numeric_distributions.png")


# ─────────────────────────────────────────────
# PLOT 3 — BOX PLOTS (NUMERIC FEATURES BY CLASS)
# Complements the histograms by showing median,
# spread, and outliers side by side per class.
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    data_by_class = [df[df["Heart Disease"] == label][col].dropna()
                     for label in [0, 1]]
    bp = ax.boxplot(data_by_class,
                    patch_artist=True,
                    widths=0.4,
                    medianprops=dict(color='white', linewidth=2))
    for patch, label in zip(bp['boxes'], [0, 1]):
        patch.set_facecolor(PALETTE[label])
        patch.set_alpha(0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([LABELS[0], LABELS[1]])
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel("Value")

axes[-1].set_visible(False)

fig.suptitle("Numeric Features — Box Plots by Class", fontsize=13,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{SAVE}eda_3_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_3_boxplots.png")


# ─────────────────────────────────────────────
# PLOT 4 — CORRELATION HEATMAP
# Shows linear relationships between numeric
# features and the target. Highly correlated
# feature pairs can cause multicollinearity issues
# in Logistic Regression specifically.
# ─────────────────────────────────────────────
numeric_for_corr = numeric_cols + ["Heart Disease",
                                    "Chest pain type", "FBS over 120",
                                    "EKG results", "Exercise angina",
                                    "Slope of ST", "Number of vessels fluro",
                                    "Thallium"]

corr = df[numeric_for_corr].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5,
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Feature Correlation Matrix", fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f"{SAVE}eda_4_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: eda_4_correlation_heatmap.png")


# ─────────────────────────────────────────────
# PLOT 5 — CATEGORICAL FEATURES VS TARGET
# For each categorical feature, shows the count
# of disease vs no-disease within each category.
# Reveals which category values are highest-risk.
# ─────────────────────────────────────────────
cat_cols = {
    "Chest pain type":         {1: "Typical angina", 2: "Atypical angina",
                                 3: "Non-anginal pain", 4: "Asymptomatic"},
    "Thallium":                {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
    "Slope of ST":             {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "Number of vessels fluro": {0: "0", 1: "1", 2: "2", 3: "3"},
    "EKG results":             {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "Exercise angina":         {0: "No", 1: "Yes"},
    "FBS over 120":            {0: "No", 1: "Yes"},
    "Gender":                  {"Female": "Female", "Male": "Male"},
}

fig, axes = plt.subplots(4, 2, figsize=(14, 18))
axes = axes.flatten()

for i, (col, label_map) in enumerate(cat_cols.items()):
    ax = axes[i]
    ct = df.groupby([col, "Heart Disease"]).size().unstack(fill_value=0)
    ct.index = [label_map.get(k, str(k)) for k in ct.index]
    ct.columns = [LABELS[c] for c in ct.columns]

    x = np.arange(len(ct))
    width = 0.35
    for j, (disease_label, color) in enumerate(zip(ct.columns, PALETTE.values())):
        bars = ax.bar(x + j*width, ct[disease_label],
                      width=width, color=color, label=disease_label,
                      alpha=0.85, **STYLE_KW)

    ax.set_xticks(x + width/2)
    ax.set_xticklabels(ct.index, rotation=15, ha='right')
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

fig.suptitle("Categorical Features vs Heart Disease", fontsize=13,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{SAVE}eda_5_categorical_vs_target.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_5_categorical_vs_target.png")


# ─────────────────────────────────────────────
# PLOT 6 — SMOKING STATUS & WORK TYPE VS TARGET
# Separated from plot 5 since these have more
# categories and need more space to be readable.
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col in zip(axes, ["smoking_status", "work_type"]):
    ct = df.groupby([col, "Heart Disease"]).size().unstack(fill_value=0)
    ct.columns = [LABELS[c] for c in ct.columns]
    x = np.arange(len(ct))
    width = 0.35
    for j, (disease_label, color) in enumerate(zip(ct.columns, PALETTE.values())):
        ax.bar(x + j*width, ct[disease_label],
               width=width, color=color, label=disease_label,
               alpha=0.85, **STYLE_KW)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(ct.index, rotation=15, ha='right')
    ax.set_title(col.replace("_", " ").title(), fontweight='bold')
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

fig.suptitle("Lifestyle Features vs Heart Disease", fontsize=13,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{SAVE}eda_6_lifestyle_vs_target.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_6_lifestyle_vs_target.png")


# ─────────────────────────────────────────────
# PLOT 7 — FEATURE IMPORTANCE PROXY
# Uses correlation with the target as a rough
# proxy for feature importance — useful for the
# report's discussion of which features matter most.
# Note: this is a linear measure; tree-based models
# may rank features differently.
# ─────────────────────────────────────────────
corr_with_target = df[numeric_for_corr].corr()["Heart Disease"].drop("Heart Disease")
corr_with_target = corr_with_target.sort_values(key=abs, ascending=True)

colors = ["#E05C5C" if v > 0 else "#4A90D9" for v in corr_with_target]

fig, ax = plt.subplots(figsize=(7, 6))
bars = ax.barh(corr_with_target.index, corr_with_target.values,
               color=colors, alpha=0.85, **STYLE_KW)
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xlabel("Correlation with Heart Disease")
ax.set_title("Feature Correlation with Target\n(red = positive, blue = negative)",
             fontweight='bold')
for bar, val in zip(bars, corr_with_target.values):
    ax.text(val + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height()/2,
            f"{val:.2f}",
            va='center',
            ha='left' if val >= 0 else 'right',
            fontsize=7)

plt.tight_layout()
plt.savefig(f"{SAVE}eda_7_feature_importance_proxy.png", dpi=150)
plt.close()
print("Saved: eda_7_feature_importance_proxy.png")


# ─────────────────────────────────────────────
# PRINTED SUMMARY STATS
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("  DATASET SUMMARY")
print("="*50)
print(f"  Total samples : {len(df)}")
print(f"  Features      : {df.shape[1] - 1}")
print(f"  Disease cases : {df['Heart Disease'].sum()} ({df['Heart Disease'].mean()*100:.1f}%)")
print(f"  No disease    : {(df['Heart Disease']==0).sum()} ({(df['Heart Disease']==0).mean()*100:.1f}%)")

print("\n  Numeric feature stats by class:")
print(df.groupby("Heart Disease")[numeric_cols].mean().round(2).rename(index=LABELS).to_string())

print("\n  Top correlated features with target:")
full_corr = df[numeric_for_corr].corr()["Heart Disease"].drop("Heart Disease")
print(full_corr.abs().sort_values(ascending=False).round(3).to_string())

print("\nAll EDA plots saved to ./Working Data/")
