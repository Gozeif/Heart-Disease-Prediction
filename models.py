import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
X_train = pd.read_csv("./Working Data/x_train.csv")
X_test  = pd.read_csv("./Working Data/x_test.csv")
y_train = pd.read_csv("./Working Data/y_train.csv").squeeze()  # squeeze to Series
y_test  = pd.read_csv("./Working Data/y_test.csv").squeeze()


# ─────────────────────────────────────────────
# EVALUATION HELPER
#
# This function takes any trained model and prints
# all required metrics, then plots the confusion matrix.
# By centralising evaluation here, every model gets
# assessed in exactly the same way — no inconsistencies.
# ─────────────────────────────────────────────
def evaluate(name, model, X, y):
    preds = model.predict(X)

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec  = recall_score(y, preds)
    f1   = f1_score(y, preds)
    cm   = confusion_matrix(y, preds)

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}  (of predicted positives, how many were right)")
    print(f"  Recall   : {rec:.4f}  (of actual positives, how many did we catch)")
    print(f"  F1 Score : {f1:.4f}  (harmonic mean of precision & recall)")
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Best params: {model.best_params_}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(4, 3))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f"./Working Data/cm_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()

    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "TN": cm[0,0], "FP": cm[0,1],
            "FN": cm[1,0], "TP": cm[1,1]}


results = []


# ─────────────────────────────────────────────
# MODEL 1 — LOGISTIC REGRESSION
#
# How it works: finds a linear boundary in feature
# space that best separates the two classes. The
# output is a probability, which is then thresholded
# at 0.5 to produce a class label.
#
# Hyperparameters we vary:
#
# C — the inverse of regularisation strength.
#   A small C means strong regularisation, which
#   shrinks coefficients toward zero and prevents
#   overfitting. A large C gives the model more
#   freedom to fit the training data.
#
# solver — the optimisation algorithm used to find
#   the best coefficients. 'lbfgs' and 'liblinear'
#   are both good for small datasets like this one.
# ─────────────────────────────────────────────
print("Training Logistic Regression...")

lr_params = {
    'C':      [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}

lr = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    lr_params,
    cv=5,           # 5-fold cross-validation
    scoring='f1',   # optimise for F1 since we care about both precision and recall
    n_jobs=-1
)
lr.fit(X_train, y_train)
results.append(evaluate("Logistic Regression", lr, X_test, y_test))


# ─────────────────────────────────────────────
# MODEL 2 — DECISION TREE
#
# How it works: recursively splits the data on the
# feature/threshold that best separates the classes,
# forming a tree of if/else rules. Very interpretable
# — you can literally follow the path from root to
# leaf to understand any individual prediction.
#
# Hyperparameters we vary:
#
# max_depth — how deep the tree is allowed to grow.
#   A shallow tree underfits (misses patterns); a
#   very deep tree overfits (memorises training data).
#
# min_samples_split — the minimum number of samples
#   a node must have before it can be split further.
#   Higher values make the tree more conservative and
#   less likely to create splits that only explain
#   one or two training examples.
# ─────────────────────────────────────────────
print("Training Decision Tree...")

dt_params = {
    'max_depth':        [3, 5, 7, 10, None],
    'min_samples_split':[2, 5, 10, 20]
}

dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
dt.fit(X_train, y_train)
results.append(evaluate("Decision Tree", dt, X_test, y_test))


# ─────────────────────────────────────────────
# MODEL 3 — RANDOM FOREST
#
# How it works: builds many decision trees, each
# trained on a random subset of rows (bagging) and
# a random subset of features at each split. The
# final prediction is a majority vote across all
# trees. This reduces the variance that single
# decision trees are prone to.
#
# Hyperparameters we vary:
#
# n_estimators — the number of trees in the forest.
#   More trees generally means better performance up
#   to a point, after which returns diminish. More
#   trees also means longer training time.
#
# max_depth — same concept as in the Decision Tree,
#   but now applied to each individual tree in the
#   forest. Shallower trees mean more bias but less
#   variance — the ensemble compensates for the bias.
# ─────────────────────────────────────────────
print("Training Random Forest...")

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [3, 5, 10, None]
}

rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
rf.fit(X_train, y_train)
results.append(evaluate("Random Forest", rf, X_test, y_test))


# ─────────────────────────────────────────────
# MODEL 4 (BONUS) — XGBOOST
#
# How it works: like Random Forest, XGBoost builds
# many trees — but instead of building them in
# parallel independently, it builds them sequentially.
# Each new tree focuses on correcting the errors the
# previous trees made. This is called gradient
# boosting, and it tends to produce very accurate
# results on structured/tabular data.
#
# Hyperparameters we vary:
#
# n_estimators — number of boosting rounds (trees).
#   Unlike Random Forest, more rounds can lead to
#   overfitting, so this needs to be tuned carefully.
#
# learning_rate — how much each new tree's contribution
#   is scaled down before being added to the ensemble.
#   A lower rate means each tree contributes less,
#   requiring more trees to reach the same performance
#   — but typically generalises better.
#
# max_depth — depth of each individual tree. XGBoost
#   trees are typically kept shallow (3-6) because
#   the boosting process itself adds complexity.
# ─────────────────────────────────────────────
print("Training XGBoost...")

xgb_params = {
    'n_estimators':  [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth':     [3, 5, 7]
}

xgb = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    xgb_params,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
xgb.fit(X_train, y_train)
results.append(evaluate("XGBoost", xgb, X_test, y_test))


# ─────────────────────────────────────────────
# MODEL 5 (BONUS) — K-NEAREST NEIGHBORS (KNN)
#
# How it works: no training in the traditional sense.
# Instead, when predicting a new patient, KNN finds
# the k most similar patients in the training set
# (using distance in feature space) and takes a
# majority vote of their labels. It's entirely
# instance-based — the "model" is just the training
# data itself.
#
# This is the primary reason StandardScaler was applied
# in preprocessing. Without scaling, Age (range ~50)
# and Cholesterol (range ~200+) would completely
# dominate the distance calculation, drowning out
# binary features like Gender or Exercise Angina.
# Scaling puts all features on equal footing.
#
# Hyperparameters we vary:
#
# n_neighbors (k) — how many neighbours vote on the
#   prediction. Small k = sensitive to noise (overfits);
#   large k = smoother boundary (underfits). Finding
#   the right k is the central tuning challenge in KNN.
#
# metric — the distance function used to measure
#   similarity. 'euclidean' is straight-line distance.
#   'manhattan' sums absolute differences dimension by
#   dimension. On high-dimensional data, manhattan
#   often generalises better than euclidean.
# ─────────────────────────────────────────────
print("Training KNN...")

knn_params = {
    'n_neighbors': [3, 5, 7, 11, 15],
    'metric':      ['euclidean', 'manhattan']
}

knn = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
knn.fit(X_train, y_train)
results.append(evaluate("KNN", knn, X_test, y_test))


# ─────────────────────────────────────────────
# SUMMARY TABLE
#
# A side-by-side comparison of all models makes it
# easy to identify which performed best and where
# each model's strengths and weaknesses lie.
# ─────────────────────────────────────────────
print("\n\n" + "="*55)
print("  MODEL COMPARISON SUMMARY")
print("="*55)

summary = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall", "F1"]]
summary = summary.sort_values("F1", ascending=False).reset_index(drop=True)
print(summary.to_string(index=False))

summary.to_csv("./Working Data/model_summary.csv", index=False)
print("\nSummary saved to ./Working Data/model_summary.csv")
print("Confusion matrix plots saved to ./Working Data/")