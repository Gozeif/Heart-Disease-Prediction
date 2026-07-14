# Heart Disease Prediction

A machine learning project that predicts the presence of heart disease from patient clinical data. Several classifiers are trained, tuned, and compared, with and without automated feature selection, to find the best-performing model.

## Project structure

```
.
├── 01_eda.ipynb                       # Exploratory data analysis and visualizations (on raw data)
├── 02_preprocessing.ipynb             # Cleans raw data, handles missing/garbage values, scales features
├── 03_modeling.ipynb                  # Model training, tuning, feature selection, evaluation
├── MODELS.md                          # Contains a per-model explanation of how each algorithm works, which hyperparameters are tuned and why
├── requirements.txt                   # Python dependencies
├── Material/
│   ├── train_data.csv                 # Raw training data
│   ├── test_data.csv                  # Raw test data
│   └── Heart Disease Prediction.pdf   # Write-up / report
└── Data/
    ├── Dataset/                       # Preprocessed train/test splits (x_train, x_test, y_train, y_test)
    ├── EDA/                           # Exported EDA plots (class balance, distributions, correlations, etc.)
    ├── Models/                        # Saved trained models (.pkl)
    ├── Summary/                       # Model comparison results, confusion matrices, feature importance plots
    └── Old Data/                      # Earlier/legacy versions of the dataset
```

## Pipeline

1. **EDA** (`01_eda.ipynb`) — runs first, on the raw data. Visualizes class balance, feature distributions, outliers, correlations, and relationships between features and the target. Runs its own light, unscaled cleaning pass (independent of step 2) so plots stay in interpretable units — Age in years, Cholesterol in mg/dL — rather than z-scores. This is where the cleaning/encoding decisions used in preprocessing are informed from.
2. **Preprocessing** (`02_preprocessing.ipynb`) — also starts from the raw data. Drops unused columns, consolidates rare categories, imputes missing values using statistics from the training set only, and scales numeric features with `StandardScaler`. Outputs the train/test splits used for modeling to `Data/Dataset/`.
3. **Modeling** (`03_modeling.ipynb`) — trains and tunes the following classifiers via `GridSearchCV` with stratified 5-fold cross-validation, using the preprocessed data from step 2:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - XGBoost

   Every model is tried with **RFECV** (Recursive Feature Elimination with Cross-Validation) to automatically select the most informative features. RFECV needs an estimator with `feature_importances_` or `coef_` to rank features, which KNN doesn't have — so for KNN a Random Forest is used as a "scout" inside the selector to rank features, while KNN itself still makes the final predictions.

   The Decision Tree is also run with an RF scout, but for a different reason: using the Decision Tree itself as the scout (`DT + RFECV`) picked features poorly and actually *hurt* performance relative to the plain Decision Tree (0.69 vs. 0.85 accuracy — see `Data/Summary/model_summary.csv`). Swapping in a Random Forest scout (`DT + RFECV with RF scout`) — whose feature rankings are more stable since they're averaged across many trees — fixed this and recovered the Decision Tree's original performance. Trained models are saved to `Data/Models/`.

   Note: `01_eda.ipynb` and `02_preprocessing.ipynb` don't depend on each other's code or output — both independently load the raw CSVs. The numbering reflects the conceptual order (EDA informs preprocessing decisions), not a code dependency.

## Results

Models are evaluated on a held-out test set using accuracy, precision, recall, and F1 score (see `Data/Summary/model_summary.csv`). The top performer is **Random Forest**, at roughly 89% accuracy, 95% precision, and 79% recall.

Full results, confusion matrices, and feature importance breakdowns are in `Data/Summary/`.

## Setup

```bash
pip install -r requirements.txt
```

Run the notebooks in order: `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_modeling.ipynb`.
