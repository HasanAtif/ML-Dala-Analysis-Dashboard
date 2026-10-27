# Universal Data Analysis Dashboard v15

An end-to-end machine learning dashboard for uploading any CSV/Excel file, exploring data, engineering features, training models, and making predictions — all from a single Streamlit interface.

---

## Table of Contents

1. [About This App](#about-this-app)
2. [Installation and Running](#installation-and-running)
3. [Sidebar Controls](#sidebar-controls)
4. [Tab-by-Tab Guide](#tab-by-tab-guide)
   - [Tab 0: Correlation & Variable Selection](#tab-0-correlation--variable-selection)
   - [Tab 1: Data Cleaning & Outlier Detection](#tab-1-data-cleaning--outlier-detection)
   - [Tab 2: Statistical Tests](#tab-2-statistical-tests)
   - [Tab 3: Visualizations](#tab-3-visualizations)
   - [Tab 4: Feature Engineering](#tab-4-feature-engineering)
   - [Tab 5: Feature Importance](#tab-5-feature-importance)
   - [Tab 6: Model Training & Evaluation](#tab-6-model-training--evaluation)
   - [Tab 7: Predict New Values](#tab-7-predict-new-values)
   - [Tab 8: Data Profiling](#tab-8-data-profiling)
5. [Encoding Flow — How It Works](#encoding-flow--how-it-works)
6. [What's New in V15](#whats-new-in-v15)
7. [Glossary of All Terms](#glossary-of-all-terms)
8. [Tips and Best Practices](#tips-and-best-practices)
9. [FAQ / Troubleshooting](#faq--troubleshooting)

---

## About This App

The Universal Data Analysis Dashboard is a comprehensive, no-code ML tool built with Streamlit. It handles the full data science pipeline:

| Tab | Name | Purpose |
|-----|------|---------|
| 0 | Correlation & Variable Selection | Correlation matrices, VIF, suggested drops |
| 1 | Data Cleaning & Outlier Detection | Missing values, IQR outliers, export |
| 2 | Statistical Tests | Chi-Square, T-Test, ANOVA, KS, Mutual Information, correlations (10+ tests) |
| 3 | Visualizations | Target distributions, feature histograms, pairplots |
| 4 | Feature Engineering | Encoding preview, apply encoding, scaling preview, class imbalance analysis |
| 5 | Feature Importance | Random Forest + XGBoost importance, 5-fold cross-validation |
| 6 | Model Training & Evaluation | Train 6 models (LR, RF, SVM, XGBoost, GB, KNN), comparison, ROC, residuals |
| 7 | Predict New Values | Single or batch prediction from trained models |
| 8 | Data Profiling | Descriptive statistics, memory usage, data quality score |

**Supported models:** Logistic/Linear Regression, Random Forest, SVM, XGBoost, Gradient Boosting, KNeighbors — for both classification and regression.

---

## Installation and Running

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn xgboost statsmodels imbalanced-learn
```

### Run the app

```bash
streamlit run universal_data_analysis_v15.py
```

The app opens in your browser at `http://localhost:8501`.

### Security

The app binds to **localhost only** by default. No network exposure — only your machine can access it. This is the recommended setting for untrusted networks (coworking spaces, client sites, shared offices).

If you need network access, run Streamlit with the `--server.address` flag:

```bash
streamlit run universal_data_analysis_v15.py --server.address 0.0.0.0 --server.port 8501
```

**Warning:** Binding to `0.0.0.0` exposes the app to all devices on your network. Only do this on trusted networks.

---

## Sidebar Controls

The sidebar (left panel) is your control center. All settings are configured before running analysis.

### Upload Data
- Upload any `.csv` or `.xlsx` file
- The app auto-detects numeric and categorical columns

### Dataset Size
- Row and column counts displayed after upload

### Column Info
- Lists numeric and categorical columns

### Visualization Style
- Choose between Matplotlib (default) and Plotly (interactive) for all charts

### Select Predictor (Target) Variable
- **Target Column:** Dropdown to select the column you want to predict
- **Target Type:** Auto-detected as Classification (categorical or low-cardinality numeric) or Regression (continuous numeric). You can override manually.

### Drop Variables
- Multiselect to remove columns from all analysis

### Feature Encoding
- **Encoding Mode:** Auto (Smart Suggestion) or Manual Selection
- **Auto mode:** Each categorical column is suggested based on cardinality (unique values)
  - ≤25 unique values → One-Hot Encoding
  - 26+ unique values → Label Encoding
  - Very high cardinality → Drop
- **Manual mode:** Bulk-assign columns to One-Hot, Label, Drop, or None (skip)

### Feature Scaling
Choose one scaler for model training:
- StandardScaler (z-score) — default, best for most cases
- MinMaxScaler (0-1) — bounded range
- RobustScaler (median/IQR) — outlier-resistant
- MaxAbsScaler — preserves sparsity
- PowerTransformer (Yeo-Johnson) — Gaussianizes skewed data
- QuantileTransformer (Normal/Uniform) — rank-based
- No Scaling — for tree-based models

### Class Imbalance (Classification only)
- None — no handling
- SMOTE — synthetic minority oversampling (adjustable neighbors)
- Random Oversampling — duplicate minority samples
- Random Undersampling — remove majority samples
- Class Weights Only — adjust loss function weights

---

## Tab-by-Tab Guide

### Tab 0: Correlation & Variable Selection

- Full correlation matrix heatmap (Pearson for numeric, Cramér's V for categorical)
- Correlation with target variable
- Highly correlated feature pairs (configurable threshold)
- Suggested variables to drop (high inter-feature correlation)
- VIF (Variance Inflation Factor) for multicollinearity detection
- **Apply Drop** button — removes selected columns, persists across all tabs

### Tab 1: Data Cleaning & Outlier Detection

- Raw data preview
- Missing values bar chart
- Handle missing values: Mean, Median, Mode, KNN Imputer, or Drop rows
- Cleaned data info and sample
- Column data types
- Export cleaned data as CSV
- Outlier detection using IQR method with box plots

### Tab 2: Statistical Tests

Auto-runs relevant tests based on your data:

**Classification targets:**
- Chi-Square Test of Independence (categorical vs target)
- Cramér's V (effect size)
- T-Test / Mann-Whitney U (numeric vs binary target)
- ANOVA / Kruskal-Wallis (numeric vs multi-class target)
- Kolmogorov-Smirnov Test (distribution comparison)
- Pairwise KS Test (multi-class)
- Mutual Information
- Pearson / Spearman / Kendall correlations

**Regression targets:**
- Pearson / Spearman / Kendall correlations with target
- One-Way ANOVA (categorical vs target)
- Normality tests (Shapiro-Wilk, D'Agostino, Anderson-Darling)
- Durbin-Watson (autocorrelation)
- Breusch-Pagan (homoscedasticity)
- Mutual Information

**Auto-recommendation:** Based on number of tests expected, the app recommends:
- ≤10 tests → Bonferroni correction (conservative)
- >10 tests → Benjamini-Hochberg FDR (recommended)

### Tab 3: Visualizations

- Target variable distribution (bar chart for classification, histogram for regression)
- Feature distributions (histograms with optional KDE)
- Box plots by target class
- Pairplot (select up to 6 features)

### Tab 4: Feature Engineering

**Encoding Preview:** Shows each categorical column with its selected encoding and auto-suggestion. Highlights mismatches.

**Apply Encoding button:** Click to apply selected encoding. This stores encoding metadata in session state for use by Tab 6 and Tab 7. Also provides:
- Encoded data preview
- Download encoded CSV

**Scaling Preview:** Shows what the data looks like after scaling.

**Class Imbalance Analysis:** Visualizes class distribution, shows imbalance ratio, explains what each method does.

### Tab 5: Feature Importance

- Gated behind "Compute Feature Importance" button (prevents accidental long computations)
- Trains Random Forest and XGBoost on the full encoded dataset
- 5-fold cross-validation scores
- Feature importance bar charts
- Uses `Pipeline([scaler, model])` for CV to prevent data leakage

### Tab 6: Model Training & Evaluation

**Configuration:**
- Test size slider (10%-50%)
- Random state input (reproducibility)

**Hyperparameters:** Per-model expander with all adjustable parameters:
- Logistic Regression: C, max_iter, penalty, solver
- Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
- SVM: C, kernel, gamma
- XGBoost: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- Gradient Boosting: n_estimators, max_depth, learning_rate, subsample
- KNN: n_neighbors, weights, metric

**Model Selection:** Multiselect to choose which models to train.

**Results:**
- Model comparison table (accuracy, precision, recall, F1 for classification; MSE, RMSE, MAE, R² for regression)
- Confusion matrices
- ROC curves (multi-class: one-vs-rest)
- Actual vs Predicted scatter plots
- Residual plots
- OLS Regression Summary (statsmodels)
- Regression diagnostics: Q-Q plot, Scale-Location, Cook's Distance, Leverage vs Residuals, VIF

### Tab 7: Predict New Values

- **Single prediction:** Enter feature values in a form, get predicted class + probabilities
- **Batch prediction:** Upload a CSV, get predictions for all rows
- Uses stored encoders and scaler from Tab 6

### Tab 8: Data Profiling

- Descriptive statistics
- Per-column profile (type, non-null, unique, sample values)
- Memory usage
- Data quality score (0-100)
- Duplicate detection

---

## Encoding Flow — How It Works

Understanding encoding is critical because it affects model quality. The app uses a **three-stage encoding pipeline** to prevent data leakage.

### Stage 1: Sidebar — User Configures Encoding

```
User selects encoding mode (Auto/Manual)
         ↓
Per-column encoding methods stored in: encoding_methods dict
Auto-suggestions based on cardinality stored in: encoding_suggestions dict
```

### Stage 2: Tab 4 — User Previews & Applies Encoding

```
User clicks "Apply Encoding to Dataset"
         ↓
App encodes a COPY of the data (df_encoded_preview)
         ↓
Stores METADATA in session state:
  ├── fe_encoding_methods: {col: method_name}     # which method per column
  ├── fe_le_dict: {col: fitted_label_encoder}      # fitted encoders (for reference)
  ├── fe_onehot_prefix_map: {col: [dummy_names]}   # dummy column names
  └── fe_dropped_cols: [col, ...]                   # dropped columns
```

**Important:** The encoded preview is for display/download only. Tab 6 does NOT use this pre-encoded data.

### Stage 3: Tab 6 — Model Training (Correctly Encoded)

```
Raw data loaded (df_model.copy())
         ↓
Columns dropped (fe_dropped_cols applied)
         ↓
Train/Test Split (BEFORE any encoding)
         ↓
    ┌─────────────────────────────────────────┐
    │  ENCODING FIT ON TRAINING DATA ONLY     │
    │                                         │
    │  Label Encoding:                        │
    │    le_new = LabelEncoder()              │
    │    X_train = le_new.fit_transform(...)  │  ← fit on train
    │    X_test  = le_new.transform(...)      │  ← transform test
    │                                         │
    │  One-Hot Encoding:                      │
    │    dummies_train = get_dummies(X_train) │  ← dummies from train
    │    dummies_test  = get_dummies(X_test)  │  ← align to train columns
    │    Missing test columns → filled with 0 │
    └─────────────────────────────────────────┘
         ↓
Class Imbalance (SMOTE/ROS/RUS on train only)
         ↓
Scaling (fit on train, transform both)
         ↓
Model Training (model.fit on train)
```

### Stage 4: Tab 7 — Prediction (Uses Stored Encoders)

```
New raw input received
         ↓
Impute NaN (median/mode from training)
         ↓
Apply stored encoders:
  ├── Label Encoding: le_dict[col].transform(...)    # .transform() only
  └── One-Hot: set matching dummy to 1, rest to 0    # using fe_onehot_prefix_map
         ↓
Apply stored scaler: scaler.transform(...)            # .transform() only
         ↓
Model.predict(...) → output
```

### Why This Matters

| Old approach (V14 and earlier) | New approach (V15) |
|-------------------------------|-------------------|
| Encoder fit on ALL data before split | Encoder fit on TRAIN only after split |
| Test data labels leak into encoder vocabulary | Test data never seen during fit |
| Inflated accuracy metrics | Honest, production-representative metrics |
| Tab 6 re-used pre-encoded data from Tab 4 | Tab 6 re-encodes from scratch using metadata |

---

## What's New in V15

| # | Fix | Description |
|---|-----|-------------|
| 1 | CV data leakage | Tab 5 cross-validation uses `Pipeline([scaler, model])` — scaler is fit inside each CV fold, not before |
| 2 | Drop persistence | Columns dropped in Tab 0 persist across all tabs via `extra_dropped_cols` session state |
| 3 | Encoding ceiling | One-Hot threshold raised from 15→25 unique values; label encoding up to 50 |
| 4 | Hyperparameter crash | `model.fit()` wrapped in try/except; LR penalty/solver validated in real-time |
| 5 | SMOTE guard | Per-class minority count check; auto-reduces k; blocks SMOTE if minority=1 |
| 6 | Tab 5 gating | Entire Tab 5 hidden behind "Compute Feature Importance" button; results cached |
| 7 | Cache optimization | `df_fingerprint()` helper added; all cached functions accept fingerprint param |
| 8 | Shared state | Tab 5 results cached, no redundant retraining |
| 9 | Multiple-testing | Bonferroni + BH-FDR correction toggle in Tab 2 with auto-recommendation |
| 10 | Multiclass tests | T-Test/KS fall back to ANOVA/Kruskal-Wallis + pairwise KS for 3+ classes |
| 11 | Train-only encoding | Tab 6 encoding: split first → fit encoders on train → transform both |
| 12 | Predict NaN | `encode_batch_df` imputes NaN (median/mode) before encoding |
| 13 | Encoder refit | FE path LabelEncoder refit on train only; target LabelEncoder moved after split |
| 14 | Session state keys | `target_col` and `target_type` renamed to `trained_*` to avoid widget key collisions |

---

## Glossary of All Terms

### Data Terms
| Term | Definition |
|------|-----------|
| **Target variable** | The column you want to predict (also called dependent variable, label, y) |
| **Feature** | An input column used to predict the target (also called independent variable, predictor, X) |
| **Categorical feature** | A column with discrete values (e.g., "Red", "Blue", "Green") |
| **Numeric feature** | A column with continuous or integer values (e.g., 1.5, 42) |
| **Cardinality** | Number of unique values in a column |
| **Data leakage** | When information from the test set accidentally enters training, causing inflated metrics |

### Encoding Terms
| Term | Definition |
|------|-----------|
| **Label Encoding** | Maps each category to an integer (e.g., "Red"→0, "Blue"→1). Best for high-cardinality or ordinal features |
| **One-Hot Encoding** | Creates binary (0/1) columns for each category. Best for low-cardinality features (≤25 unique values) |
| **Drop** | Removes the column entirely. Used for high-cardinality features where encoding would create too many columns |
| **None (skip)** | Keeps the column as-is (string). Non-numeric columns are later dropped automatically before model training |
| **Dummy columns** | Binary columns created by One-Hot Encoding (e.g., `Color_Red`, `Color_Blue`) |

### Scaling Terms
| Term | Definition |
|------|-----------|
| **StandardScaler** | Removes mean, scales to unit variance (z-score). Result: mean=0, std=1 |
| **MinMaxScaler** | Scales to [0,1] range. Sensitive to outliers |
| **RobustScaler** | Uses median and IQR. Robust to outliers |
| **MaxAbsScaler** | Scales by maximum absolute value. Preserves sparsity |
| **PowerTransformer** | Gaussianizes features using Yeo-Johnson transform |
| **QuantileTransformer** | Maps to normal or uniform distribution using quantiles |

### Model Terms
| Term | Definition |
|------|-----------|
| **Classification** | Predicting a discrete category (e.g., spam/not spam) |
| **Regression** | Predicting a continuous value (e.g., house price) |
| **Train/Test Split** | Dividing data into training (fit model) and test (evaluate model) sets |
| **Cross-Validation** | Splitting data into K folds, training on K-1, testing on 1, rotating |
| **Hyperparameters** | Settings you configure before training (e.g., n_estimators, learning_rate) |
| **ROC Curve** | Receiver Operating Characteristic curve — plots True Positive Rate vs False Positive Rate |
| **AUC** | Area Under the ROC Curve — 1.0 is perfect, 0.5 is random |
| **Confusion Matrix** | Table showing True Positives, True Negatives, False Positives, False Negatives |

### Statistical Terms
| Term | Definition |
|------|-----------|
| **P-value** | Probability of observing the result if null hypothesis is true. Lower = more significant |
| **Alpha (α)** | Significance threshold (default 0.05). P-value < α → reject null hypothesis |
| **Bonferroni correction** | Adjusts p-values by multiplying by number of tests. Conservative — controls false positives |
| **BH-FDR (Benjamini-Hochberg)** | Adjusts p-values to control false discovery rate. Less conservative, recommended for many tests |
| **Chi-Square** | Tests association between two categorical variables |
| **T-Test** | Tests if means of two groups differ significantly |
| **ANOVA** | Tests if means of 3+ groups differ significantly |
| **Kolmogorov-Smirnov** | Tests if two distributions differ significantly |
| **Mutual Information** | Measures non-linear dependency between variables (0 = independent) |
| **VIF** | Variance Inflation Factor — detects multicollinearity (VIF > 5-10 = concern) |

### Class Imbalance Terms
| Term | Definition |
|------|-----------|
| **SMOTE** | Synthetic Minority Oversampling Technique — creates synthetic minority samples using k-NN |
| **Random Oversampling** | Duplicates random minority class samples |
| **Random Undersampling** | Removes random majority class samples |
| **Class Weights** | Adjusts the loss function to penalize minority class misclassification more |

---

## Tips and Best Practices

### Before Training
1. **Always check Tab 0 first** — use the correlation matrix to identify redundant features. Apply drops before training.
2. **Check encoding suggestions** — the auto-suggestion is based on cardinality. If you disagree, use Manual mode in the sidebar.
3. **Click "Apply Encoding" in Tab 4** — this stores encoding metadata for Tab 6. Without it, Tab 6 uses sidebar defaults.

### Encoding
4. **One-Hot for low cardinality** — columns with ≤25 unique values. Creates fewer dummy columns.
5. **Label for high cardinality** — columns with 26-50 unique values. Maps to integers.
6. **Drop for very high cardinality** — columns with 50+ unique values. Too many dummies would be created.
7. **Encoding is re-applied fresh each training run** — no double-application. Tab 6 always starts from raw data.

### Model Training
8. **Start with default hyperparameters** — only tune if you see underfitting/overfitting.
9. **Use Random State** — set to any integer (default: 42) for reproducible results.
10. **Train multiple models** — select 2-3 models to compare. Random Forest and XGBoost are good defaults.
11. **Check ROC curves** — AUC > 0.8 is good, > 0.9 is excellent, < 0.7 needs improvement.

### Statistical Tests
12. **Use the auto-recommended correction** — BH-FDR for >10 tests, Bonferroni for ≤10.
13. **Focus on "Significant = Yes"** features — these are most predictive of the target.

### Class Imbalance
14. **Try "Class Weights Only" first** — least invasive, no synthetic data.
15. **Use SMOTE if classes are heavily imbalanced** (e.g., 95% vs 5%).
16. **Avoid Random Undersampling on small datasets** — you lose majority class information.

### Predictions
17. **Ensure prediction CSV has the same columns** as training data (minus the target).
18. **NaN values in predictions are imputed** using median (numeric) or mode (categorical) from training data.

---

## FAQ / Troubleshooting

**Q: The app shows "Using encoding from Feature Engineering tab." What does this mean?**
A: You clicked "Apply Encoding" in Tab 4. Tab 6 is using those encoding settings. To use sidebar defaults instead, clear session state by refreshing the page.

**Q: My model shows 100% accuracy. Is something wrong?**
A: Likely data leakage. Check that you're not encoding the target variable before splitting. V15 fixes this automatically.

**Q: SMOTE fails with "k neighbors cannot be greater than..."**
A: The app auto-reduces k. If the smallest class has only 1 sample, SMOTE is blocked — use Random Oversampling instead.

**Q: Prediction fails with "columns not matching" error**
A: Ensure your prediction CSV has the same column names as training data. The app handles missing/extra columns gracefully where possible.

**Q: Can I use this app without clicking "Apply Encoding" in Tab 4?**
A: Yes. Tab 6 will use the sidebar encoding settings directly. Tab 4 is optional — it provides preview and download functionality.

**Q: Why does Tab 5 say "Click to compute"?**
A: Tab 5 is gated behind a button to prevent accidental long computations. Click "Compute Feature Importance" to run it.

**Q: Can I access the app from my phone?**
A: By default, the app binds to localhost only (no network exposure). To enable network access, run with `streamlit run universal_data_analysis_v15.py --server.address 0.0.0.0 --server.port 8501`. Only do this on trusted networks.
