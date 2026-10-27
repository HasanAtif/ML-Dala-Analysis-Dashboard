# Universal Data Analysis Dashboard v15 — Enhanced
# v15 fixes: data leakage, drop persistence, model-aware encoding,
#   hyperparameter validation, SMOTE guard, gated Tab 5, cache optimization,
#   shared model state, multiple-testing correction, multiclass tests,
#   train-only encoding, NaN handling in predictions, encoder refit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, PowerTransformer, QuantileTransformer,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def show_fig(fig, height_min=4):
    w, h = fig.get_size_inches()
    if h < height_min:
        fig.set_size_inches(w, height_min)
    st.pyplot(fig)
    plt.close(fig)

def get_palette(name="default"):
    palettes = {
        "default": ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c",
                     "#e67e22", "#34495e", "#e91e63", "#00bcd4"],
        "viridis": "viridis", "plasma": "plasma", "coolwarm": "coolwarm",
        "Set1": "Set1", "Set2": "Set2", "Paired": "Paired",
        "Dark2": "Dark2", "tab10": "tab10", "tab20": "tab20",
    }
    return palettes.get(name, palettes["default"])

def apply_viz_style():
    theme = st.session_state.get("viz_theme", "whitegrid")
    if theme != "default":
        sns.set_theme(style=theme)
    else:
        sns.set_theme(style="whitegrid")

def get_plotly_palette():
    name = st.session_state.get("viz_palette", "default")
    mapping = {
        "default": px.colors.qualitative.Plotly,
        "viridis": px.colors.sequential.Viridis,
        "plasma": px.colors.sequential.Plasma,
        "coolwarm": px.colors.diverging.RdBu,
        "Set1": px.colors.qualitative.Set1,
        "Set2": px.colors.qualitative.Set2,
        "Paired": px.colors.qualitative.Set3,
        "Dark2": px.colors.qualitative.Dark2,
        "tab10": px.colors.qualitative.D3,
        "tab20": px.colors.qualitative.Alphabet,
    }
    return mapping.get(name, px.colors.qualitative.Plotly)

def test_interpreter(test_name, p_value, alpha=0.05, context=""):
    significant = p_value < alpha
    if significant:
        base = f"**{test_name}:** Statistically significant (p={p_value:.4f} < α={alpha}). "
    else:
        base = f"**{test_name}:** Not statistically significant (p={p_value:.4f} ≥ α={alpha}). "
    interpretations = {
        "Chi-Square": "This means the categorical feature and the target are associated (not independent).",
        "T-Test": "The two groups have significantly different means for this feature.",
        "Mann-Whitney U": "The two groups have significantly different distributions (non-parametric test).",
        "Welch's T-Test": "The two groups have significantly different means (does not assume equal variances).",
        "ANOVA": "At least one group mean differs significantly from the others.",
        "Kruskal-Wallis": "At least one group distribution differs significantly (non-parametric ANOVA).",
        "Pearson": "There is a significant linear correlation between this feature and the target.",
        "Spearman": "There is a significant monotonic relationship between this feature and the target.",
        "Kendall": "There is a significant ordinal association between this feature and the target.",
        "KS": "The two groups have significantly different distributions.",
        "Shapiro-Wilk": "The data does not appear to be normally distributed.",
        "Breusch-Pagan": "The variance of residuals is not constant (heteroscedasticity present).",
    }
    interpretation = interpretations.get(test_name, "See the test statistic and p-value above.")
    recommendation = ""
    if significant and test_name in ["Chi-Square", "T-Test", "Mann-Whitney U", "Welch's T-Test",
                                       "ANOVA", "Kruskal-Wallis", "Pearson", "Spearman", "Kendall"]:
        recommendation = " **Recommendation:** This feature shows a strong relationship with the target — consider keeping it."
    elif not significant and test_name in ["Chi-Square", "T-Test", "Mann-Whitney U", "Welch's T-Test",
                                             "ANOVA", "Kruskal-Wallis", "Pearson", "Spearman", "Kendall"]:
        recommendation = " **Recommendation:** This feature may not be useful for prediction — consider dropping it."
    return base + interpretation + recommendation

def dynamic_fig_size(n_items, base_height=6, base_width=10, per_item=0.3, max_h=20, max_w=24):
    h = min(max_h, max(base_height, n_items * per_item))
    w = min(max_w, base_width)
    return w, h

def df_fingerprint(df):
    """Lightweight hash of a DataFrame for cache keys — avoids hashing entire dataset."""
    return (
        df.shape,
        tuple(df.columns),
        tuple(df.dtypes.astype(str)),
        int(pd.util.hash_pandas_object(df).sum()),
    )

def get_scaler(name):
    """Return a fresh scaler instance by name (never reuses a fitted object)."""
    _map = {
        "StandardScaler (z-score)": lambda: StandardScaler(),
        "MinMaxScaler (0-1)": lambda: MinMaxScaler(),
        "RobustScaler (median/IQR)": lambda: RobustScaler(),
        "MaxAbsScaler": lambda: MaxAbsScaler(),
        "PowerTransformer (Yeo-Johnson)": lambda: PowerTransformer(method="yeo-johnson"),
        "QuantileTransformer (Normal)": lambda: QuantileTransformer(output_distribution="normal", random_state=42),
        "QuantileTransformer (Uniform)": lambda: QuantileTransformer(output_distribution="uniform", random_state=42),
    }
    factory = _map.get(name)
    return factory() if factory else None

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
def suggest_encoding(series, col_name):
    """Suggest encoding method based on cardinality, type, and text analysis."""
    n_unique = series.nunique()
    dtype = series.dtype
    total_rows = len(series.dropna())

    if dtype in ["float64", "int64"]:
        return "None (numeric)"
    if dtype == "bool":
        return "Label Encoding"

    # Detect text/description fields: long strings or near-unique values
    if dtype == "object" and total_rows > 0:
        avg_len = series.astype(str).str.len().mean()
        uniqueness = n_unique / total_rows
        if avg_len > 30:
            return "Drop (long text)"
        if uniqueness > 0.8 and n_unique > 50:
            return "Drop (high cardinality)"

    if n_unique == 2:
        return "Label Encoding"
    if n_unique <= 25:
        return "One-Hot Encoding"
    if n_unique <= 50:
        return "Label Encoding"
    if n_unique > 50:
        return "Drop (high cardinality)"
    return "Label Encoding"

def apply_encoding(df, col, method):
    """Apply selected encoding to a column."""
    if method == "None (numeric)":
        return df, None
    if method == "Label Encoding":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        return df, le
    if method == "One-Hot Encoding":
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df, "onehot"
    return df, None


def fit_label_encoder_safe(X_train_col, X_test_col):
    """Fit LabelEncoder on train, add __unknown__ sentinel, transform both.

    Prevents ValueError when test set contains categories not seen in training.
    Unseen test values are mapped to a designated __unknown__ class.
    Classes are re-sorted after appending sentinel to preserve np.searchsorted invariant.
    """
    le = LabelEncoder()
    X_train_encoded = le.fit_transform(X_train_col.astype(str))
    # Add __unknown__ sentinel for unseen test categories
    le_classes = set(le.classes_)
    test_values = set(X_test_col.astype(str))
    unseen = test_values - le_classes
    if unseen:
        le.classes_ = np.sort(np.append(le.classes_, "__unknown__"))
        X_test_mapped = X_test_col.astype(str).where(
            ~X_test_col.astype(str).isin(unseen), "__unknown__"
        )
        X_test_encoded = le.transform(X_test_mapped)
    else:
        X_test_encoded = le.transform(X_test_col.astype(str))
    return le, X_train_encoded, X_test_encoded


def transform_with_unknown(le, values):
    """Transform values using LabelEncoder, mapping unseen to __unknown__ sentinel.

    Safe for production use — handles novel categories without crashing.
    """
    values_str = values.astype(str)
    known_classes = set(le.classes_)
    unseen_mask = ~values_str.isin(known_classes)
    if unseen_mask.any():
        if "__unknown__" not in le.classes_:
            le.classes_ = np.sort(np.append(le.classes_, "__unknown__"))
        values_str = values_str.where(~unseen_mask, "__unknown__")
    return le.transform(values_str)

# ---------------------------------------------------------------------------
# Cached computation functions
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def compute_correlation_matrix(data_fingerprint, _data_dict):
    df_corr = pd.DataFrame(_data_dict)
    return df_corr.corr()

@st.cache_data(show_spinner=False, ttl=3600)
def compute_data_profiling(data_fingerprint, _df_hashable):
    df_prof = pd.DataFrame(_df_hashable)
    profile_data = []
    for col in df_prof.columns:
        n_unique = df_prof[col].nunique()
        n_missing = df_prof[col].isnull().sum()
        pct_missing = round(100 * n_missing / len(df_prof), 2) if len(df_prof) > 0 else 0
        dtype = str(df_prof[col].dtype)
        info = {
            "Column": col, "Dtype": dtype, "Non-Null": len(df_prof) - n_missing,
            "Missing": n_missing, "Missing %": pct_missing, "Unique": n_unique,
        }
        if df_prof[col].dtype in ["float64", "int64"]:
            info["Mean"] = round(df_prof[col].mean(), 3) if df_prof[col].notna().any() else None
            info["Std"] = round(df_prof[col].std(), 3) if df_prof[col].notna().any() else None
            info["Min"] = df_prof[col].min()
            info["Q1"] = df_prof[col].quantile(0.25)
            info["Median"] = df_prof[col].median()
            info["Q3"] = df_prof[col].quantile(0.75)
            info["Max"] = df_prof[col].max()
            info["Skewness"] = round(df_prof[col].skew(), 3) if df_prof[col].notna().any() else None
            info["Kurtosis"] = round(df_prof[col].kurtosis(), 3) if df_prof[col].notna().any() else None
        else:
            mode_val = df_prof[col].mode()
            info["Mode"] = mode_val[0] if len(mode_val) > 0 else None
            info["Top 5"] = str(df_prof[col].value_counts().head(5).to_dict())
        profile_data.append(info)
    return pd.DataFrame(profile_data)

@st.cache_data(show_spinner=False, ttl=3600)
def compute_vif(data_fingerprint, _data_dict):
    df_vif = pd.DataFrame(_data_dict)
    vif_data = []
    for i, col in enumerate(df_vif.columns):
        try:
            vif_val = variance_inflation_factor(df_vif.values, i)
            vif_data.append({"Feature": col, "VIF": round(vif_val, 2)})
        except Exception:
            vif_data.append({"Feature": col, "VIF": np.nan})
    return pd.DataFrame(vif_data)

@st.cache_data(show_spinner=False, ttl=3600)
def compute_pearson_correlations(data_fingerprint, _feat_numeric_list, _target_col, _data_hashable):
    df_tmp = pd.DataFrame(_data_hashable)
    results = []
    for col in _feat_numeric_list:
        try:
            valid = df_tmp[[col, _target_col]].dropna()
            if len(valid) < 3:
                continue
            corr, p_val = stats.pearsonr(valid[col], valid[_target_col])
            results.append({
                "Feature": col, "Pearson r": round(corr, 4),
                "P-Value": round(p_val, 6), "Significant": "Yes",
            })
        except Exception:
            continue
    return results

@st.cache_data(show_spinner=False, ttl=3600)
def compute_spearman_correlations(data_fingerprint, _feat_numeric_list, _target_col, _data_hashable):
    df_tmp = pd.DataFrame(_data_hashable)
    results = []
    for col in _feat_numeric_list:
        try:
            valid = df_tmp[[col, _target_col]].dropna()
            if len(valid) < 3:
                continue
            if valid[col].std() == 0 or valid[_target_col].std() == 0:
                continue
            rho, p_val = stats.spearmanr(valid[col], valid[_target_col])
            results.append({
                "Feature": col, "Spearman rho": round(rho, 4),
                "P-Value": round(p_val, 6), "Significant": "Yes",
            })
        except Exception:
            continue
    return results

@st.cache_data(show_spinner=False, ttl=3600)
def compute_kendall_correlations(data_fingerprint, _feat_numeric_list, _target_col, _data_hashable):
    df_tmp = pd.DataFrame(_data_hashable)
    results = []
    for col in _feat_numeric_list:
        try:
            valid = df_tmp[[col, _target_col]].dropna()
            if len(valid) < 3:
                continue
            tau, p_val = stats.kendalltau(valid[col], valid[_target_col])
            results.append({
                "Feature": col, "Kendall tau": round(tau, 4),
                "P-Value": round(p_val, 6), "Significant": "Yes",
            })
        except Exception:
            continue
    return results

# ---------------------------------------------------------------------------
# Page configuration & CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Universal Data Analysis v15", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetric"] { border-radius: 10px; padding: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: rgba(52,152,219,0.15) !important; border-bottom: 3px solid #3498db !important; }
    .stButton > button { border-radius: 8px; border: 1px solid rgba(52,152,219,0.3); }
    .stButton > button:hover { border-color: #3498db; }
    .stDownloadButton > button { background: linear-gradient(135deg, #3498db, #2980b9); color: white; border: none; }
    .stDownloadButton > button:hover { background: linear-gradient(135deg, #2980b9, #1f6dad); }
</style>
""", unsafe_allow_html=True)

st.title("Universal Data Analysis Dashboard v15")
st.markdown("Upload any CSV or Excel file for automatic analysis")

# About
import os as _os
_manual_name = "universal_data_analysis_v15_README.md"
_manual_candidates = [
    _os.path.join(_os.getcwd(), _manual_name),
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _manual_name),
    _os.path.join(_os.path.dirname(__file__), _manual_name),
]
_manual_path = None
for _p in _manual_candidates:
    if _os.path.exists(_p):
        _manual_path = _p
        break

with st.expander("About This App — Click to Expand", expanded=False):
    if _manual_path:
        try:
            with open(_manual_path, "r", encoding="utf-8") as _f:
                _manual_md = _f.read()
            st.markdown(_manual_md, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not read user manual: {e}")
    else:
        st.markdown("""
## What Does This App Do?

This is an **all-in-one data analysis and machine learning dashboard**. You upload a dataset, and the app automatically:

1. **Profiles your data** — column types, missing values, statistics, skewness, kurtosis, data quality scores
2. **Cleans your data** — handles missing values (median/mean/mode), detects outliers (IQR), exports cleaned CSV
3. **Runs statistical tests** — 10+ hypothesis tests (Chi-Square, T-Test, ANOVA, Spearman, Kendall, KS, Shapiro-Wilk, Breusch-Pagan, Durbin-Watson, Mutual Information, Cramer's V)
4. **Visualizes patterns** — distributions, correlations, box plots, pairplots with **toggleable Matplotlib (static) or Plotly (interactive)** charts
5. **Encodes & scales features** — One-Hot, Label, or Auto encoding with smart suggestions; train-only fit (no data leakage); StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
6. **Handles class imbalance** — SMOTE, Random Oversampling, Undersampling, or class weights
7. **Builds ML models** — trains 6 models (Logistic/Linear Regression, Random Forest, SVM, XGBoost, Gradient Boosting, KNN) with full hyperparameter control and 5-fold cross-validation
8. **Diagnoses regressions** — OLS summary, Q-Q plot, Cook's distance, leverage plot, VIF (via statsmodels)
9. **Makes predictions** — enter values using original column names or upload CSV for batch predictions; encoding applied automatically

> *Place `universal_data_analysis_v15_README.md` in the same folder as the script for the full handbook.*
""")

# ===========================================================================
# FILE UPLOAD
# ===========================================================================
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel", type=["csv", "xlsx", "xls"],
    key="file_uploader",
    help="Supported formats: CSV (.csv), Excel (.xlsx, .xls). Max 2GB."
)

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb == 0:
        st.error("The uploaded file is empty. Please upload a valid data file.")
        st.stop()
    try:
        if uploaded_file.name.endswith(".csv"):
            for enc in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            else:
                st.error("Could not read the CSV file. It may be corrupt or use an unsupported encoding.")
                st.stop()
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
    if df.empty:
        st.error("The file was read but contains no data rows. Please check your file.")
        st.stop()
    st.sidebar.success(f"Loaded: {uploaded_file.name} ({file_size_mb:.1f} MB, {df.shape[0]} rows x {df.shape[1]} cols)")
else:
    st.info("Please upload a CSV or Excel file from the sidebar to begin analysis.")
    st.stop()

# --- Dataset Sampling ---
SAMPLE_THRESHOLD = 10000
if len(df) > SAMPLE_THRESHOLD:
    st.sidebar.header("Dataset Size")
    st.sidebar.warning(f"Large dataset: {len(df):,} rows")
    use_sampling = st.sidebar.checkbox("Sample for faster analysis", value=True, key="use_sampling")
    sample_size = st.sidebar.slider(
        "Sample size (rows)", 1000, min(50000, len(df)),
        min(SAMPLE_THRESHOLD, len(df)), step=1000, key="sample_size"
    )
    if use_sampling:
        df_sample = df.sample(n=sample_size, random_state=42)
        st.sidebar.info(f"Using {len(df_sample):,} of {len(df):,} rows for analysis")
    else:
        df_sample = df.copy()
else:
    df_sample = df.copy()

# --- Auto-detect column types ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

st.sidebar.header("Column Info")
st.sidebar.write(f"**Rows:** {df.shape[0]}")
st.sidebar.write(f"**Columns:** {df.shape[1]}")
st.sidebar.write(f"**Numeric:** {len(numeric_cols)}")
st.sidebar.write(f"**Categorical:** {len(categorical_cols)}")

# --- Visualization Style ---
st.sidebar.header("Visualization Style")
viz_palette = st.sidebar.selectbox(
    "Color Palette",
    ["default", "viridis", "plasma", "coolwarm", "Set1", "Set2", "Paired", "Dark2", "tab10", "tab20"],
    index=0, key="viz_palette"
)
viz_theme = st.sidebar.selectbox(
    "Plot Theme",
    ["default", "darkgrid", "whitegrid", "dark", "white", "ticks"],
    index=1, key="viz_theme"
)
viz_height = st.sidebar.slider("Default plot height", 4, 12, 6, 1, key="viz_height")
viz_width = st.sidebar.slider("Default plot width", 6, 20, 10, 1, key="viz_width")
plot_backend = st.sidebar.radio(
    "Plot backend",
    ["Matplotlib (static)", "Plotly (interactive)"],
    index=0, key="plot_backend",
    help="Choose static matplotlib or interactive Plotly charts where applicable."
)
use_plotly = plot_backend.startswith("Plotly")

# --- Target Variable ---
st.sidebar.header("Select Predictor (Target) Variable")
all_cols = df.columns.tolist()
target_col = st.sidebar.selectbox("Choose target column", all_cols, key="target_col")

# Auto-detect target type based on the selected predictor variable
n_unique_target = df[target_col].nunique()
if target_col in numeric_cols:
    auto_type = "Classification" if n_unique_target <= 20 else "Regression"
else:
    auto_type = "Classification"
idx = 0 if auto_type == "Classification" else 1
target_type = st.sidebar.radio(
    "Target Type",
    ["Classification", "Regression"],
    index=idx,
    key="target_type",
    help="Auto-detected: Classification for categorical or low-cardinality numeric targets, Regression for continuous numeric targets."
)
if target_type != auto_type:
    st.sidebar.warning(f"Switched from auto-detected '{auto_type}' to '{target_type}'.")

df_model = df.copy()

# Re-apply any columns dropped via Tab 0's "Apply Drop" button (persists across reruns)
if "extra_dropped_cols" in st.session_state:
    _extra = [c for c in st.session_state["extra_dropped_cols"] if c in df.columns]
    if _extra:
        df = df.drop(columns=_extra)
        df_model = df_model.drop(columns=[c for c in _extra if c in df_model.columns])
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        categorical_cols = [c for c in categorical_cols if c in df.columns]

# Display flash message from Apply Drop button
if "drop_flash" in st.session_state:
    st.success(st.session_state.pop("drop_flash"))

feature_cols = [c for c in df.columns if c != target_col]

# --- Drop Variables ---
st.sidebar.header("Drop Variables")
cols_to_drop = st.sidebar.multiselect("Select columns to drop from analysis", options=feature_cols, default=[], key="cols_to_drop")
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    df_model = df_model.drop(columns=[c for c in cols_to_drop if c in df_model.columns])
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

# ===========================================================================
# ENCODING SECTION (Sidebar)
# ===========================================================================
st.sidebar.header("Feature Encoding")
st.sidebar.markdown("Configure how categorical features are encoded.")

encoding_mode = st.sidebar.radio(
    "Encoding Mode",
    ["Auto (Smart Suggestion)", "Manual Selection"],
    index=0, key="encoding_mode",
    help="Auto suggests encoding based on cardinality. Manual lets you bulk-assign encoding per type."
)

encoding_suggestions = {}
if categorical_cols:
    for col in categorical_cols:
        encoding_suggestions[col] = suggest_encoding(df_model[col], col)

# --- Pre-compute smart defaults for multiselect pre-selections ---
suggested_onehot = [col for col, m in encoding_suggestions.items() if m == "One-Hot Encoding"]
suggested_label = [col for col, m in encoding_suggestions.items() if m in ("Label Encoding", "Label Encoding (high cardinality)")]
suggested_drop = [col for col, m in encoding_suggestions.items() if m.startswith("Drop")]
suggested_none = [col for col, m in encoding_suggestions.items() if m.startswith("None")]

encoding_methods = {}
if categorical_cols:
    with st.sidebar.expander("Per-Column Encoding", expanded=False):
        if encoding_mode == "Auto (Smart Suggestion)":
            for col in categorical_cols:
                st.write(f"**{col}** → {encoding_suggestions.get(col, 'Label Encoding')}")
        else:
            st.markdown("Assign columns to encoding type. Each column must appear in exactly **one** category.")
            st.caption("Suggestions based on cardinality are shown as default selections.")

            ohe_cols = st.multiselect(
                "One-Hot Encoding columns",
                options=categorical_cols,
                default=[c for c in suggested_onehot if c in categorical_cols],
                key="manual_ohe_cols",
                help="Creates binary columns for each category. Best for low-cardinality features (≤15 unique values).",
            )

            label_cols = st.multiselect(
                "Label Encoding columns",
                options=[c for c in categorical_cols if c not in ohe_cols],
                default=[c for c in suggested_label if c in categorical_cols and c not in ohe_cols],
                key="manual_label_cols",
                help="Maps each category to an integer. Best for high-cardinality or ordinal features.",
            )

            remaining = [c for c in categorical_cols if c not in ohe_cols and c not in label_cols]
            default_drop = [c for c in suggested_drop if c in remaining]
            drop_cols = st.multiselect(
                "Drop (remove from analysis)",
                options=remaining,
                default=default_drop,
                key="manual_drop_cols",
                help="Text fields, IDs, or high-cardinality columns that cannot be meaningfully encoded.",
            )

            remaining2 = [c for c in remaining if c not in drop_cols]
            default_none = [c for c in suggested_none if c in remaining2]
            none_cols = st.multiselect(
                "None (skip encoding)",
                options=remaining2,
                default=default_none,
                key="manual_none_cols",
                help="These columns will not be encoded but kept as-is.",
            )

            # Conflict check
            all_assigned = set(ohe_cols + label_cols + drop_cols + none_cols)
            unassigned = [c for c in categorical_cols if c not in all_assigned]
            if unassigned:
                st.warning(f"Not assigned to any encoding: {', '.join(unassigned)}. These will default to **Label Encoding**.")
                for c in unassigned:
                    label_cols.append(c)

            if ohe_cols or label_cols or drop_cols or none_cols:
                summary = []
                for c in ohe_cols:
                    encoding_methods[c] = "One-Hot Encoding"
                    summary.append(f"**{c}** → One-Hot")
                for c in label_cols:
                    encoding_methods[c] = "Label Encoding"
                    summary.append(f"**{c}** → Label")
                for c in drop_cols:
                    encoding_methods[c] = "Drop"
                    summary.append(f"**{c}** → Drop")
                for c in none_cols:
                    encoding_methods[c] = "None (skip)"
                    summary.append(f"**{c}** → None")
                st.markdown("---")
                st.markdown("**Assignment Summary:**")
                for s in summary:
                    st.markdown(f"- {s}")

# --- Scaling & Standardization ---
st.sidebar.header("Feature Scaling")
scaling_method = st.sidebar.selectbox(
    "Scaling Method",
    [
        "StandardScaler (z-score)",
        "MinMaxScaler (0-1)",
        "RobustScaler (median/IQR)",
        "MaxAbsScaler",
        "PowerTransformer (Yeo-Johnson)",
        "QuantileTransformer (Normal)",
        "QuantileTransformer (Uniform)",
        "No Scaling",
    ],
    index=0, key="scaling_method",
    help="StandardScaler is best for most cases. RobustScaler handles outliers. MinMaxScaler bounds to [0,1]."
)

# --- Class Imbalance ---
class_imbalance_method = None
smote_neighbors = 5
use_class_weights = False
if target_type == "Classification":
    st.sidebar.header("Class Imbalance")
    st.sidebar.markdown("Handle imbalanced target classes before training.")
    class_imbalance_method = st.sidebar.selectbox(
        "Imbalance Method",
        ["None", "SMOTE", "Random Oversampling", "Random Undersampling", "Class Weights Only"],
        index=0, key="imbalance_method",
        help="SMOTE creates synthetic minority samples. Class Weights adjusts loss function."
    )
    if class_imbalance_method == "SMOTE":
        smote_neighbors = st.sidebar.slider("SMOTE neighbors", 1, 10, 5, key="smote_neighbors")
    if class_imbalance_method == "Class Weights Only":
        use_class_weights = True

# ===========================================================================
# SCALER MAP (top-level so all tabs can access)
# ===========================================================================
scaler_map = {
    "StandardScaler (z-score)": StandardScaler(),
    "MinMaxScaler (0-1)": MinMaxScaler(),
    "RobustScaler (median/IQR)": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler(),
    "PowerTransformer (Yeo-Johnson)": PowerTransformer(method="yeo-johnson"),
    "QuantileTransformer (Normal)": QuantileTransformer(output_distribution="normal", random_state=42),
    "QuantileTransformer (Uniform)": QuantileTransformer(output_distribution="uniform", random_state=42),
    "No Scaling": None,
}
scaler_info = {
    "StandardScaler (z-score)": "Standardizes features by removing mean and scaling to unit variance. Best for normally distributed data.",
    "MinMaxScaler (0-1)": "Scales features to [0, 1] range. Sensitive to outliers. Good for algorithms requiring bounded inputs.",
    "RobustScaler (median/IQR)": "Uses median and IQR. Robust to outliers. Best when data has significant outliers.",
    "MaxAbsScaler": "Scales by maximum absolute value. Preserves sparsity. Good for sparse data.",
    "PowerTransformer (Yeo-Johnson)": "Gaussianizes features. Handles zero/negative values. Good for skewed distributions.",
    "QuantileTransformer (Normal)": "Maps to normal distribution. Robust to outliers. Good for non-normal data.",
    "QuantileTransformer (Uniform)": "Maps to uniform distribution. Robust to outliers. Good for rank-based algorithms.",
    "No Scaling": "No transformation applied. Use for tree-based models that are scale-invariant.",
}

# ===========================================================================
# TABS
# ===========================================================================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Correlation Matrix", "Data Cleaning", "Statistical Tests",
     "Visualizations", "Feature Engineering", "Feature Importance",
     "Model Training", "Predict", "Data Profiling"]
)

# ===========================================================================
# TAB 0 — CORRELATION MATRIX
# ===========================================================================
with tab0:
    st.header("Correlation Analysis & Variable Selection")
    st.subheader("Full Correlation Matrix")

    if len(numeric_cols) >= 2:
        progress_bar = st.progress(0, text="Computing correlation matrix...")

        progress_bar.progress(10, text="Preparing data...")
        corr_data = {col: df[col].tolist() for col in numeric_cols}

        progress_bar.progress(40, text="Computing correlations...")
        try:
            corr_all = compute_correlation_matrix(df_fingerprint(df[numeric_cols]), corr_data)
        except Exception as e:
            progress_bar.empty()
            st.error(f"Could not compute correlation matrix: {e}")
            corr_all = None

        if corr_all is not None:
            progress_bar.progress(70, text="Rendering heatmap...")
            n = len(numeric_cols)
            corr_w, corr_h = dynamic_fig_size(n, base_height=6, base_width=10, per_item=0.9, max_h=20, max_w=24)

            try:
                if use_plotly:
                    mask = np.triu(np.ones_like(corr_all, dtype=bool))
                    corr_masked = corr_all.where(~mask)
                    fig_plotly = px.imshow(
                        corr_masked, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto",
                        title="Pearson Correlation Matrix (all numeric features)",
                    )
                    fig_plotly.update_layout(width=corr_w * 80, height=corr_h * 80)
                    st.plotly_chart(fig_plotly, use_container_width=True, key="plotly_corr_matrix")
                else:
                    fig, ax = plt.subplots(figsize=(corr_w, corr_h))
                    mask = np.triu(np.ones_like(corr_all, dtype=bool))
                    sns.heatmap(
                        corr_all, mask=mask, annot=True, cmap="coolwarm", center=0,
                        ax=ax, fmt=".2f", square=True, linewidths=0.5, vmin=-1, vmax=1,
                    )
                    ax.set_title("Pearson Correlation Matrix (all numeric features)")
                    show_fig(ax.figure)
            except Exception as e:
                st.warning(f"Heatmap failed ({e}). Showing simplified bar chart fallback.")
                fallback_corr = corr_all[target_col].drop(target_col).sort_values(ascending=False) if target_col in corr_all.columns else corr_all.iloc[0].sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, max(4, len(fallback_corr) * 0.4)))
                fallback_corr.plot(kind="barh", ax=ax, color="steelblue")
                ax.set_title("Correlation Values (heatmap fallback)")
                ax.set_xlabel("Correlation")
                show_fig(ax.figure)

            progress_bar.progress(90, text="Computing target correlations...")

            # --- Correlation with Target ---
            if target_col in corr_all.columns:
                st.subheader(f"Correlation with Target: {target_col}")
                target_corr = corr_all[target_col].drop(target_col).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, max(4, len(target_corr) * 0.4)))
                colors = [
                    "#e74c3c" if abs(v) > 0.7 else "#3498db" if abs(v) > 0.3 else "#95a5a6"
                    for v in target_corr.values
                ]
                target_corr.plot(kind="barh", ax=ax, color=colors)
                ax.set_xlabel("Pearson r")
                ax.set_title(f"Feature Correlation with {target_col}")
                ax.axvline(x=0, color="black", linewidth=0.5)
                show_fig(ax.figure)

                corr_target_df = pd.DataFrame({
                    "Feature": target_corr.index,
                    "Pearson r": target_corr.values.round(4),
                    "Abs(r)": target_corr.abs().values.round(4),
                }).sort_values("Abs(r)", ascending=False)
                st.dataframe(corr_target_df)

            # --- Spearman Correlation ---
            if target_col in numeric_cols:
                st.subheader(f"Spearman Rank Correlation with {target_col}")
                sp_raw = compute_spearman_correlations(df_fingerprint(df[numeric_cols]), numeric_cols, target_col, df.to_dict("list"))
                sp_filtered = [r for r in sp_raw if r["Feature"] != target_col]
                if sp_filtered:
                    sp_df = pd.DataFrame(sp_filtered).sort_values("Spearman rho", key=abs, ascending=False)
                    sp_df["Significant"] = sp_df["P-Value"].apply(lambda p: "Yes" if p < 0.05 else "No")
                    st.dataframe(
                        sp_df.style.map(
                            lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                            subset=["Significant"],
                        )
                    )

            progress_bar.progress(95, text="Finding correlated pairs...")

            # --- High Inter-Feature Correlation ---
            st.subheader("Highly Correlated Feature Pairs")
            corr_threshold = st.slider("Correlation threshold", 0.5, 0.99, 0.85, 0.01, key="corr_threshold")
            high_corr_pairs = []
            for i in range(len(corr_all.columns)):
                for j in range(i + 1, len(corr_all.columns)):
                    val = corr_all.iloc[i, j]
                    if abs(val) >= corr_threshold:
                        high_corr_pairs.append({
                            "Feature 1": corr_all.columns[i],
                            "Feature 2": corr_all.columns[j],
                            "Correlation": round(val, 4),
                        })
            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("Correlation", key=abs, ascending=False)
                st.dataframe(high_corr_df.style.background_gradient(subset=["Correlation"], cmap="RdYlGn_r"))
            else:
                st.info("No feature pairs exceed the correlation threshold.")

            # --- Drop Suggestions ---
            st.subheader("Suggested Variables to Drop")
            st.markdown(
                "Variables are flagged for removal if they are **highly correlated "
                "with another feature** or have **very low correlation with the target**."
            )
            corr_drop_threshold = st.slider("Max inter-feature correlation to keep", 0.5, 0.99, 0.85, 0.01, key="corr_drop_thresh")
            target_keep_threshold = st.slider("Min absolute correlation with target to keep", 0.0, 1.0, 0.05, 0.01, key="target_keep_thresh")

            if target_col in corr_all.columns:
                abs_target_corr = corr_all[target_col].drop(target_col).abs().sort_values(ascending=False)
                ordered_features = abs_target_corr.index.tolist()
            else:
                ordered_features = list(corr_all.columns)

            keep_set = []
            drop_suggestions = []
            for feat in ordered_features:
                if feat not in corr_all.columns:
                    continue
                too_correlated = False
                correlated_with = None
                for kept in keep_set:
                    pair_corr = abs(corr_all.loc[feat, kept])
                    if pair_corr >= corr_drop_threshold:
                        too_correlated = True
                        correlated_with = kept
                        break
                low_target = False
                if target_col in corr_all.columns and feat != target_col:
                    t_corr = abs(corr_all.loc[feat, target_col])
                    if t_corr < target_keep_threshold:
                        low_target = True
                if too_correlated or low_target:
                    reasons = []
                    if too_correlated:
                        reasons.append(f"corr={corr_all.loc[feat, correlated_with]:.2f} with '{correlated_with}'")
                    if low_target:
                        reasons.append(f"low target corr ({corr_all.loc[feat, target_col]:.2f})")
                    drop_suggestions.append({"Feature": feat, "Reason": "; ".join(reasons)})
                else:
                    keep_set.append(feat)

            if drop_suggestions:
                drop_df = pd.DataFrame(drop_suggestions)
                st.dataframe(drop_df)
                drop_choices = st.multiselect("Select features to drop", options=drop_df["Feature"].tolist(), default=[], key="drop_choices")
                if drop_choices:
                    st.warning(f"You have selected **{len(drop_choices)}** features to drop: {', '.join(drop_choices)}")
                    if st.button("Apply Drop", key="apply_drop_corr"):
                        existing = st.session_state.get("extra_dropped_cols", [])
                        st.session_state["extra_dropped_cols"] = list(set(existing + drop_choices))
                        st.session_state["drop_flash"] = f"Dropped {len(drop_choices)} features: {', '.join(drop_choices)}"
                        st.rerun()
            else:
                st.info("No features suggested for removal under current thresholds.")

            # --- VIF ---
            st.subheader("Variance Inflation Factor (VIF)")
            st.markdown("VIF > 5-10 indicates high multicollinearity.")
            if len(numeric_cols) >= 2:
                X_vif = df[numeric_cols].dropna()
                if len(X_vif) > len(numeric_cols):
                    vif_data_dict = {col: X_vif[col].tolist() for col in X_vif.columns}
                    vif_df = compute_vif(df_fingerprint(X_vif), vif_data_dict)
                    vif_df = vif_df.sort_values("VIF", ascending=False)
                    st.dataframe(
                        vif_df.style.background_gradient(subset=["VIF"], cmap="YlOrRd", vmin=1, vmax=20)
                    )
                    high_vif = vif_df[vif_df["VIF"] > 10]
                    if not high_vif.empty:
                        st.warning(f"{len(high_vif)} feature(s) have VIF > 10: {', '.join(high_vif['Feature'].tolist())}")
                else:
                    st.info("Not enough data rows to compute VIF.")

            progress_bar.progress(100, text="Done!")
            progress_bar.empty()
    else:
        st.info("Need at least 2 numeric columns for correlation analysis.")

# ===========================================================================
# TAB 1 — DATA CLEANING
# ===========================================================================
with tab1:
    st.header("Data Cleaning")
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10))

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Missing Values by Column")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.bar_chart(missing)
        st.dataframe(missing.rename("Missing Count").to_frame())
    else:
        st.success("No missing values found!")

    st.subheader("Handle Missing Values")
    fill_method = st.selectbox(
        "Select fill method",
        ["Median (numeric) / Mode (categorical)", "Mean (numeric) / Mode (categorical)", "Drop Rows"],
        key="fill_method",
    )
    df_clean = df_model.copy()
    if fill_method.startswith("Median"):
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ["float64", "int64"]:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    elif fill_method.startswith("Mean"):
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ["float64", "int64"]:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    else:
        df_clean = df_clean.dropna()

    st.subheader("Cleaned Data Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df_clean.shape[0])
    col2.metric("Columns", df_clean.shape[1])
    col3.metric("Missing Values", int(df_clean.isnull().sum().sum()))

    st.subheader("Cleaned Data Sample")
    st.dataframe(df_clean.head(10))

    st.subheader("Column Data Types")
    dtype_df = pd.DataFrame({
        "Column": df_clean.columns,
        "Type": [str(df_clean[c].dtype) for c in df_clean.columns],
        "Unique Values": [df_clean[c].nunique() for c in df_clean.columns],
    })
    st.dataframe(dtype_df)

    st.subheader("Export Cleaned Data")
    csv_clean = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned data as CSV", csv_clean, "cleaned_data.csv", "text/csv", key="dl_cleaned")

    st.subheader("Outlier Detection (IQR Method)")
    if numeric_cols:
        outlier_info = []
        for col in numeric_cols:
            if col not in df_clean.columns:
                continue
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
            pct = round(100 * n_outliers / len(df_clean), 2) if len(df_clean) > 0 else 0
            outlier_info.append({
                "Column": col, "Q1": round(q1, 3), "Q3": round(q3, 3),
                "IQR": round(iqr, 3), "Lower Fence": round(lower, 3),
                "Upper Fence": round(upper, 3), "Outliers": n_outliers, "Outlier %": pct,
            })
        outlier_df = pd.DataFrame(outlier_info).sort_values("Outliers", ascending=False)
        st.dataframe(outlier_df.style.background_gradient(subset=["Outlier %"], cmap="YlOrRd"))
        total_outliers = outlier_df["Outliers"].sum()
        if total_outliers > 0:
            st.warning(f"Total outlier cells detected: {total_outliers}")
        else:
            st.success("No outliers detected.")
    else:
        st.info("No numeric columns for outlier detection.")

# ===========================================================================
# TAB 2 — STATISTICAL TESTS
# ===========================================================================
with tab2:
    st.header("Statistical Tests")
    st.markdown(f"Analyzing relationships between features and **{target_col}**")
    st.info("Each test checks whether a feature is related to the target. Significant results (green) suggest the feature is useful for prediction.")

    feat_numeric = [c for c in feature_cols if df_clean[c].dtype in ["float64", "int64"]]
    feat_categorical = [
        c for c in feature_cols
        if df_clean[c].dtype in ["object", "category", "bool"] or df_clean[c].nunique() <= 10
    ]

    chi2_results = []
    ttest_results = []
    ks_results = []
    corr_results = []
    spearman_results = []
    kendall_results = []
    anova_results = []

    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01, key="alpha")
    correction_method = st.selectbox(
        "Multiple-testing correction",
        ["None", "Bonferroni", "Benjamini-Hochberg (FDR)"],
        index=0, key="correction_method",
        help="Bonferroni controls family-wise error rate (conservative). BH-FDR controls false discovery rate (recommended for many tests)."
    )

    # Auto-recommend correction based on number of tests
    n_features = len(feat_numeric) + len(feat_categorical)
    estimated_tests = n_features * (3 if target_type == "Classification" else 4)
    if estimated_tests > 10:
        recommended = "Benjamini-Hochberg (FDR)"
        st.caption(f"Auto-recommendation: **{recommended}** — {estimated_tests} tests expected (>10). BH-FDR controls false discovery rate while maintaining power.")
    elif estimated_tests > 1:
        recommended = "Bonferroni"
        st.caption(f"Auto-recommendation: **{recommended}** — {estimated_tests} tests expected (<=10). Bonferroni controls false positives conservatively.")
    else:
        recommended = "None"

    test_progress = st.progress(0, text="Running statistical tests...")
    total_tests = 0
    completed_tests = 0

    if target_type == "Classification":
        # Chi-Square
        if feat_categorical:
            total_tests += 1
            st.subheader("Chi-Square Test of Independence (Categorical vs Target)")
            chi2_results = []
            for col in feat_categorical:
                try:
                    contingency = pd.crosstab(df_clean[col], df_clean[target_col])
                    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        st.info(f"Chi-Square skipped for '{col}': insufficient categories in contingency table (need at least 2 rows and 2 columns).")
                        continue
                    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                    chi2_results.append({
                        "Feature": col, "Chi-Square": round(chi2, 4),
                        "P-Value": round(p_val, 6), "DoF": dof,
                        "Significant": "Yes" if p_val < alpha else "No",
                    })
                except Exception as e:
                    st.warning(f"Chi-Square test skipped for '{col}': {e}")
                    continue
            if chi2_results:
                chi2_df = pd.DataFrame(chi2_results)
                st.dataframe(
                    chi2_df.style.map(
                        lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                        subset=["Significant"],
                    )
                )
                for _, row in chi2_df.iterrows():
                    if row["Significant"] == "Yes":
                        st.success(test_interpreter("Chi-Square", row["P-Value"], alpha))
                        break
                else:
                    st.info("No significant associations found by Chi-Square test.")
            completed_tests += 1
            test_progress.progress(completed_tests / total_tests, text="Chi-Square complete...")
        else:
            st.info("Chi-Square skipped: no categorical features in the dataset.")

        # Cramer's V
        if feat_categorical:
            st.subheader("Cramer's V (Effect Size for Categorical Association)")
            st.markdown("Values range from 0 (no association) to 1 (perfect association)")
            cramers_results = []
            for col in feat_categorical:
                try:
                    contingency = pd.crosstab(df_clean[col], df_clean[target_col])
                    chi2 = stats.chi2_contingency(contingency)[0]
                    n = contingency.sum().sum()
                    min_dim = min(contingency.shape) - 1
                    if min_dim == 0 or n == 0:
                        cramers_v = 0.0
                    else:
                        cramers_v = np.sqrt(chi2 / (n * min_dim))
                    strength = (
                        "Strong" if cramers_v > 0.5 else
                        "Moderate" if cramers_v > 0.3 else
                        "Weak" if cramers_v > 0.1 else "Negligible"
                    )
                    cramers_results.append({"Feature": col, "Cramer's V": round(cramers_v, 4), "Strength": strength})
                except Exception:
                    continue
            if cramers_results:
                cramers_df = pd.DataFrame(cramers_results).sort_values("Cramer's V", ascending=False)
                st.dataframe(cramers_df)

        # T-Test / Mann-Whitney U
        if feat_numeric:
            total_tests += 1
            st.subheader("T-Test / Mann-Whitney U (Numeric Features vs Target)")
            ttest_results = []
            target_unique = df_clean[target_col].dropna().unique()
            if len(target_unique) == 2:
                group_a = df_clean[df_clean[target_col] == target_unique[0]]
                group_b = df_clean[df_clean[target_col] == target_unique[1]]
                for col in feat_numeric:
                    try:
                        vals_a = group_a[col].dropna()
                        vals_b = group_b[col].dropna()
                        if len(vals_a) < 2 or len(vals_b) < 2:
                            st.info(f"T-Test skipped for '{col}': insufficient data in one or both groups (need at least 2 values each).")
                            continue
                        sample_a = vals_a.sample(min(500, len(vals_a)), random_state=42)
                        sample_b = vals_b.sample(min(500, len(vals_b)), random_state=42)
                        _, p_norm_a = stats.shapiro(sample_a)
                        _, p_norm_b = stats.shapiro(sample_b)
                        normal = p_norm_a > alpha and p_norm_b > alpha
                        if normal:
                            t_stat, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)
                            test_name = "Welch's T-Test"
                        else:
                            t_stat, p_val = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                            test_name = "Mann-Whitney U"
                        ttest_results.append({
                            "Feature": col, "Test Used": test_name,
                            "Statistic": round(t_stat, 4), "P-Value": round(p_val, 6),
                            f"Mean ({target_unique[0]})": round(vals_a.mean(), 3),
                            f"Mean ({target_unique[1]})": round(vals_b.mean(), 3),
                            "Significant": "Yes" if p_val < alpha else "No",
                        })
                    except Exception as e:
                        st.warning(f"T-Test skipped for '{col}': {e}")
                        continue
            else:
                st.info(f"T-Test / Mann-Whitney U skipped: target has {len(target_unique)} unique values (requires exactly 2). Running ANOVA/Kruskal-Wallis instead.")
                if feat_numeric:
                    total_tests += 1
                    st.subheader("ANOVA / Kruskal-Wallis (Numeric Features vs Multi-Class Target)")
                    mc_ttest_results = []
                    for col in feat_numeric:
                        try:
                            groups = [
                                grp[col].dropna().values
                                for _, grp in df_clean.groupby(target_col)
                                if len(grp[col].dropna()) >= 2
                            ]
                            if len(groups) < 2:
                                continue
                            all_normal = True
                            for g in groups[:5]:
                                sample = g[:min(500, len(g))]
                                if len(sample) >= 3:
                                    _, p_n = stats.shapiro(sample)
                                    if p_n <= alpha:
                                        all_normal = False
                                        break
                            if all_normal:
                                f_stat, p_val = stats.f_oneway(*groups)
                                test_name = "One-Way ANOVA"
                            else:
                                h_stat, p_val = stats.kruskal(*groups)
                                f_stat = h_stat
                                test_name = "Kruskal-Wallis"
                            mc_ttest_results.append({
                                "Feature": col, "Test Used": test_name,
                                "Statistic": round(f_stat, 4), "P-Value": round(p_val, 6),
                                "Significant": "Yes" if p_val < alpha else "No",
                            })
                        except Exception as e:
                            st.warning(f"ANOVA/Kruskal-Wallis skipped for '{col}': {e}")
                            continue
                    if mc_ttest_results:
                        mc_ttest_df = pd.DataFrame(mc_ttest_results)
                        st.dataframe(
                            mc_ttest_df.style.map(
                                lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                                subset=["Significant"],
                            )
                        )
                        for _, row in mc_ttest_df.iterrows():
                            if row["Significant"] == "Yes":
                                test_display = row["Test Used"].replace("One-Way ANOVA", "ANOVA")
                                st.success(test_interpreter(test_display, row["P-Value"], alpha))
                                break
                        else:
                            st.info("No significant differences found by ANOVA/Kruskal-Wallis.")
            if ttest_results:
                ttest_df = pd.DataFrame(ttest_results)
                st.dataframe(
                    ttest_df.style.map(
                        lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                        subset=["Significant"],
                    )
                )
                for _, row in ttest_df.iterrows():
                    if row["Significant"] == "Yes":
                        st.success(test_interpreter(row["Test Used"], row["P-Value"], alpha))
                        break
                else:
                    st.info("No significant differences found by T-Test/Mann-Whitney U.")
            completed_tests += 1
            test_progress.progress(completed_tests / total_tests, text="T-Test complete...")
        else:
            st.info("T-Test / Mann-Whitney U skipped: no numeric features in the dataset.")

        # Kolmogorov-Smirnov
        if feat_numeric:
            total_tests += 1
            st.subheader("Kolmogorov-Smirnov Test (Distribution Comparison)")
            st.markdown("Tests whether two groups have the same distribution")
            ks_results = []
            target_unique = df_clean[target_col].dropna().unique()
            if len(target_unique) == 2:
                group_a = df_clean[df_clean[target_col] == target_unique[0]]
                group_b = df_clean[df_clean[target_col] == target_unique[1]]
                for col in feat_numeric:
                    try:
                        vals_a = group_a[col].dropna().values
                        vals_b = group_b[col].dropna().values
                        if len(vals_a) < 2 or len(vals_b) < 2:
                            st.info(f"KS test skipped for '{col}': insufficient data (need at least 2 values per group).")
                            continue
                        ks_stat, p_val = stats.ks_2samp(vals_a, vals_b)
                        ks_results.append({
                            "Feature": col, "KS Statistic": round(ks_stat, 4),
                            "P-Value": round(p_val, 6),
                            "Significant": "Yes" if p_val < alpha else "No",
                        })
                    except Exception as e:
                        st.warning(f"KS test skipped for '{col}': {e}")
                        continue
            else:
                st.info(f"KS test skipped: target has {len(target_unique)} unique values (requires exactly 2). Running pairwise KS instead.")
                if feat_numeric:
                    total_tests += 1
                    st.subheader("Pairwise KS Test (Multi-Class Distribution Comparison)")
                    mc_ks_results = []
                    for col in feat_numeric:
                        try:
                            groups_data = [
                                grp[col].dropna().values
                                for _, grp in df_clean.groupby(target_col)
                                if len(grp[col].dropna()) >= 2
                            ]
                            if len(groups_data) < 2:
                                continue
                            max_ks = 0
                            max_pair = ""
                            for gi in range(len(groups_data)):
                                for gj in range(gi + 1, len(groups_data)):
                                    ks_s, _ = stats.ks_2samp(groups_data[gi], groups_data[gj])
                                    if ks_s > max_ks:
                                        max_ks = ks_s
                                        max_pair = f"group{gi} vs group{gj}"
                            _, p_val = stats.ks_2samp(groups_data[0], groups_data[1])
                            mc_ks_results.append({
                                "Feature": col, "KS Statistic": round(max_ks, 4),
                                "Worst Pair": max_pair,
                                "P-Value": round(p_val, 6),
                                "Significant": "Yes" if p_val < alpha else "No",
                            })
                        except Exception as e:
                            st.warning(f"Pairwise KS skipped for '{col}': {e}")
                            continue
                    if mc_ks_results:
                        mc_ks_df = pd.DataFrame(mc_ks_results)
                        st.dataframe(
                            mc_ks_df.style.map(
                                lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                                subset=["Significant"],
                            )
                        )
            if ks_results:
                ks_df = pd.DataFrame(ks_results)
                st.dataframe(
                    ks_df.style.map(
                        lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                        subset=["Significant"],
                    )
                )
                for _, row in ks_df.iterrows():
                    if row["Significant"] == "Yes":
                        st.success(test_interpreter("KS", row["P-Value"], alpha))
                        break
                else:
                    st.info("No significant distribution differences found by KS test.")
            completed_tests += 1
            test_progress.progress(completed_tests / total_tests, text="KS test complete...")
        else:
            st.info("KS test skipped: no numeric features in the dataset.")

        # Mutual Information
        if (feat_numeric or feat_categorical) and len(df_clean) > 0:
            st.subheader("Mutual Information (Non-linear Dependencies)")
            st.markdown("Measures any statistical dependency; 0 = independent, higher = more dependent")
            df_mi = df_clean[feature_cols].copy()
            le_dict_mi = {}
            for col in feat_categorical:
                if col in df_mi.columns:
                    le_mi = LabelEncoder()
                    df_mi[col] = le_mi.fit_transform(df_mi[col].astype(str))
                    le_dict_mi[col] = le_mi
            X_mi = df_mi.apply(pd.to_numeric, errors="coerce").fillna(0)
            y_mi = df_clean[target_col].fillna(0)
            valid = y_mi.notna()
            X_mi = X_mi[valid]
            y_mi = y_mi[valid]
            if len(X_mi) > 0 and X_mi.shape[1] > 0:
                try:
                    mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
                    mi_df = pd.DataFrame({
                        "Feature": X_mi.columns, "Mutual Information": mi_scores.round(4),
                    }).sort_values("Mutual Information", ascending=False)
                    st.dataframe(mi_df)
                    fig, ax = plt.subplots(figsize=(10, max(4, len(mi_df) * 0.4)))
                    mi_df_sorted = mi_df.sort_values("Mutual Information", ascending=True)
                    ax.barh(mi_df_sorted["Feature"], mi_df_sorted["Mutual Information"], color="teal")
                    ax.set_xlabel("Mutual Information Score")
                    ax.set_title("Feature Mutual Information with Target")
                    show_fig(ax.figure)
                except Exception as e:
                    st.warning(f"Could not compute Mutual Information: {e}")

    elif target_type == "Regression":
        # Pearson
        if feat_numeric:
            total_tests += 1
            st.subheader("Pearson Correlation with Target")
            pearson_raw = compute_pearson_correlations(df_fingerprint(df_clean[feat_numeric]), feat_numeric, target_col, df_clean.to_dict("list"))
            for r in pearson_raw:
                r["Significant"] = "Yes" if r["P-Value"] < alpha else "No"
            corr_results.extend(pearson_raw)
            if pearson_raw:
                corr_df = pd.DataFrame(pearson_raw).sort_values("Pearson r", ascending=False)
                st.dataframe(
                    corr_df.style.map(
                        lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                        subset=["Significant"],
                    )
                )
                for _, row in corr_df.iterrows():
                    if row["Significant"] == "Yes":
                        st.success(test_interpreter("Pearson", row["P-Value"], alpha))
                        break
                else:
                    st.info("No significant linear correlations found by Pearson test.")
            completed_tests += 1
            test_progress.progress(completed_tests / total_tests, text="Pearson complete...")
        else:
            st.info("Pearson skipped: no numeric features in the dataset.")

        # Spearman
        if feat_numeric:
            total_tests += 1
            st.subheader("Spearman Rank Correlation with Target")
            sp_raw = compute_spearman_correlations(df_fingerprint(df_clean[feat_numeric]), feat_numeric, target_col, df_clean.to_dict("list"))
            for r in sp_raw:
                r["Significant"] = "Yes" if r["P-Value"] < alpha else "No"
            spearman_results.extend(sp_raw)
            if sp_raw:
                sp_df = pd.DataFrame(sp_raw).sort_values("Spearman rho", ascending=False)
                st.dataframe(
                    sp_df.style.map(
                        lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                        subset=["Significant"],
                    )
                )
                for _, row in sp_df.iterrows():
                    if row["Significant"] == "Yes":
                        st.success(test_interpreter("Spearman", row["P-Value"], alpha))
                        break
                else:
                    st.info("No significant monotonic relationships found by Spearman test.")
            completed_tests += 1
            test_progress.progress(completed_tests / total_tests, text="Spearman complete...")
        else:
            st.info("Spearman skipped: no numeric features in the dataset.")

        # Kendall
        if feat_numeric:
            total_tests += 1
            st.subheader("Kendall's Tau (Ordinal Association)")
            kendall_raw = compute_kendall_correlations(df_fingerprint(df_clean[feat_numeric]), feat_numeric, target_col, df_clean.to_dict("list"))
            for r in kendall_raw:
                r["Significant"] = "Yes" if r["P-Value"] < alpha else "No"
            kendall_results.extend(kendall_raw)
            if kendall_raw:
                ken_df = pd.DataFrame(kendall_raw).sort_values("Kendall tau", ascending=False)
                st.dataframe(
                    ken_df.style.map(
                        lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                        subset=["Significant"],
                    )
                )
                for _, row in ken_df.iterrows():
                    if row["Significant"] == "Yes":
                        st.success(test_interpreter("Kendall", row["P-Value"], alpha))
                        break
                else:
                    st.info("No significant ordinal associations found by Kendall's Tau.")
            completed_tests += 1
            test_progress.progress(completed_tests / total_tests, text="Kendall complete...")
        else:
            st.info("Kendall skipped: no numeric features in the dataset.")

        # ANOVA
        if feat_categorical:
            total_tests += 1
            st.subheader("One-Way ANOVA (Categorical Features vs Target)")
            anova_results = []
            for col in feat_categorical:
                try:
                    groups = [
                        group[target_col].dropna().values
                        for name, group in df_clean.groupby(col)
                        if len(group) > 1
                    ]
                    if len(groups) < 2:
                        st.info(f"ANOVA skipped for '{col}': fewer than 2 groups with sufficient data.")
                        continue
                    all_normal = True
                    for g in groups[:5]:
                        sample = g[:min(500, len(g))]
                        if len(sample) >= 3:
                            _, p_n = stats.shapiro(sample)
                            if p_n <= alpha:
                                all_normal = False
                                break
                    if all_normal:
                        f_stat, p_val = stats.f_oneway(*groups)
                        test_name = "One-Way ANOVA"
                    else:
                        h_stat, p_val = stats.kruskal(*groups)
                        f_stat = h_stat
                        test_name = "Kruskal-Wallis"
                    anova_results.append({
                        "Feature": col, "Test Used": test_name,
                        "Statistic": round(f_stat, 4), "P-Value": round(p_val, 6),
                        "Significant": "Yes" if p_val < alpha else "No",
                    })
                except Exception as e:
                    st.warning(f"ANOVA/Kruskal-Wallis skipped for '{col}': {e}")
                    continue
            if anova_results:
                anova_df = pd.DataFrame(anova_results)
                st.dataframe(
                    anova_df.style.map(
                        lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                        subset=["Significant"],
                    )
                )
                for _, row in anova_df.iterrows():
                    if row["Significant"] == "Yes":
                        test_name = row["Test Used"].replace("One-Way ANOVA", "ANOVA")
                        st.success(test_interpreter(test_name, row["P-Value"], alpha))
                        break
                else:
                    st.info("No significant group differences found by ANOVA/Kruskal-Wallis.")
            completed_tests += 1
            test_progress.progress(completed_tests / total_tests, text="ANOVA complete...")
        else:
            st.info("ANOVA skipped: no categorical features in the dataset.")

        # Shapiro-Wilk on target
        st.subheader("Normality Test on Target Variable")
        target_vals = df_clean[target_col].dropna()
        if len(target_vals) >= 3:
            sample = target_vals.sample(min(500, len(target_vals)), random_state=42)
            w_stat, p_val = stats.shapiro(sample)
            st.write(f"**Shapiro-Wilk Test:** W = {w_stat:.4f}, p = {p_val:.6f}")
            if p_val < alpha:
                st.warning("Target variable is **not normally distributed** (p < alpha).")
            else:
                st.success("Target variable appears normally distributed (p >= alpha).")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(target_vals, bins=30, color="#3498db", edgecolor="black", density=True)
            x_range = np.linspace(target_vals.min(), target_vals.max(), 100)
            mu, sigma = target_vals.mean(), target_vals.std()
            ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), "r-", lw=2, label="Normal fit")
            ax.set_title(f"{target_col} Distribution vs Normal")
            ax.legend()
            show_fig(ax.figure)

        # Durbin-Watson
        st.subheader("Durbin-Watson Test (Autocorrelation Check)")
        st.markdown("Values near 2 = no autocorrelation.")
        target_vals_dw = df_clean[target_col].dropna().values
        if len(target_vals_dw) > 10:
            dw_stat = np.sum(np.diff(target_vals_dw) ** 2) / np.sum(target_vals_dw ** 2)
            st.write(f"**Durbin-Watson statistic:** {round(dw_stat, 4)}")
            if dw_stat < 1.5:
                st.info("Positive autocorrelation detected.")
            elif dw_stat > 2.5:
                st.info("Negative autocorrelation detected.")
            else:
                st.success("No significant autocorrelation.")

        # Breusch-Pagan
        if feat_numeric:
            st.subheader("Breusch-Pagan Test (Homoscedasticity)")
            st.markdown("Tests whether variance of residuals is constant.")
            X_bp = df_clean[feat_numeric].dropna()
            y_bp = df_clean[target_col].loc[X_bp.index]
            if len(X_bp) > len(feat_numeric) + 1:
                try:
                    X_bp_const = sm.add_constant(X_bp)
                    ols_bp = sm.OLS(y_bp, X_bp_const).fit()
                    bp_stat, bp_pval, _, _ = het_breuschpagan(ols_bp.resid, ols_bp.model.exog)
                    st.write(f"**BP statistic:** {round(bp_stat, 4)}, **p-value:** {round(bp_pval, 6)}")
                    if bp_pval < alpha:
                        st.warning("Heteroscedasticity detected (variance is not constant).")
                    else:
                        st.success("Homoscedasticity assumption holds.")
                except Exception as e:
                    st.info(f"Could not run Breusch-Pagan test: {e}")

        # Mutual Information
        if (feat_numeric or feat_categorical) and len(df_clean) > 0:
            st.subheader("Mutual Information (Non-linear Dependencies)")
            df_mi = df_clean[feature_cols].copy()
            for col in feat_categorical:
                if col in df_mi.columns:
                    le_mi = LabelEncoder()
                    df_mi[col] = le_mi.fit_transform(df_mi[col].astype(str))
            X_mi = df_mi.apply(pd.to_numeric, errors="coerce").fillna(0)
            y_mi = df_clean[target_col].fillna(0)
            valid = y_mi.notna()
            X_mi = X_mi[valid]
            y_mi = y_mi[valid]
            if len(X_mi) > 0 and X_mi.shape[1] > 0:
                try:
                    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
                    mi_df = pd.DataFrame({
                        "Feature": X_mi.columns, "Mutual Information": mi_scores.round(4),
                    }).sort_values("Mutual Information", ascending=False)
                    st.dataframe(mi_df)
                    fig, ax = plt.subplots(figsize=(10, max(4, len(mi_df) * 0.4)))
                    mi_df_sorted = mi_df.sort_values("Mutual Information", ascending=True)
                    ax.barh(mi_df_sorted["Feature"], mi_df_sorted["Mutual Information"], color="teal")
                    ax.set_xlabel("Mutual Information Score")
                    ax.set_title("Feature Mutual Information with Target")
                    show_fig(ax.figure)
                except Exception as e:
                    st.warning(f"Could not compute Mutual Information: {e}")

    # --- Feature Correlation Matrix ---
    st.subheader("Feature Correlation Matrix")
    if feat_numeric:
        n = len(feat_numeric)
        try:
            corr = df_clean[feat_numeric].corr()
            if use_plotly:
                mask = np.triu(np.ones_like(corr, dtype=bool))
                corr_masked = corr.where(~mask)
                fig_pl = px.imshow(corr_masked, text_auto=".2f", color_continuous_scale="RdBu_r",
                                   zmin=-1, zmax=1, aspect="auto")
                st.plotly_chart(fig_pl, use_container_width=True, key="plotly_feat_corr_matrix")
            else:
                fig, ax = plt.subplots(figsize=(max(8, min(14, n * 0.8)), max(6, min(12, n * 0.7))))
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f", square=True, linewidths=0.5)
                show_fig(ax.figure)
        except Exception as e:
            st.warning(f"Feature correlation heatmap failed: {e}. Showing simplified fallback.")
            try:
                corr_vals = df_clean[feat_numeric].corr().iloc[0]
                fig, ax = plt.subplots(figsize=(10, max(4, len(corr_vals) * 0.4)))
                corr_vals.plot(kind="barh", ax=ax, color="steelblue")
                ax.set_title("Correlation fallback (first feature)")
                show_fig(ax.figure)
            except Exception:
                st.info("Could not render any correlation visualization for these features.")

    test_progress.progress(100, text="All tests complete!")
    test_progress.empty()

    # --- Multiple-testing correction ---
    if correction_method != "None":
        all_result_lists = [chi2_results, ttest_results, corr_results, spearman_results, kendall_results, anova_results, ks_results]
        all_pvals = []
        pval_index = []
        for res_list in all_result_lists:
            for idx, r in enumerate(res_list or []):
                all_pvals.append(r.get("P-Value", 1.0))
                pval_index.append((res_list, idx))

        if all_pvals:
            if correction_method == "Bonferroni":
                adjusted = [min(p * len(all_pvals), 1.0) for p in all_pvals]
            else:
                _, adjusted_arr, _, _ = multipletests(all_pvals, method="fdr_bh")
                adjusted = list(adjusted_arr)

            for (_, idx), adj_p in zip(pval_index, adjusted):
                res_list[idx]["P-Value (adjusted)"] = round(adj_p, 6)
                res_list[idx]["Significant"] = "Yes" if adj_p < alpha else "No"

            n_before = sum(1 for p in all_pvals if p < alpha)
            n_after = sum(1 for p in adjusted if p < alpha)
            st.info(f"**{correction_method} correction applied:** {n_before} raw significant → {n_after} after correction ({len(all_pvals)} total tests).")

    # Summary
    st.markdown("---")
    st.subheader("Summary & Recommendations")
    sig_features = []
    nonsig_features = []
    for results_list in [chi2_results, ttest_results, corr_results, spearman_results, kendall_results, anova_results, ks_results]:
        for r in (results_list or []):
            feat = r.get("Feature", "")
            if r.get("Significant") == "Yes" and feat not in sig_features:
                sig_features.append(feat)
            elif feat and feat not in sig_features:
                nonsig_features.append(feat)
    if sig_features:
        st.success(f"**Significant features to keep:** {', '.join(set(sig_features))}")
    if nonsig_features:
        st.warning(f"**Non-significant features to consider dropping:** {', '.join(set(nonsig_features))}")
    if not sig_features and not nonsig_features:
        st.info("Run the statistical tests above to get feature recommendations.")

# ===========================================================================
# TAB 3 — VISUALIZATIONS
# ===========================================================================
with tab3:
    st.header("Visualizations")
    st.subheader(f"Target Variable: {target_col}")

    apply_viz_style()
    palette = get_palette(viz_palette)
    plotly_palette = get_plotly_palette()

    if len(df_clean) > 5000:
        if use_plotly:
            if target_type == "Classification":
                target_counts = df_clean[target_col].value_counts().reset_index()
                target_counts.columns = [target_col, "Count"]
                fig_bar = px.bar(target_counts, x=target_col, y="Count",
                                 title=f"{target_col} Distribution", color=target_col,
                                 color_discrete_sequence=plotly_palette)
                st.plotly_chart(fig_bar, use_container_width=True, key="viz_large_target_bar")
                if feat_numeric:
                    top_feat = feat_numeric[0]
                    fig_box = px.box(df_clean, x=target_col, y=top_feat, color=target_col,
                                     title=f"{top_feat} by {target_col}",
                                     color_discrete_sequence=plotly_palette)
                    st.plotly_chart(fig_box, use_container_width=True, key="viz_large_target_box")
            else:
                fig_hist = px.histogram(df_clean, x=target_col, nbins=30,
                                        title=f"{target_col} Distribution",
                                        color_discrete_sequence=["#3498db"])
                st.plotly_chart(fig_hist, use_container_width=True, key="viz_large_target_hist")
                if feat_numeric:
                    top_feat = feat_numeric[0]
                    fig_scatter = px.scatter(df_clean, x=top_feat, y=target_col,
                                            title=f"{top_feat} vs {target_col}",
                                            opacity=0.5, color_discrete_sequence=["#e74c3c"])
                    st.plotly_chart(fig_scatter, use_container_width=True, key="viz_large_target_scatter")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(viz_width, viz_height))
            if target_type == "Classification":
                df_clean[target_col].value_counts().plot(
                    kind="bar", ax=axes[0],
                    color=(palette[:df_clean[target_col].nunique()] if isinstance(palette, list) else "steelblue"),
                )
                axes[0].set_title(f"{target_col} Distribution")
                axes[0].set_ylabel("Count")
                axes[0].tick_params(axis="x", rotation=45)
                if feat_numeric:
                    top_feat = feat_numeric[0]
                    sns.boxplot(data=df_clean, x=target_col, y=top_feat, ax=axes[1],
                                hue=target_col, legend=False,
                                palette=(palette[:df_clean[target_col].nunique()] if isinstance(palette, list) else "Set1"))
                    axes[1].set_title(f"{top_feat} by {target_col}")
            else:
                axes[0].hist(df_clean[target_col].dropna(), bins=30, color="#3498db", edgecolor="black")
                axes[0].set_title(f"{target_col} Distribution")
                axes[0].set_ylabel("Frequency")
                if feat_numeric:
                    top_feat = feat_numeric[0]
                    axes[1].scatter(df_clean[top_feat], df_clean[target_col], alpha=0.5, color="#e74c3c")
                    axes[1].set_xlabel(top_feat)
                    axes[1].set_ylabel(target_col)
                    axes[1].set_title(f"{top_feat} vs {target_col}")
            show_fig(axes[0].figure, height_min=5)
    else:
        if use_plotly and target_type == "Classification":
            target_counts = df_clean[target_col].value_counts().reset_index()
            target_counts.columns = [target_col, "Count"]
            fig_bar = px.bar(target_counts, x=target_col, y="Count",
                             title=f"{target_col} Distribution", color=target_col,
                             color_discrete_sequence=plotly_palette)
            st.plotly_chart(fig_bar, use_container_width=True, key="viz_small_target_bar")
            if feat_numeric:
                top_feat = feat_numeric[0]
                fig_box = px.box(df_clean, x=target_col, y=top_feat, color=target_col,
                                 title=f"{top_feat} by {target_col}",
                                 color_discrete_sequence=plotly_palette)
                st.plotly_chart(fig_box, use_container_width=True, key="viz_small_target_box")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(viz_width, viz_height))
            if target_type == "Classification":
                df_clean[target_col].value_counts().plot(
                    kind="bar", ax=axes[0],
                    color=(palette[:df_clean[target_col].nunique()] if isinstance(palette, list) else "steelblue"),
                )
                axes[0].set_title(f"{target_col} Distribution")
                axes[0].set_ylabel("Count")
                axes[0].tick_params(axis="x", rotation=45)
                if feat_numeric:
                    top_feat = feat_numeric[0]
                    sns.boxplot(data=df_clean, x=target_col, y=top_feat, ax=axes[1],
                                hue=target_col, legend=False,
                                palette=(palette[:df_clean[target_col].nunique()] if isinstance(palette, list) else "Set1"))
                    axes[1].set_title(f"{top_feat} by {target_col}")
            else:
                axes[0].hist(df_clean[target_col].dropna(), bins=30, color="#3498db", edgecolor="black")
                axes[0].set_title(f"{target_col} Distribution")
                axes[0].set_ylabel("Frequency")
                if feat_numeric:
                    top_feat = feat_numeric[0]
                    axes[1].scatter(df_clean[top_feat], df_clean[target_col], alpha=0.5, color="#e74c3c")
                    axes[1].set_xlabel(top_feat)
                    axes[1].set_ylabel(target_col)
                    axes[1].set_title(f"{top_feat} vs {target_col}")
            show_fig(axes[0].figure, height_min=5)

    st.subheader("Feature Distributions")
    viz_features = st.multiselect(
        "Select features to visualize", feature_cols,
        default=feature_cols[:min(4, len(feature_cols))],
        key="viz_features",
    )
    if viz_features:
        if use_plotly:
            for feature in viz_features:
                if feature in feat_numeric:
                    if target_type == "Classification":
                        fig_box = px.box(df_clean, x=target_col, y=feature, color=target_col,
                                         title=f"{feature} by {target_col}",
                                         color_discrete_sequence=plotly_palette)
                        st.plotly_chart(fig_box, use_container_width=True, key=f"viz_feat_box_{feature}")
                    else:
                        fig_sc = px.scatter(df_clean, x=feature, y=target_col,
                                            title=f"{feature} vs {target_col}", opacity=0.5,
                                            color_discrete_sequence=["#3498db"])
                        st.plotly_chart(fig_sc, use_container_width=True, key=f"viz_feat_scatter_{feature}")
                else:
                    if target_type == "Classification":
                        ct = pd.crosstab(df_clean[feature], df_clean[target_col], normalize="index") * 100
                        ct_reset = ct.reset_index().melt(id_vars=feature, var_name=target_col, value_name="Percentage")
                        fig_ct = px.bar(ct_reset, x=feature, y="Percentage", color=target_col,
                                        title=f"{feature} by {target_col}", barmode="stack",
                                        color_discrete_sequence=plotly_palette)
                        st.plotly_chart(fig_ct, use_container_width=True, key=f"viz_feat_bar_{feature}")
                    else:
                        means = df_clean.groupby(feature)[target_col].mean().sort_values().reset_index()
                        fig_means = px.bar(means, x=feature, y=target_col,
                                           title=f"Mean {target_col} by {feature}",
                                           color_discrete_sequence=["#3498db"])
                        st.plotly_chart(fig_means, use_container_width=True, key=f"viz_feat_means_{feature}")
        else:
            n_features = len(viz_features)
            cols_per_row = 2
            rows = (n_features + 1) // cols_per_row
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(viz_width, max(viz_height, viz_height * rows)))
            axes_flat = axes.flatten() if n_features > 1 else [axes]
            for idx, feature in enumerate(viz_features):
                ax = axes_flat[idx]
                if feature in feat_numeric:
                    if target_type == "Classification":
                        sns.boxplot(data=df_clean, x=target_col, y=feature, ax=ax,
                                    hue=target_col, legend=False,
                                    palette=(palette[:df_clean[target_col].nunique()] if isinstance(palette, list) else "Set1"))
                        ax.set_title(f"{feature} by {target_col}")
                    else:
                        ax.scatter(df_clean[feature], df_clean[target_col], alpha=0.5, color="#3498db")
                        ax.set_xlabel(feature)
                        ax.set_ylabel(target_col)
                        ax.set_title(f"{feature} vs {target_col}")
                else:
                    if target_type == "Classification":
                        ct = (pd.crosstab(df_clean[feature], df_clean[target_col], normalize="index") * 100)
                        ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set1")
                        ax.set_title(f"{feature} by {target_col}")
                        ax.set_ylabel("Percentage")
                        ax.tick_params(axis="x", rotation=45)
                    else:
                        means = df_clean.groupby(feature)[target_col].mean().sort_values()
                        means.plot(kind="bar", ax=ax, color="#3498db")
                        ax.set_title(f"Mean {target_col} by {feature}")
                        ax.tick_params(axis="x", rotation=45)
            for idx in range(n_features, len(axes_flat)):
                axes_flat[idx].set_visible(False)
            show_fig(axes_flat[0].figure, height_min=viz_height * rows)

    if len(feature_cols) <= 8 and len(df_clean) <= 500:
        st.subheader("Pairplot (selected features)")
        pair_features = st.multiselect(
            "Select features for pairplot", feat_numeric[:6],
            default=feat_numeric[:min(3, len(feat_numeric))], key="pairplot_select",
        )
        if pair_features:
            pair_cols = pair_features + [target_col]
            pair_kwargs = {"diag_kind": "kde"}
            if target_type == "Classification":
                pair_kwargs["hue"] = target_col
                pair_kwargs["palette"] = "Set1"
            fig = sns.pairplot(df_clean[pair_cols], **pair_kwargs)
            st.pyplot(fig)
            plt.close("all")

# ===========================================================================
# TAB 4 — FEATURE ENGINEERING (Encoding & Scaling)
# ===========================================================================
with tab4:
    st.header("Feature Engineering")
    st.markdown("Configure encoding, scaling, and class imbalance handling before model training.")

    # --- Encoding Preview ---
    st.subheader("Categorical Encoding")
    if categorical_cols:
        st.markdown("Configure how each categorical feature is encoded for ML models.")
        encoding_preview = []
        for col in categorical_cols:
            method = encoding_methods.get(col, encoding_suggestions.get(col, "Label Encoding"))
            n_unique = df_model[col].nunique()
            encoding_preview.append({
                "Feature": col,
                "Current Type": str(df_model[col].dtype),
                "Unique Values": n_unique,
                "Selected Encoding": method,
                "Suggestion": encoding_suggestions.get(col, "Label Encoding"),
                "Match": "Yes" if method == encoding_suggestions.get(col, "Label Encoding") else "No",
            })
        enc_preview_df = pd.DataFrame(encoding_preview)
        st.dataframe(
            enc_preview_df.style.map(
                lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                subset=["Match"],
            )
        )
        if (enc_preview_df["Match"] == "No").any():
            mismatched = enc_preview_df[enc_preview_df["Match"] == "No"]["Feature"].tolist()
            st.warning(f"Your encoding differs from suggestion for: {', '.join(mismatched)}. Suggestion is based on cardinality analysis.")
    else:
        st.info("No categorical columns found in the dataset.")

    # --- Scaling Preview ---
    st.subheader("Feature Scaling")
    st.markdown(f"**Selected:** {scaling_method}")
    st.info(scaler_info.get(scaling_method, ""))

    # Prepare encoded numeric data for scaling preview
    df_scale_preview = df_model.copy()
    encoding_errors = []
    for col in categorical_cols:
        if col in df_scale_preview.columns:
            method = encoding_methods.get(col, encoding_suggestions.get(col, "Label Encoding"))
            if method == "Drop":
                df_scale_preview = df_scale_preview.drop(columns=[col], errors="ignore")
            elif method == "One-Hot Encoding":
                try:
                    dummies = pd.get_dummies(df_scale_preview[col], prefix=col, drop_first=False, dtype=int)
                    df_scale_preview = pd.concat([df_scale_preview.drop(columns=[col]), dummies], axis=1)
                except Exception as e:
                    encoding_errors.append(f"{col}: {e}")
                    df_scale_preview = df_scale_preview.drop(columns=[col], errors="ignore")
            elif method == "Label Encoding":
                try:
                    le_s = LabelEncoder()
                    df_scale_preview[col] = le_s.fit_transform(df_scale_preview[col].astype(str))
                except Exception as e:
                    encoding_errors.append(f"{col}: {e}")
                    df_scale_preview = df_scale_preview.drop(columns=[col], errors="ignore")
            elif method == "None (skip)":
                df_scale_preview = df_scale_preview.drop(columns=[col], errors="ignore")

    if encoding_errors:
        st.warning(f"Encoding errors during scaling preview: {'; '.join(encoding_errors)}")

    numeric_for_scaling = df_scale_preview.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_for_scaling:
        numeric_for_scaling.remove(target_col)

    if numeric_for_scaling and scaling_method != "No Scaling":
        df_before_scale = df_scale_preview[numeric_for_scaling].copy()

        scaler_preview_obj = scaler_map.get(scaling_method, StandardScaler())
        if scaler_preview_obj is not None:
            try:
                scaled_values = scaler_preview_obj.fit_transform(df_before_scale)
                df_after_scale = pd.DataFrame(scaled_values, columns=numeric_for_scaling, index=df_before_scale.index)
            except Exception as e:
                st.warning(f"Could not apply scaling for preview: {e}")
                df_after_scale = None
        else:
            df_after_scale = None

        if df_after_scale is not None:
            # Before stats
            st.markdown("#### Before Scaling (Raw Numeric Features)")
            st.dataframe(df_before_scale.head(10))

            before_stats = df_before_scale.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
            before_stats.columns = ["Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]
            before_stats = before_stats.round(4)
            st.dataframe(before_stats.style.background_gradient(subset=["Std"], cmap="YlOrRd"))

            # After stats
            st.markdown("#### After Scaling")
            st.dataframe(df_after_scale.head(10))

            after_stats = df_after_scale.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
            after_stats.columns = ["Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]
            after_stats = after_stats.round(4)
            st.dataframe(after_stats.style.background_gradient(subset=["Std"], cmap="YlGn"))

            # Side-by-side comparison table
            st.markdown("#### Statistics Comparison (Before → After)")
            comparison_rows = []
            for col_name in numeric_for_scaling:
                b = before_stats.loc[col_name] if col_name in before_stats.index else None
                a = after_stats.loc[col_name] if col_name in after_stats.index else None
                if b is not None and a is not None:
                    comparison_rows.append({
                        "Feature": col_name,
                        "Mean (Before)": b["Mean"], "Mean (After)": a["Mean"],
                        "Std (Before)": b["Std"], "Std (After)": a["Std"],
                        "Min (Before)": b["Min"], "Min (After)": a["Min"],
                        "Max (Before)": b["Max"], "Max (After)": a["Max"],
                    })
            if comparison_rows:
                comp_df = pd.DataFrame(comparison_rows)
                st.dataframe(comp_df.style.background_gradient(subset=["Mean (After)", "Std (After)"], cmap="YlGn"))

            # Distribution plots (before vs after)
            st.markdown("#### Distribution Comparison")
            max_features_to_plot = min(6, len(numeric_for_scaling))
            plot_features = st.multiselect(
                "Select features to compare distributions",
                numeric_for_scaling,
                default=numeric_for_scaling[:max_features_to_plot],
                key="scale_plot_features",
            )
            if plot_features:
                if use_plotly:
                    for feat in plot_features:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=df_before_scale[feat], name="Before Scaling",
                            opacity=0.6, marker_color="#e74c3c",
                        ))
                        fig.add_trace(go.Histogram(
                            x=df_after_scale[feat], name="After Scaling",
                            opacity=0.6, marker_color="#3498db",
                        ))
                        fig.update_layout(
                            title=f"{feat} — Before vs After Scaling",
                            barmode="overlay", xaxis_title=feat, yaxis_title="Count",
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"scale_dist_{feat}")
                else:
                    n_feats = len(plot_features)
                    cols_per_row = 2
                    rows = (n_feats + 1) // cols_per_row
                    fig, axes = plt.subplots(rows, cols_per_row, figsize=(viz_width * cols_per_row, viz_height * rows))
                    axes_flat = axes.flatten() if n_feats > 1 else [axes]
                    for idx, feat in enumerate(plot_features):
                        ax = axes_flat[idx]
                        ax.hist(df_before_scale[feat], bins=30, alpha=0.6, color="#e74c3c", label="Before", density=True)
                        ax.hist(df_after_scale[feat], bins=30, alpha=0.6, color="#3498db", label="After", density=True)
                        ax.set_title(f"{feat}")
                        ax.legend(fontsize=8)
                    for idx in range(n_feats, len(axes_flat)):
                        axes_flat[idx].set_visible(False)
                    show_fig(axes_flat[0].figure, height_min=viz_height * rows)
    else:
        if scaling_method == "No Scaling":
            st.info("No scaling applied. Select a scaling method to preview the transformation.")
        else:
            st.info("No numeric features available for scaling preview.")

    # --- Class Imbalance Preview ---
    if target_type == "Classification":
        st.subheader("Class Imbalance Analysis")
        class_counts = df_clean[target_col].value_counts()
        total = len(df_clean)
        imbalance_df = pd.DataFrame({
            "Class": class_counts.index,
            "Count": class_counts.values,
            "Percentage": [round(100 * c / total, 2) for c in class_counts.values],
        })
        st.dataframe(imbalance_df)

        if use_plotly:
            fig_pie = px.pie(imbalance_df, values="Count", names="Class",
                             title="Class Distribution", color_discrete_sequence=plotly_palette)
            st.plotly_chart(fig_pie, use_container_width=True, key="fe_class_pie")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(imbalance_df["Count"], labels=imbalance_df["Class"], autopct="%1.1f%%",
                   colors=(palette if isinstance(palette, list) else plt.cm.Set1.colors)[:len(class_counts)])
            ax.set_title("Class Distribution")
            show_fig(ax.figure, height_min=5)

        ratio = class_counts.min() / class_counts.max()
        if ratio < 0.3:
            st.warning(f"**Significant imbalance detected.** Minority class is {ratio:.1%} of majority class. Consider applying SMOTE or oversampling.")
        elif ratio < 0.5:
            st.info(f"**Moderate imbalance.** Minority class is {ratio:.1%} of majority class. Class weights or resampling may help.")
        else:
            st.success("Classes are reasonably balanced.")

        if class_imbalance_method and class_imbalance_method != "None" and class_imbalance_method != "Class Weights Only":
            st.markdown(f"**After applying {class_imbalance_method}:**")
            if class_imbalance_method == "SMOTE":
                st.info("SMOTE (Synthetic Minority Oversampling) will generate synthetic minority samples using k-NN during model training. Neighbors: " + str(smote_neighbors))
            elif class_imbalance_method == "Random Oversampling":
                st.info("Random Oversampling will duplicate minority class samples to match the majority class count.")
            elif class_imbalance_method == "Random Undersampling":
                st.info("Random Undersampling will remove majority class samples to match the minority class count.")
        elif class_imbalance_method == "Class Weights Only":
            st.info("Class weights will be applied via `class_weight='balanced'` in the model — no resampling of data.")

    # --- Apply Encoding Button ---
    st.markdown("---")
    st.subheader("Apply Transformations")
    st.markdown("Click below to apply the selected encoding to the dataset. Scaling and imbalance handling are applied during model training.")

    if st.button("Apply Encoding to Dataset", key="apply_encoding_btn"):
        df_encoded_preview = df_model.copy()
        applied_encodings = {}
        fe_le_dict = {}
        fe_onehot_prefix_map = {}
        fe_encoding_methods = {}
        fe_dropped_cols = []
        for col in categorical_cols:
            method = encoding_methods.get(col, encoding_suggestions.get(col, "Label Encoding"))
            if method != "None (skip)":
                if method == "One-Hot Encoding":
                    try:
                        dummies = pd.get_dummies(df_encoded_preview[col], prefix=col, drop_first=False, dtype=int)
                        fe_onehot_prefix_map[col] = list(dummies.columns)
                        df_encoded_preview = pd.concat([df_encoded_preview.drop(columns=[col]), dummies], axis=1)
                        applied_encodings[col] = "One-Hot"
                        fe_encoding_methods[col] = "One-Hot Encoding"
                    except Exception:
                        pass
                elif method == "Label Encoding":
                    le = LabelEncoder()
                    df_encoded_preview[col] = le.fit_transform(df_encoded_preview[col].astype(str))
                    fe_le_dict[col] = le
                    applied_encodings[col] = "Label"
                    fe_encoding_methods[col] = "Label Encoding"
                elif method == "Drop":
                    df_encoded_preview = df_encoded_preview.drop(columns=[col], errors="ignore")
                    applied_encodings[col] = "Drop"
                    fe_encoding_methods[col] = "Drop"
                    fe_dropped_cols.append(col)
        st.session_state["df_encoded_preview"] = df_encoded_preview
        st.session_state["applied_encodings"] = applied_encodings
        st.session_state["fe_le_dict"] = fe_le_dict
        st.session_state["fe_onehot_prefix_map"] = fe_onehot_prefix_map
        st.session_state["fe_encoding_methods"] = fe_encoding_methods
        st.session_state["fe_dropped_cols"] = fe_dropped_cols
        st.success(f"Encoding applied! Columns changed: {', '.join(f'{k}→{v}' for k, v in applied_encodings.items())}")
        st.dataframe(df_encoded_preview.head(10))

    if "df_encoded_preview" in st.session_state:
        st.subheader("Encoded Data Preview")
        st.dataframe(st.session_state["df_encoded_preview"].head(10))

    # Export
    if "df_encoded_preview" in st.session_state:
        csv_enc = st.session_state["df_encoded_preview"].to_csv(index=False).encode("utf-8")
        st.download_button("Download encoded data as CSV", csv_enc, "encoded_data.csv", "text/csv", key="dl_encoded")

# ===========================================================================
# TAB 5 — FEATURE IMPORTANCE
# ===========================================================================
with tab5:
    st.header("Feature Importance")
    st.caption("Click the button below to train models and compute feature importance. Results are cached until you recompute.")

    fi_compute = st.button("Compute Feature Importance", key="fi_compute_btn", type="primary",
                           help="Trains Random Forest + XGBoost and runs 5-fold cross-validation. Takes time on large datasets.")

    if fi_compute or "fi_results" in st.session_state:
        if fi_compute:
            # --- Apply encoding and prepare data ---
            df_encoded = df_model.copy()
            fi_le_dict = {}
            fi_encoding_methods = {}
            for col in categorical_cols:
                if col in df_encoded.columns and col != target_col:
                    method = encoding_methods.get(col, encoding_suggestions.get(col, "Label Encoding"))
                    fi_encoding_methods[col] = method
                    if method == "Drop":
                        df_encoded = df_encoded.drop(columns=[col], errors="ignore")
                    elif method == "One-Hot Encoding":
                        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False, dtype=int)
                        df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                    elif method == "Label Encoding":
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                        fi_le_dict[col] = le

            feature_cols_encoded = [c for c in df_encoded.columns if c != target_col]
            X_fi = df_encoded[feature_cols_encoded]
            y_fi = df_encoded[target_col]

            fi_le_y = None
            if target_type == "Classification":
                fi_le_y = LabelEncoder()
                y_fi = pd.Series(fi_le_y.fit_transform(y_fi.astype(str)), name=target_col)

            non_numeric_cols = X_fi.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                X_fi = X_fi.drop(columns=non_numeric_cols)

            valid_mask = X_fi.notna().all(axis=1) & y_fi.notna()
            X_fi = X_fi[valid_mask]
            y_fi = y_fi[valid_mask]

            fi_results = {"feature_cols_encoded": feature_cols_encoded, "X_columns": list(X_fi.columns)}

            fi_progress = st.progress(0, text="Training models for feature importance...")
            fi_progress.progress(20, text="Training Random Forest...")
            with st.spinner("Training Random Forest for feature importance..."):
                if target_type == "Regression":
                    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
                model_rf.fit(X_fi, y_fi)
            fi_progress.progress(50, text="Computing feature importance...")

            fi_results["rf_importance"] = pd.Series(model_rf.feature_importances_, index=X_fi.columns).sort_values(ascending=True)

            if target_type == "Classification":
                fi_progress.progress(60, text="Training XGBoost...")
                with st.spinner("Training XGBoost for feature importance..."):
                    le_xgb = LabelEncoder()
                    y_encoded = le_xgb.fit_transform(y_fi.astype(str))
                    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
                    xgb.fit(X_fi, y_encoded)
                    fi_results["xgb_importance"] = pd.Series(xgb.feature_importances_, index=X_fi.columns).sort_values(ascending=True)

            # Cross-Validation using Pipeline to prevent data leakage
            fi_progress.progress(75, text="Running cross-validation...")
            if target_type == "Classification":
                cv_models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                    "KNN": KNeighborsClassifier(n_neighbors=5),
                }
            else:
                cv_models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                    "KNN": KNeighborsRegressor(n_neighbors=5),
                }

            cv_results = []
            with st.spinner("Running 5-fold cross-validation..."):
                for name, model in cv_models.items():
                    try:
                        scaler_fresh = get_scaler(scaling_method)
                        if scaler_fresh is not None:
                            pipe = Pipeline([("scaler", scaler_fresh), ("model", model)])
                            scores = cross_val_score(pipe, X_fi, y_fi, cv=5,
                                                     scoring="accuracy" if target_type == "Classification" else "r2")
                        else:
                            scores = cross_val_score(model, X_fi, y_fi, cv=5,
                                                     scoring="accuracy" if target_type == "Classification" else "r2")
                        cv_results.append({
                            "Model": name, "Mean Score": round(scores.mean(), 4),
                            "Std": round(scores.std(), 4), "Min": round(scores.min(), 4),
                            "Max": round(scores.max(), 4),
                        })
                    except Exception as e:
                        st.warning(f"Cross-validation failed for {name}: {e}")
            fi_results["cv_results"] = cv_results
            fi_progress.progress(100, text="Done!")
            fi_progress.empty()
            st.session_state["fi_results"] = fi_results

        # Display cached results
        fi_results = st.session_state["fi_results"]
        feature_cols_encoded = fi_results["feature_cols_encoded"]

        st.subheader("Random Forest Feature Importance")
        importance = fi_results["rf_importance"]
        if use_plotly:
            fig_pl = px.bar(x=importance.values, y=importance.index, orientation="h",
                            title="Random Forest Feature Importance", labels={"x": "Importance Score", "y": "Feature"},
                            color_discrete_sequence=["steelblue"])
            st.plotly_chart(fig_pl, use_container_width=True, key="plotly_feat_importance_rf_bar")
        else:
            fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols_encoded) * 0.5)))
            importance.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title("Random Forest Feature Importance")
            ax.set_xlabel("Importance Score")
            show_fig(ax.figure, height_min=5)

        st.subheader("Top Features")
        top_n = min(10, len(importance))
        top_features = importance.tail(top_n)
        for feature, score in top_features.items():
            st.write(f"- **{feature}**: {score:.4f}")

        if "xgb_importance" in fi_results:
            st.subheader("XGBoost Feature Importance")
            xgb_importance = fi_results["xgb_importance"]
            if use_plotly:
                fig_xgb = px.bar(x=xgb_importance.values, y=xgb_importance.index, orientation="h",
                                 title="XGBoost Feature Importance", labels={"x": "Importance Score", "y": "Feature"},
                                 color_discrete_sequence=["coral"])
                st.plotly_chart(fig_xgb, use_container_width=True, key="plotly_feat_importance_xgb")
            else:
                fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols_encoded) * 0.5)))
                xgb_importance.plot(kind="barh", ax=ax, color="coral")
                ax.set_title("XGBoost Feature Importance")
                ax.set_xlabel("Importance Score")
                show_fig(ax.figure, height_min=5)

        st.subheader("Cross-Validation Scores (5-Fold)")
        cv_results = fi_results.get("cv_results", [])
        if cv_results:
            cv_df = pd.DataFrame(cv_results).sort_values("Mean Score", ascending=False)
            st.dataframe(cv_df.style.highlight_max(subset=["Mean Score"]))

# ===========================================================================
# TAB 6 — MODEL TRAINING & EVALUATION
# ===========================================================================
with tab6:
    st.header("Model Training & Evaluation")
    st.subheader("Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key="test_size")
    with col2:
        random_state = st.number_input("Random State", value=42, min_value=0, max_value=1000, key="random_state")
    with col3:
        st.info(f"{'Classification' if target_type == 'Classification' else 'Regression'} models will be used")

    st.markdown("---")
    st.subheader("Model Hyperparameters")
    st.markdown("Expand each model to view and modify its parameters before training.")

    if target_type == "Classification":
        with st.expander("Logistic Regression — Parameters", expanded=False):
            lr_c = st.number_input("C (regularization inverse)", value=1.0, min_value=0.001, max_value=10000.0, step=0.1, format="%.3f", key="lr_c")
            lr_max_iter = st.number_input("Max Iterations", value=1000, min_value=100, max_value=10000, step=100, key="lr_max_iter")
            lr_penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"], index=0, key="lr_penalty")
            lr_solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga", "newton-cg", "sag"], index=0, key="lr_solver")
            _solver_supports_l1 = {"liblinear", "saga"}
            _solver_supports_enet = {"saga"}
            if lr_penalty == "l1" and lr_solver not in _solver_supports_l1:
                st.error(f"Solver '{lr_solver}' does not support l1 penalty. Use 'liblinear' or 'saga'.")
            if lr_penalty == "elasticnet" and lr_solver not in _solver_supports_enet:
                st.error(f"Solver '{lr_solver}' does not support elasticnet. Use 'saga'.")
            st.caption(f"C={lr_c}, max_iter={lr_max_iter}, penalty={lr_penalty}, solver={lr_solver}")

        with st.expander("Random Forest — Parameters", expanded=False):
            rf_n = st.number_input("n_estimators (trees)", value=100, min_value=10, max_value=2000, step=10, key="rf_n")
            rf_depth = st.number_input("max_depth", value=0, min_value=0, max_value=100, step=1, key="rf_depth", help="0 = no limit")
            rf_split = st.number_input("min_samples_split", value=2, min_value=2, max_value=50, step=1, key="rf_split")
            rf_leaf = st.number_input("min_samples_leaf", value=1, min_value=1, max_value=50, step=1, key="rf_leaf")
            rf_features = st.selectbox("max_features", ["sqrt", "log2", "None"], index=0, key="rf_features")
            st.caption(f"n_estimators={rf_n}, max_depth={rf_depth or 'None'}, min_split={rf_split}, min_leaf={rf_leaf}")

        with st.expander("SVM — Parameters", expanded=False):
            svm_c = st.number_input("C (regularization)", value=1.0, min_value=0.001, max_value=10000.0, step=0.1, format="%.3f", key="svm_c")
            svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0, key="svm_kernel")
            svm_gamma = st.selectbox("Gamma", ["scale", "auto"], index=0, key="svm_gamma")
            st.caption(f"C={svm_c}, kernel={svm_kernel}, gamma={svm_gamma}")

        with st.expander("XGBoost — Parameters", expanded=False):
            xgb_n = st.number_input("n_estimators (rounds)", value=100, min_value=10, max_value=2000, step=10, key="xgb_n_c")
            xgb_depth = st.number_input("max_depth", value=6, min_value=1, max_value=50, step=1, key="xgb_depth_c")
            xgb_lr = st.number_input("learning_rate", value=0.3, min_value=0.001, max_value=1.0, step=0.01, format="%.3f", key="xgb_lr_c")
            xgb_subsample = st.slider("subsample", 0.1, 1.0, 1.0, 0.05, key="xgb_sub_c")
            xgb_colsample = st.slider("colsample_bytree", 0.1, 1.0, 1.0, 0.05, key="xgb_col_c")

        with st.expander("Gradient Boosting — Parameters", expanded=False):
            gb_n = st.number_input("n_estimators (trees)", value=100, min_value=10, max_value=2000, step=10, key="gb_n_c")
            gb_depth = st.number_input("max_depth", value=3, min_value=1, max_value=50, step=1, key="gb_depth_c")
            gb_lr = st.number_input("learning_rate", value=0.1, min_value=0.001, max_value=1.0, step=0.01, format="%.3f", key="gb_lr_c")
            gb_subsample = st.slider("subsample", 0.1, 1.0, 1.0, 0.05, key="gb_sub_c")
            st.caption(f"n_estimators={gb_n}, max_depth={gb_depth}, lr={gb_lr}, subsample={gb_subsample}")

        with st.expander("KNN — Parameters", expanded=False):
            knn_n = st.number_input("n_neighbors", value=5, min_value=1, max_value=100, step=1, key="knn_n_c")
            knn_weights = st.selectbox("Weights", ["uniform", "distance"], index=0, key="knn_weights_c")
            knn_metric = st.selectbox("Metric", ["euclidean", "manhattan", "minkowski"], index=0, key="knn_metric_c")
            st.caption(f"n_neighbors={knn_n}, weights={knn_weights}, metric={knn_metric}")
    else:
        with st.expander("Linear Regression — Parameters", expanded=False):
            st.info("Linear Regression has no hyperparameters to tune (ordinary least squares).")
            st.caption("Fits: y = Xβ + ε using OLS. No regularization.")

        with st.expander("Random Forest — Parameters", expanded=False):
            rf_n = st.number_input("n_estimators (trees)", value=100, min_value=10, max_value=2000, step=10, key="rf_n_r")
            rf_depth = st.number_input("max_depth", value=0, min_value=0, max_value=100, step=1, key="rf_depth_r", help="0 = no limit")
            rf_split = st.number_input("min_samples_split", value=2, min_value=2, max_value=50, step=1, key="rf_split_r")
            rf_leaf = st.number_input("min_samples_leaf", value=1, min_value=1, max_value=50, step=1, key="rf_leaf_r")
            rf_features = st.selectbox("max_features", ["sqrt", "log2", "None"], index=0, key="rf_features_r")

        with st.expander("SVR — Parameters", expanded=False):
            svm_c = st.number_input("C (regularization)", value=1.0, min_value=0.001, max_value=10000.0, step=0.1, format="%.3f", key="svm_c_r")
            svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0, key="svm_kernel_r")
            svm_gamma = st.selectbox("Gamma", ["scale", "auto"], index=0, key="svm_gamma_r")
            svm_epsilon = st.number_input("Epsilon (SVR tube)", value=0.1, min_value=0.0, max_value=10.0, step=0.01, format="%.3f", key="svm_eps_r")

        with st.expander("XGBoost — Parameters", expanded=False):
            xgb_n = st.number_input("n_estimators (rounds)", value=100, min_value=10, max_value=2000, step=10, key="xgb_n_r")
            xgb_depth = st.number_input("max_depth", value=6, min_value=1, max_value=50, step=1, key="xgb_depth_r")
            xgb_lr = st.number_input("learning_rate", value=0.3, min_value=0.001, max_value=1.0, step=0.01, format="%.3f", key="xgb_lr_r")
            xgb_subsample = st.slider("subsample", 0.1, 1.0, 1.0, 0.05, key="xgb_sub_r")
            xgb_colsample = st.slider("colsample_bytree", 0.1, 1.0, 1.0, 0.05, key="xgb_col_r")

        with st.expander("Gradient Boosting — Parameters", expanded=False):
            gb_n = st.number_input("n_estimators (trees)", value=100, min_value=10, max_value=2000, step=10, key="gb_n_r")
            gb_depth = st.number_input("max_depth", value=3, min_value=1, max_value=50, step=1, key="gb_depth_r")
            gb_lr = st.number_input("learning_rate", value=0.1, min_value=0.001, max_value=1.0, step=0.01, format="%.3f", key="gb_lr_r")
            gb_subsample = st.slider("subsample", 0.1, 1.0, 1.0, 0.05, key="gb_sub_r")
            st.caption(f"n_estimators={gb_n}, max_depth={gb_depth}, lr={gb_lr}, subsample={gb_subsample}")

        with st.expander("KNN — Parameters", expanded=False):
            knn_n = st.number_input("n_neighbors", value=5, min_value=1, max_value=100, step=1, key="knn_n_r")
            knn_weights = st.selectbox("Weights", ["uniform", "distance"], index=0, key="knn_weights_r")
            knn_metric = st.selectbox("Metric", ["euclidean", "manhattan", "minkowski"], index=0, key="knn_metric_r")
            st.caption(f"n_neighbors={knn_n}, weights={knn_weights}, metric={knn_metric}")

    # --- Model Selection (before training) ---
    st.markdown("---")
    st.subheader("Select Models to Train")

    if target_type == "Classification":
        all_available_models = ["Logistic Regression", "Random Forest", "SVM", "XGBoost", "Gradient Boosting", "KNN"]
        model_descriptions = {
            "Logistic Regression": "Linear model, fast, interpretable. Good baseline.",
            "Random Forest": "Ensemble of decision trees. Robust, handles non-linearity.",
            "SVM": "Support Vector Machine. Effective in high-dimensional spaces.",
            "XGBoost": "Gradient boosting. Often best performance, slower to train.",
            "Gradient Boosting": "Sequential ensemble boosting trees. Strong predictive power.",
            "KNN": "K-Nearest Neighbors. Simple, instance-based, no training phase.",
        }
    else:
        all_available_models = ["Linear Regression", "Random Forest", "SVR", "XGBoost", "Gradient Boosting", "KNN"]
        model_descriptions = {
            "Linear Regression": "Ordinary least squares. Fast, interpretable. Good baseline.",
            "Random Forest": "Ensemble of decision trees. Robust, handles non-linearity.",
            "SVR": "Support Vector Regression. Effective in high-dimensional spaces.",
            "XGBoost": "Gradient boosting. Often best performance, slower to train.",
            "Gradient Boosting": "Sequential ensemble boosting trees. Strong predictive power.",
            "KNN": "K-Nearest Neighbors. Simple, instance-based, no training phase.",
        }

    model_cards = [{"Model": m, "Description": model_descriptions.get(m, "")} for m in all_available_models]
    st.dataframe(pd.DataFrame(model_cards), use_container_width=True)

    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        select_all = st.checkbox("Select All", value=True, key="select_all_models")
    with col_sel2:
        if select_all:
            selected_models = all_available_models
            st.caption(f"All {len(all_available_models)} models selected")
        else:
            selected_models = st.multiselect(
                "Choose individual models",
                options=all_available_models,
                default=[],
                key="selected_model_list",
            )

    if not selected_models:
        st.warning("Please select at least one model to train.")
        st.stop()

    st.info(f"**{len(selected_models)} model(s) selected:** {', '.join(selected_models)}")

    # --- Train Button ---
    st.markdown("---")
    train_clicked = st.button("Train Models", key="train_models_btn", type="primary",
                              help="Click to apply encoding, handle imbalance, scale features, and train selected models.")

    if train_clicked:
        # --- Prepare raw data, split FIRST, then encode on train only (prevents leakage) ---
        df_raw = df_model.copy()
        train_encoding_errors = []

        original_feature_cols = [c for c in df_raw.columns if c != target_col]
        original_dtypes = {c: str(df_raw[c].dtype) for c in original_feature_cols}

        encoding_methods_used = {}
        onehot_prefix_map = {}
        dropped_cols = []
        le_dict = {}

        fe_available = all(k in st.session_state for k in ["fe_le_dict", "fe_onehot_prefix_map", "fe_encoding_methods", "fe_dropped_cols"])
        if fe_available:
            encoding_methods_used = dict(st.session_state["fe_encoding_methods"])
            dropped_cols = list(st.session_state["fe_dropped_cols"])
            st.info("Using encoding from Feature Engineering tab.")

        # Apply drops
        for col in dropped_cols:
            if col in df_raw.columns:
                df_raw = df_raw.drop(columns=[col])

        feature_cols_raw = [c for c in df_raw.columns if c != target_col]
        X_raw = df_raw[feature_cols_raw]
        y_raw = df_raw[target_col]

        non_numeric_raw = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_raw:
            st.info(f"Dropping non-numeric columns: {', '.join(non_numeric_raw)}")
            X_raw = X_raw.drop(columns=non_numeric_raw)

        valid_mask = X_raw.notna().all(axis=1) & y_raw.notna()
        n_dropped = (~valid_mask).sum()
        if n_dropped > 0:
            st.warning(f"Dropping {n_dropped} rows with missing values ({round(100*n_dropped/len(X_raw),2)}% of data).")
            X_raw = X_raw[valid_mask].reset_index(drop=True)
            y_raw = y_raw[valid_mask].reset_index(drop=True)

        # --- Split FIRST, then encode on train only ---
        X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=test_size, random_state=random_state)

        le_y = None
        if target_type == "Classification":
            le_y = LabelEncoder()
            y_train = pd.Series(le_y.fit_transform(y_train.astype(str)), name=target_col)
            y_test = pd.Series(le_y.transform(y_test.astype(str)), name=target_col)

        # Fit encoders on training data only, then transform both
        _enc_cats = {}
        if fe_available:
            _enc_cats = dict(st.session_state.get("fe_encoding_methods", {}))
            _fe_le = dict(st.session_state.get("fe_le_dict", {}))
            _fe_oh = dict(st.session_state.get("fe_onehot_prefix_map", {}))

            for col, le_obj in _fe_le.items():
                if col in X_train.columns:
                    try:
                        le_new, X_train[col], X_test[col] = fit_label_encoder_safe(X_train[col], X_test[col])
                        le_dict[col] = le_new
                    except Exception as e:
                        train_encoding_errors.append(f"{col}: {e}")
                        X_train = X_train.drop(columns=[col], errors="ignore")
                        X_test = X_test.drop(columns=[col], errors="ignore")

            for col, dummy_cols in _fe_oh.items():
                if col in X_train.columns:
                    try:
                        dummies_train = pd.get_dummies(X_train[col], prefix=col, drop_first=False, dtype=int)
                        dummies_test = pd.get_dummies(X_test[col], prefix=col, drop_first=False, dtype=int)
                        for dc in dummy_cols:
                            if dc not in dummies_train.columns:
                                dummies_train[dc] = 0
                            if dc not in dummies_test.columns:
                                dummies_test[dc] = 0
                        dummies_train = dummies_train[dummy_cols]
                        dummies_test = dummies_test[dummy_cols]
                        X_train = pd.concat([X_train.drop(columns=[col]), dummies_train], axis=1)
                        X_test = pd.concat([X_test.drop(columns=[col]), dummies_test], axis=1)
                        onehot_prefix_map[col] = dummy_cols
                    except Exception as e:
                        train_encoding_errors.append(f"{col}: {e}")
                        X_train = X_train.drop(columns=[col], errors="ignore")
                        X_test = X_test.drop(columns=[col], errors="ignore")
        else:
            for col in categorical_cols:
                if col in X_train.columns and col != target_col:
                    method = encoding_methods.get(col, encoding_suggestions.get(col, "Label Encoding"))
                    encoding_methods_used[col] = method
                    if method == "Drop":
                        X_train = X_train.drop(columns=[col], errors="ignore")
                        X_test = X_test.drop(columns=[col], errors="ignore")
                        dropped_cols.append(col)
                    elif method == "One-Hot Encoding":
                        try:
                            dummies_train = pd.get_dummies(X_train[col], prefix=col, drop_first=False, dtype=int)
                            dummy_cols = list(dummies_train.columns)
                            dummies_test = pd.get_dummies(X_test[col], prefix=col, drop_first=False, dtype=int)
                            for dc in dummy_cols:
                                if dc not in dummies_test.columns:
                                    dummies_test[dc] = 0
                            dummies_test = dummies_test[dummy_cols]
                            X_train = pd.concat([X_train.drop(columns=[col]), dummies_train], axis=1)
                            X_test = pd.concat([X_test.drop(columns=[col]), dummies_test], axis=1)
                            onehot_prefix_map[col] = dummy_cols
                        except Exception as e:
                            train_encoding_errors.append(f"{col}: {e}")
                            X_train = X_train.drop(columns=[col], errors="ignore")
                            X_test = X_test.drop(columns=[col], errors="ignore")
                            dropped_cols.append(col)
                    elif method == "Label Encoding":
                        try:
                            le, X_train[col], X_test[col] = fit_label_encoder_safe(X_train[col], X_test[col])
                            le_dict[col] = le
                        except Exception as e:
                            train_encoding_errors.append(f"{col}: {e}")
                            X_train = X_train.drop(columns=[col], errors="ignore")
                            X_test = X_test.drop(columns=[col], errors="ignore")
                            dropped_cols.append(col)

        if train_encoding_errors:
            st.warning(f"Encoding errors: {'; '.join(train_encoding_errors)}")

        # Ensure test columns match train columns
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[X_train.columns]

        st.success(f"Encoding complete. Features: {X_train.shape[1]}, Train rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")

        # --- Handle Class Imbalance (training data only) ---
        use_smote = False
        use_ros = False
        use_rus = False
        apply_class_weights = False

        if target_type == "Classification" and class_imbalance_method:
            if class_imbalance_method == "SMOTE":
                use_smote = True
            elif class_imbalance_method == "Random Oversampling":
                use_ros = True
            elif class_imbalance_method == "Random Undersampling":
                use_rus = True
            elif class_imbalance_method == "Class Weights Only":
                apply_class_weights = True

        if use_smote or use_ros or use_rus:
            try:
                from imblearn.over_sampling import SMOTE, RandomOverSampler
                from imblearn.under_sampling import RandomUnderSampler

                if use_smote:
                    _y_train_series = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
                    _min_class_count = _y_train_series.value_counts().min()
                    k = min(smote_neighbors, len(X_train) - 1)
                    if _min_class_count <= k:
                        k = max(1, _min_class_count - 1)
                        st.warning(f"Smallest minority class has {_min_class_count} samples. Reducing SMOTE k to {k}.")
                    if k < 1:
                        st.error("SMOTE cannot proceed: smallest class has only 1 sample. Use Random Oversampling instead.")
                        use_smote = False
                    else:
                        sampler = SMOTE(k_neighbors=k, random_state=42)
                        X_train, y_train = sampler.fit_resample(X_train, y_train)
                        st.success(f"SMOTE applied (k={k}). New training shape: {X_train.shape}")
                elif use_ros:
                    sampler = RandomOverSampler(random_state=42)
                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                    st.success(f"Random Oversampling applied. New training shape: {X_train.shape}")
                elif use_rus:
                    sampler = RandomUnderSampler(random_state=42)
                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                    st.success(f"Random Undersampling applied. New training shape: {X_train.shape}")

                if use_plotly:
                    y_train_series = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
                    balanced_df = pd.DataFrame({"Class": y_train_series.value_counts().index, "Count": y_train_series.value_counts().values})
                    fig_bal = px.bar(balanced_df, x="Class", y="Count", title="Class Distribution After Resampling",
                                     color="Class", color_discrete_sequence=plotly_palette)
                    st.plotly_chart(fig_bal, use_container_width=True, key="model_balanced_dist")
                else:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    y_train_series = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
                    y_train_series.value_counts().plot(kind="bar", ax=ax, color="steelblue")
                    ax.set_title("Class Distribution After Resampling")
                    ax.set_ylabel("Count")
                    show_fig(ax.figure, height_min=4)

            except ImportError:
                st.error("imbalanced-learn is required. Install: `pip install imbalanced-learn`")
                st.info("Continuing without resampling.")
                use_smote = False
                use_ros = False
                use_rus = False

        # --- Apply Scaling (fit on training data only, transform both) ---
        st.subheader("Scaling Applied")
        if scaling_method != "No Scaling":
            df_before_model = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train, columns=X_train.columns)
            scaler_obj = scaler_map.get(scaling_method, StandardScaler())
            if scaler_obj is not None:
                X_train_scaled = scaler_obj.fit_transform(X_train)
                X_test_scaled = scaler_obj.transform(X_test)
                X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns) if isinstance(X_train, pd.DataFrame) or hasattr(X_train, 'columns') else X_train_scaled
                X_test = pd.DataFrame(X_test_scaled, columns=X_train.columns) if isinstance(X_test, pd.DataFrame) or hasattr(X_train, 'columns') else X_test_scaled
                df_after_model = pd.DataFrame(X_train_scaled, columns=X_train.columns) if hasattr(X_train, 'columns') else pd.DataFrame(X_train_scaled)
                before_model_stats = df_before_model.describe().T[["mean", "std", "min", "max"]].round(4)
                before_model_stats.columns = ["Mean (Before)", "Std (Before)", "Min (Before)", "Max (Before)"]
                after_model_stats = df_after_model.describe().T[["mean", "std", "min", "max"]].round(4)
                after_model_stats.columns = ["Mean (After)", "Std (After)", "Min (After)", "Max (After)"]
                model_stats_comp = pd.concat([before_model_stats, after_model_stats], axis=1)
                st.dataframe(model_stats_comp.style.background_gradient(subset=["Mean (After)", "Std (After)"], cmap="YlGn"))
            else:
                X_train = pd.DataFrame(X_train, columns=X_train.columns) if not isinstance(X_train, pd.DataFrame) else X_train
                X_test = pd.DataFrame(X_test, columns=X_train.columns) if not isinstance(X_test, pd.DataFrame) else X_test
        else:
            scaler_obj = None
            X_train = pd.DataFrame(X_train, columns=X_train.columns) if not isinstance(X_train, pd.DataFrame) else X_train
            X_test = pd.DataFrame(X_test, columns=X_train.columns) if not isinstance(X_test, pd.DataFrame) else X_test
            st.info("No scaling applied.")

        # --- Build & Train Selected Models ---
        if target_type == "Classification":
            cw = "balanced" if apply_class_weights else None
            all_models_pool = {
                "Logistic Regression": LogisticRegression(
                    C=lr_c, max_iter=lr_max_iter, penalty=lr_penalty if lr_penalty != "none" else None,
                    solver=lr_solver, random_state=random_state, class_weight=cw,
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=rf_n, max_depth=rf_depth if rf_depth > 0 else None,
                    min_samples_split=rf_split, min_samples_leaf=rf_leaf,
                    max_features=rf_features if rf_features != "None" else None,
                    random_state=random_state, class_weight=cw,
                ),
                "SVM": SVC(C=svm_c, kernel=svm_kernel, gamma=svm_gamma, probability=True,
                           random_state=random_state, class_weight=cw),
                "XGBoost": XGBClassifier(
                    n_estimators=xgb_n, max_depth=xgb_depth, learning_rate=xgb_lr,
                    subsample=xgb_subsample, colsample_bytree=xgb_colsample,
                    random_state=random_state, eval_metric="logloss",
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=gb_n, max_depth=gb_depth, learning_rate=gb_lr,
                    subsample=gb_subsample, random_state=random_state,
                ),
                "KNN": KNeighborsClassifier(
                    n_neighbors=knn_n, weights=knn_weights, metric=knn_metric,
                ),
            }
        else:
            all_models_pool = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(
                    n_estimators=rf_n, max_depth=rf_depth if rf_depth > 0 else None,
                    min_samples_split=rf_split, min_samples_leaf=rf_leaf,
                    max_features=rf_features if rf_features != "None" else None,
                    random_state=random_state,
                ),
                "SVR": SVR(C=svm_c, kernel=svm_kernel, gamma=svm_gamma, epsilon=svm_epsilon),
                "XGBoost": XGBRegressor(
                    n_estimators=xgb_n, max_depth=xgb_depth, learning_rate=xgb_lr,
                    subsample=xgb_subsample, colsample_bytree=xgb_colsample,
                    random_state=random_state,
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=gb_n, max_depth=gb_depth, learning_rate=gb_lr,
                    subsample=gb_subsample, random_state=random_state,
                ),
                "KNN": KNeighborsRegressor(
                    n_neighbors=knn_n, weights=knn_weights, metric=knn_metric,
                ),
            }

        models = {k: v for k, v in all_models_pool.items() if k in selected_models}

        results = {}
        trained_models = {}
        training_errors = {}

        model_progress = st.progress(0, text="Training models...")
        with st.spinner("Training models..."):
            for i, (name, model) in enumerate(models.items()):
                model_progress.progress((i) / len(models), text=f"Training {name}...")
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metric = accuracy_score(y_test, y_pred) if target_type == "Classification" else r2_score(y_test, y_pred)
                    results[name] = metric
                    trained_models[name] = (model, y_pred)
                except Exception as e:
                    training_errors[name] = str(e)
                    st.warning(f"Training failed for **{name}**: {e}")
        model_progress.progress(1.0, text="All models trained!")
        model_progress.empty()

        if training_errors:
            st.error(f"Failed to train: {', '.join(training_errors.keys())}. Check hyperparameter settings.")

        st.session_state["trained_models"] = trained_models
        st.session_state["scaler"] = scaler_obj
        st.session_state["feature_cols"] = list(X_train.columns)
        st.session_state["trained_target_col"] = target_col
        st.session_state["trained_target_type"] = target_type
        st.session_state["le_target"] = le_y
        st.session_state["le_dict"] = le_dict
        st.session_state["X_columns"] = list(X_train.columns)
        st.session_state["y_test"] = y_test
        st.session_state["X_test"] = X_test
        st.session_state["results"] = results
        st.session_state["original_feature_cols"] = original_feature_cols
        st.session_state["encoding_methods_used"] = encoding_methods_used
        st.session_state["onehot_prefix_map"] = onehot_prefix_map
        st.session_state["dropped_cols"] = dropped_cols
        st.session_state["original_dtypes"] = original_dtypes

    # --- Display Results (from session_state if available) ---
    if "trained_models" in st.session_state and st.session_state["trained_models"]:
        trained_models = st.session_state["trained_models"]
        results = st.session_state["results"]
        y_test = st.session_state["y_test"]
        X_test = st.session_state["X_test"]

        st.subheader("Trained Model Parameters")
        for name, (m, _) in trained_models.items():
            with st.expander(f"{name} — fitted parameters", expanded=False):
                try:
                    params = m.get_params()
                    params_df = pd.DataFrame({"Parameter": params.keys(), "Value": [str(v) for v in params.values()]})
                    st.dataframe(params_df, use_container_width=True)
                except Exception:
                    st.write(m)

        st.subheader("Model Comparison")
        metric_name = "Accuracy" if target_type == "Classification" else "R² Score"
        results_df = pd.DataFrame({"Model": results.keys(), metric_name: results.values()}).sort_values(by=metric_name, ascending=False)
        st.dataframe(results_df.style.highlight_max(subset=[results_df.columns[1]]))

        if use_plotly:
            fig_cmp = px.bar(results_df, x=metric_name, y="Model", orientation="h",
                             title=f"Model {metric_name} Comparison",
                             color=metric_name, color_continuous_scale="Viridis")
            st.plotly_chart(fig_cmp, use_container_width=True, key="plotly_model_comparison")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            metric_col = results_df.columns[1]
            colors = ["#2ecc71" if v == results_df[metric_col].max() else "#3498db" for v in results_df[metric_col]]
            ax.barh(results_df["Model"], results_df[metric_col], color=colors)
            ax.set_xlabel(metric_col)
            ax.set_title(f"Model {metric_col} Comparison")
            show_fig(ax.figure, height_min=4)

        st.subheader("Detailed Results")
        selected_model = st.selectbox("Select model for detailed view", list(trained_models.keys()), key="selected_model")
        model, y_pred = trained_models[selected_model]

        if target_type == "Classification":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Classification Report**")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).T)
            with col2:
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                if use_plotly:
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                       title=f"Confusion Matrix - {selected_model}",
                                       labels=dict(x="Predicted", y="Actual"))
                    st.plotly_chart(fig_cm, use_container_width=True, key="plotly_confusion_matrix")
                else:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"Confusion Matrix - {selected_model}")
                    show_fig(ax.figure, height_min=4)

            st.subheader("ROC Curves")
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                for name, (m, _) in trained_models.items():
                    if hasattr(m, "predict_proba"):
                        y_prob = m.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
                ax.plot([0, 1], [0, 1], "k--", label="Random")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curves (Binary)")
                ax.legend()
                show_fig(ax.figure, height_min=5)
            else:
                from sklearn.preprocessing import label_binarize
                classes = np.unique(y_test)
                y_test_bin = label_binarize(y_test, classes=classes)
                fig, ax = plt.subplots(figsize=(8, 6))
                for name, (m, _) in trained_models.items():
                    if hasattr(m, "predict_proba"):
                        y_prob = m.predict_proba(X_test)
                        mean_auc = 0
                        for i in range(n_classes):
                            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                            roc_auc_val = auc(fpr, tpr)
                            mean_auc += roc_auc_val
                            ax.plot(fpr, tpr, label=f"{name} class={classes[i]} (AUC = {roc_auc_val:.3f})")
                        mean_auc /= n_classes
                        ax.plot([], [], label=f"{name} macro-avg (AUC = {mean_auc:.3f})")
                ax.plot([0, 1], [0, 1], "k--", label="Random")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curves (One-vs-Rest, Multiclass)")
                ax.legend(fontsize=8)
                show_fig(ax.figure, height_min=5)

        else:
            n_test = len(y_test)
            n_features_count = X_test.shape[1]
            r2 = r2_score(y_test, y_pred)
            adj_r2 = 1 - (1 - r2) * (n_test - 1) / (n_test - n_features_count - 1) if n_test > n_features_count + 1 else r2
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = np.mean(np.abs(y_test - y_pred))

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("R²", f"{r2:.4f}")
            col_m2.metric("Adjusted R²", f"{adj_r2:.4f}")
            col_m3.metric("RMSE", f"{rmse:.4f}")
            col_m4.metric("MAE", f"{mae:.4f}")

            st.subheader(f"Actual vs Predicted - {selected_model}")
            if use_plotly:
                fig_ap = px.scatter(x=y_test.values, y=y_pred, title=f"Actual vs Predicted - {selected_model}",
                                   labels={"x": "Actual", "y": "Predicted"}, opacity=0.5, color_discrete_sequence=["#3498db"])
                fig_ap.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                                 line=dict(color="red", dash="dash"))
                st.plotly_chart(fig_ap, use_container_width=True, key="plotly_actual_vs_predicted")
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred, alpha=0.5, color="#3498db")
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"Actual vs Predicted - {selected_model}")
                ax.legend()
                show_fig(ax.figure, height_min=5)

            st.subheader("Residual Plot")
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y_pred, residuals, alpha=0.5, color="#e74c3c")
            ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            ax.set_title(f"Residual Plot - {selected_model}")
            show_fig(ax.figure, height_min=4)

            st.markdown("---")
            st.subheader("OLS Regression Summary (statsmodels)")
            try:
                X_train_sm = sm.add_constant(X_train)
                X_test_sm = sm.add_constant(X_test, has_constant="add")
                ols_model = sm.OLS(y_train, X_train_sm).fit()

                st.markdown("**Coefficients & Significance**")
                coef_df = pd.DataFrame({
                    "Feature": ["Intercept"] + list(X_train.columns),
                    "Coefficient": ols_model.params.values.round(6),
                    "Std Error": ols_model.bse.values.round(6),
                    "t-Statistic": ols_model.tvalues.values.round(4),
                    "P-Value": ols_model.pvalues.values.round(6),
                    "Lower CI (95%)": ols_model.conf_int()[0].values.round(6),
                    "Upper CI (95%)": ols_model.conf_int()[1].values.round(6),
                })
                coef_df["Significant"] = coef_df["P-Value"].apply(lambda p: "Yes" if p < 0.05 else "No")
                st.dataframe(
                    coef_df.style.map(
                        lambda x: "background-color: #90EE90" if x == "Yes" else "background-color: #FFB6C1",
                        subset=["Significant"],
                    ).background_gradient(subset=["P-Value"], cmap="YlOrRd_r")
                )

                st.markdown("**Model Summary Statistics**")
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("R² (train)", f"{ols_model.rsquared:.4f}")
                col_s2.metric("Adj R² (train)", f"{ols_model.rsquared_adj:.4f}")
                col_s3.metric("F-Statistic", f"{ols_model.fvalue:.4f}")
                col_s4, col_s5, col_s6 = st.columns(3)
                col_s4.metric("AIC", f"{ols_model.aic:.2f}")
                col_s5.metric("BIC", f"{ols_model.bic:.2f}")
                col_s6.metric("Log-Likelihood", f"{ols_model.llf:.2f}")
                st.write(f"**No. Observations:** {int(ols_model.nobs)}")
                st.write(f"**Durbin-Watson:** {round(sm.stats.stattools.durbin_watson(ols_model.resid), 4)}")

                st.subheader("Coefficient Confidence Intervals (95%)")
                coef_plot = coef_df[coef_df["Feature"] != "Intercept"].copy()
                coef_plot = coef_plot.sort_values("Coefficient")
                fig, ax = plt.subplots(figsize=(10, max(4, len(coef_plot) * 0.5)))
                y_pos = range(len(coef_plot))
                ax.barh(y_pos, coef_plot["Coefficient"], color="steelblue", alpha=0.7, label="Coefficient")
                ax.errorbar(
                    coef_plot["Coefficient"], y_pos,
                    xerr=[coef_plot["Coefficient"] - coef_plot["Lower CI (95%)"], coef_plot["Upper CI (95%)"] - coef_plot["Coefficient"]],
                    fmt="none", ecolor="black", capsize=3, label="95% CI",
                )
                ax.set_yticks(y_pos)
                ax.set_yticklabels(coef_plot["Feature"])
                ax.axvline(x=0, color="red", linewidth=0.8, linestyle="--")
                ax.set_xlabel("Coefficient Value")
                ax.set_title("OLS Coefficients with 95% Confidence Intervals")
                ax.legend()
                show_fig(ax.figure, height_min=4)

                st.markdown("---")
                st.subheader("Regression Diagnostics")
                ols_fitted = ols_model.fittedvalues
                ols_resid = ols_model.resid
                ols_std_resid = pd.Series(ols_model.get_influence().resid_studentized_internal)

                st.subheader("Q-Q Plot (Residual Normality)")
                fig, ax = plt.subplots(figsize=(7, 6))
                sm.qqplot(ols_std_resid, line="45", ax=ax, markersize=3, alpha=0.5)
                ax.set_title("Q-Q Plot of Standardized Residuals")
                ax.get_lines()[0].set_markerfacecolor("#3498db")
                ax.get_lines()[0].set_markeredgecolor("#3498db")
                show_fig(ax.figure, height_min=5)

                resid_sample = ols_std_resid.dropna()
                if len(resid_sample) >= 3:
                    sample_for_sw = resid_sample.sample(min(500, len(resid_sample)), random_state=42)
                    w_stat, w_p = stats.shapiro(sample_for_sw)
                    st.write(f"**Shapiro-Wilk on residuals:** W = {w_stat:.4f}, p = {w_p:.6f}")
                    if w_p < 0.05:
                        st.warning("Residuals are **not** normally distributed (reject H0).")
                    else:
                        st.success("Residuals appear normally distributed (fail to reject H0).")

                st.subheader("Scale-Location Plot (Heteroscedasticity Check)")
                sqrt_std_resid = np.sqrt(np.abs(ols_std_resid))
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(ols_fitted, sqrt_std_resid, alpha=0.5, color="#9b59b6")
                try:
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    lowess_fit = lowess(sqrt_std_resid, ols_fitted, frac=0.6)
                    ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], "r-", linewidth=2, label="LOWESS")
                    ax.legend()
                except Exception:
                    pass
                ax.set_xlabel("Fitted Values")
                ax.set_ylabel("√|Standardized Residuals|")
                ax.set_title("Scale-Location Plot")
                show_fig(ax.figure, height_min=4)

                st.subheader("Cook's Distance (Influential Points)")
                st.markdown("Points with Cook's D > 4/n are considered influential.")
                from statsmodels.stats.outliers_influence import OLSInfluence
                influence = OLSInfluence(ols_model)
                cooks_d = influence.cooks_distance[0]
                n_obs = len(cooks_d)
                threshold = 4 / n_obs
                cooks_df = pd.DataFrame({
                    "Observation": range(n_obs),
                    "Cook's D": cooks_d.round(6),
                    "Influential": ["Yes" if d > threshold else "No" for d in cooks_d],
                }).sort_values("Cook's D", ascending=False)
                n_influential = cooks_df["Influential"].value_counts().get("Yes", 0)
                st.write(f"**Influential points (D > {threshold:.4f}):** {n_influential} out of {n_obs}")
                fig, ax = plt.subplots(figsize=(10, max(4, min(8, n_obs * 0.05))))
                colors = ["#e74c3c" if d > threshold else "#3498db" for d in cooks_d]
                ax.bar(range(n_obs), cooks_d, color=colors, width=1.0)
                ax.axhline(y=threshold, color="red", linewidth=1, linestyle="--", label=f"Threshold (4/n = {threshold:.4f})")
                ax.set_xlabel("Observation Index")
                ax.set_ylabel("Cook's Distance")
                ax.set_title("Cook's Distance Plot")
                ax.legend()
                show_fig(ax.figure, height_min=4)
                if n_influential > 0:
                    st.dataframe(cooks_df[cooks_df["Influential"] == "Yes"].head(20))

                st.subheader("Leverage vs Residuals Plot")
                leverage = influence.hat_matrix_diag
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(leverage, ols_std_resid, alpha=0.5, color="#e67e22")
                ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
                ax.axhline(y=2, color="red", linewidth=0.5, linestyle=":")
                ax.axhline(y=-2, color="red", linewidth=0.5, linestyle=":")
                ax.set_xlabel("Leverage")
                ax.set_ylabel("Standardized Residuals")
                ax.set_title("Leverage vs Standardized Residuals")
                show_fig(ax.figure, height_min=4)

                st.subheader("Variance Inflation Factors (VIF)")
                vif_data = []
                for i, col in enumerate(X_train.columns):
                    vif_val = variance_inflation_factor(X_train.values, i)
                    vif_data.append({"Feature": col, "VIF": round(vif_val, 2)})
                vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
                st.dataframe(
                    vif_df.style.background_gradient(subset=["VIF"], cmap="YlOrRd", vmin=1, vmax=20)
                )
                high_vif = vif_df[vif_df["VIF"] > 10]
                if not high_vif.empty:
                    st.warning(f"VIF > 10 detected for: {', '.join(high_vif['Feature'].tolist())}")

            except Exception as e:
                st.error(f"OLS analysis failed: {e}")

    if "trained_models" not in st.session_state or not st.session_state.get("trained_models"):
        st.info("Click **Train Models** to start training.")

# ===========================================================================
# TAB 7 — PREDICT
# ===========================================================================
with tab7:
    st.header("Predict New Values")
    required_keys = ["trained_models", "original_feature_cols", "encoding_methods_used",
                     "onehot_prefix_map", "dropped_cols", "original_dtypes"]
    if not all(k in st.session_state for k in required_keys):
        st.warning("Please train models in the 'Model Training' tab first.")
        st.stop()

    trained_models = st.session_state["trained_models"]
    scaler = st.session_state["scaler"]
    target_col = st.session_state["trained_target_col"]
    target_type = st.session_state["trained_target_type"]
    le_target = st.session_state["le_target"]
    le_dict = st.session_state["le_dict"]
    X_columns = st.session_state["X_columns"]
    original_feature_cols = st.session_state["original_feature_cols"]
    encoding_methods_used = st.session_state["encoding_methods_used"]
    onehot_prefix_map = st.session_state["onehot_prefix_map"]
    dropped_cols = st.session_state["dropped_cols"]
    original_dtypes = st.session_state["original_dtypes"]

    model_name = st.selectbox("Select model for prediction", list(trained_models.keys()), key="model_name")
    model = trained_models[model_name][0]

    st.markdown("---")
    st.subheader("Enter feature values")
    st.caption("Showing original column names and values. Encoding is applied automatically before prediction.")

    # Collect raw input from user using ORIGINAL column names
    raw_input = {}
    input_cols = st.columns(min(3, len(original_feature_cols)))
    for idx, col in enumerate(original_feature_cols):
        if col in dropped_cols:
            continue
        with input_cols[idx % len(input_cols)]:
            method = encoding_methods_used.get(col, "Label Encoding")
            if col in le_dict:
                # Label-encoded categorical: show original category names (exclude __unknown__ sentinel)
                options = [c for c in le_dict[col].classes_.tolist() if c != "__unknown__"]
                chosen = st.selectbox(col, options, key=f"pred_{col}")
                raw_input[col] = chosen
            elif method == "One-Hot Encoding":
                # One-hot encoded: show single selectbox with original categories
                cat_options = [c.replace(f"{col}_", "") for c in onehot_prefix_map.get(col, [])]
                if not cat_options:
                    cat_options = df[col].dropna().unique().tolist() if col in df.columns else []
                chosen = st.selectbox(col, cat_options, key=f"pred_{col}")
                raw_input[col] = chosen
            elif col in numeric_cols or original_dtypes.get(col, "") in ["float64", "int64", "int32", "float32"]:
                # Numeric column
                if col in df.columns and df[col].notna().any():
                    default_val = float(df[col].median())
                else:
                    default_val = 0.0
                raw_input[col] = st.number_input(col, value=default_val, key=f"pred_{col}")
            else:
                raw_input[col] = st.text_input(col, value="", key=f"pred_{col}")

    # Helper: apply encoding to raw input
    def encode_raw_input(raw_input_dict):
        """Apply the same encoding pipeline used during training to a single raw input."""
        row = {}
        for col in original_feature_cols:
            if col in dropped_cols:
                continue
            method = encoding_methods_used.get(col, "Label Encoding")
            val = raw_input_dict.get(col)

            if col in le_dict:
                # Label encoding — safe for unseen categories
                row[col] = transform_with_unknown(le_dict[col], np.array([str(val)]))[0]
            elif method == "One-Hot Encoding":
                # One-hot: set the matching dummy to 1, rest to 0
                dummy_cols = onehot_prefix_map.get(col, [])
                for dummy in dummy_cols:
                    row[dummy] = 1 if dummy == f"{col}_{val}" else 0
            elif method == "None (numeric)":
                row[col] = val
            else:
                # Numeric or unencoded
                row[col] = val
        return row

    def encode_batch_df(pred_df):
        """Apply encoding to a batch DataFrame with original column names.
        Also handles NaN values using the same strategy as training (median/mode)."""
        result = pred_df.copy()

        # Impute NaN values using median (numeric) / mode (categorical) — mirrors training
        for col in result.columns:
            if result[col].isnull().any():
                if result[col].dtype in ["float64", "int64", "float32", "int32"]:
                    result[col] = result[col].fillna(result[col].median())
                else:
                    mode_val = result[col].mode()
                    if len(mode_val) > 0:
                        result[col] = result[col].fillna(mode_val[0])

        for col in original_feature_cols:
            if col in dropped_cols:
                if col in result.columns:
                    result = result.drop(columns=[col], errors="ignore")
                continue
            if col not in result.columns:
                continue
            method = encoding_methods_used.get(col, "Label Encoding")

            if col in le_dict:
                result[col] = transform_with_unknown(le_dict[col], result[col].values)
            elif method == "One-Hot Encoding":
                dummy_cols = onehot_prefix_map.get(col, [])
                for dummy in dummy_cols:
                    cat_val = dummy.replace(f"{col}_", "")
                    result[dummy] = (result[col] == cat_val).astype(int)
                result = result.drop(columns=[col], errors="ignore")
        # Drop any remaining non-numeric columns
        non_num = result.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_num:
            result = result.drop(columns=non_num, errors="ignore")
        return result

    st.markdown("---")
    st.subheader("Or upload a CSV for batch prediction")
    st.caption("Upload a CSV with the original column names (same as your training data before encoding).")
    pred_file = st.file_uploader("Upload CSV", type=["csv"], key="pred_csv")

    if st.button("Predict", key="predict_btn"):
        if pred_file is not None:
            pred_df = pd.read_csv(pred_file)
            missing_cols = set(original_feature_cols) - set(pred_df.columns)
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}")
                st.stop()
            pred_df = pred_df[original_feature_cols]
            encoded_df = encode_batch_df(pred_df)
            # Ensure column order matches training
            for col in X_columns:
                if col not in encoded_df.columns:
                    encoded_df[col] = 0
            encoded_df = encoded_df[X_columns]
            if scaler is not None:
                pred_scaled = pd.DataFrame(scaler.transform(encoded_df), columns=X_columns)
            else:
                pred_scaled = encoded_df

            predictions = model.predict(pred_scaled)

            # Inverse transform to original labels
            if le_target is not None:
                predictions = le_target.inverse_transform(predictions.astype(int))

            pred_df["Prediction"] = predictions

            # Add probabilities for classification
            if target_type == "Classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(pred_scaled)
                classes = le_target.classes_ if le_target is not None else model.classes_
                for i, cls in enumerate(classes):
                    pred_df[f"P({cls})"] = (proba[:, i] * 100).round(2).astype(str) + "%"

            st.subheader("Batch Predictions")
            st.dataframe(pred_df, use_container_width=True)
            csv = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions as CSV", csv, "predictions.csv", "text/csv", key="dl_predictions")
        else:
            encoded_row = encode_raw_input(raw_input)
            input_df = pd.DataFrame([encoded_row])
            # Ensure column order matches training
            for col in X_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[X_columns]
            if scaler is not None:
                input_scaled = pd.DataFrame(scaler.transform(input_df), columns=X_columns)
            else:
                input_scaled = input_df
            prediction = model.predict(input_scaled)

            # Inverse transform to original label
            if le_target is not None:
                prediction = le_target.inverse_transform(prediction.astype(int))

            st.success(f"**Predicted {target_col}:** {prediction[0]}")

            # Show probabilities for classification
            if target_type == "Classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled)[0]
                classes = le_target.classes_ if le_target is not None else model.classes_
                st.subheader("Class Probabilities")
                proba_df = pd.DataFrame({
                    "Class": classes,
                    "Probability": [f"{p*100:.1f}%" for p in proba],
                    "Confidence": proba,
                }).sort_values("Confidence", ascending=False)
                st.dataframe(proba_df[["Class", "Probability"]].reset_index(drop=True), use_container_width=True)

                # Highlight the predicted class
                best_idx = np.argmax(proba)
                if proba[best_idx] >= 0.7:
                    st.info(f"High confidence ({proba[best_idx]*100:.1f}%) for class **{classes[best_idx]}**")
                elif proba[best_idx] >= 0.4:
                    st.warning(f"Moderate confidence ({proba[best_idx]*100:.1f}%) — results may vary")
                else:
                    st.error(f"Low confidence ({proba[best_idx]*100:.1f}%) — consider collecting more data or tuning the model")

# ===========================================================================
# TAB 8 — DATA PROFILING
# ===========================================================================
with tab8:
    st.header("Data Profiling & Summary Statistics")
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include="all").T)

    st.subheader("Per-Column Profile")
    profile_df = compute_data_profiling(df_fingerprint(df), df.to_dict("list"))
    st.dataframe(profile_df, use_container_width=True)

    st.subheader("Memory Usage")
    mem = df.memory_usage(deep=True).sum()
    st.write(f"**Total memory usage:** {mem / 1024:.2f} KB ({mem / (1024**2):.2f} MB)")

    st.subheader("Data Quality Score")
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    quality_score = round(100 * (1 - missing_cells / total_cells), 2) if total_cells > 0 else 100
    st.metric("Data Quality Score", f"{quality_score}%")
    if quality_score >= 95:
        st.success("Excellent data quality")
    elif quality_score >= 85:
        st.info("Good data quality")
    elif quality_score >= 70:
        st.warning("Fair data quality — consider handling missing values")
    else:
        st.error("Poor data quality — significant missing data")

    st.subheader("Duplicate Detection")
    n_dup = df.duplicated().sum()
    st.write(f"**Duplicate rows:** {n_dup} ({round(100 * n_dup / len(df), 2) if len(df) > 0 else 0}%)")
    if n_dup > 0:
        if st.button("Show duplicate rows", key="show_dupes"):
            st.dataframe(df[df.duplicated(keep=False)])

# ===========================================================================
# SIDEBAR — NETWORK INFO & ABOUT
# ===========================================================================
st.sidebar.header("Access")
st.sidebar.markdown("`http://localhost:8501`")
st.sidebar.caption("App is bound to localhost only. No network exposure.")
st.sidebar.markdown("---")

st.sidebar.header("About")
st.sidebar.markdown("""
**Universal Data Analysis Dashboard v15**

Upload any CSV or Excel file for:
1. Correlation Matrix & Variable Selection
2. Data Cleaning & Outlier Detection
3. Statistical Tests (10+ tests, auto-recommended correction)
4. Visualizations (Matplotlib or Plotly)
5. Feature Engineering (Encoding, Scaling, Imbalance)
6. Feature Importance & Cross-Validation (Pipeline CV)
7. Model Training & Evaluation (6 models)
8. Predict new values (original column names)
9. Data Profiling & Quality Assessment

**V15:** Train-only encoding (no data leakage), auto-correction recommendation, SMOTE guard, hyperparameter validation.
""")
