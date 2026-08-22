# Universal Data Analysis Dashboard v14 — Complete User Handbook

---

## Table of Contents

1. [About This App](#1-about-this-app)
2. [Installation and Running](#2-installation-and-running)
3. [Sidebar Controls](#3-sidebar-controls)
4. [Tab 0 — Correlation Matrix](#4-tab-0--correlation-matrix)
5. [Tab 1 — Data Cleaning](#5-tab-1--data-cleaning)
6. [Tab 2 — Statistical Tests](#6-tab-2--statistical-tests)
7. [Tab 3 — Visualizations](#7-tab-3--visualizations)
8. [Tab 4 — Feature Engineering](#8-tab-4--feature-engineering)
9. [Tab 5 — Feature Importance](#9-tab-5--feature-importance)
10. [Tab 6 — Model Training and Evaluation](#10-tab-6--model-training-and-evaluation)
11. [Tab 7 — Predict](#11-tab-7--predict)
12. [Tab 8 — Data Profiling](#12-tab-8--data-profiling)
13. [Glossary of All Terms](#13-glossary-of-all-terms)
14. [Tips and Best Practices](#14-tips-and-best-practices)

---

## 1. About This App

The **Universal Data Analysis Dashboard v14** is an all-in-one data analysis and machine learning dashboard. You upload a CSV or Excel file, and the app automatically performs a complete end-to-end analysis — from understanding your data, to cleaning it, to building predictive models, to making real predictions. You do not need to write a single line of code.

Think of it as having a team of data scientists working for you: one who profiles your data, one who cleans it, one who runs statistical tests, one who builds charts, one who engineers features, one who trains machine learning models, and one who makes predictions. All of them work together in one easy-to-use web interface.

### What This App Does (In Plain English)

Imagine you have a spreadsheet full of data — sales figures, patient records, survey results, housing prices, or anything else. This app lets you:

1. **Understand your data** — it reads your file and tells you what columns exist, what types they are, how many values are missing, and gives you a full statistical profile of every column (mean, median, standard deviation, skewness, kurtosis, and more).
2. **Clean your data** — it finds missing values and fills them in using smart methods (median for numbers, most-frequent value for categories), and detects unusual values (outliers) using the IQR method. You can download the cleaned version.
3. **Discover relationships** — it runs over 10 statistical tests to figure out which columns are actually related to the column you want to predict. It tells you in plain English: "this feature is useful" or "this feature can be dropped."
4. **See your data visually** — it generates charts (bar charts, box plots, scatter plots, histograms, pairplots) so you can see patterns with your eyes, not just numbers. You can choose between static charts (Matplotlib) or interactive charts (Plotly) that you can zoom, hover over, and download.
5. **Prepare data for machine learning** — it converts text categories into numbers (encoding), normalizes number scales so all features are on the same footing (scaling), and fixes imbalanced classes so your model trains fairly.
6. **Rank feature importance** — it uses Random Forest and XGBoost to tell you which features matter most for your prediction.
7. **Train machine learning models** — it builds up to 6 different models simultaneously, compares them, and shows you which one performs best.
8. **Diagnose model quality** — for regression problems, it shows detailed OLS regression summaries, Q-Q plots, Cook's distance, leverage plots, and other diagnostics so you know if your model is trustworthy.
9. **Make real predictions** — you enter values using the original column names from your CSV (no need to understand encoding), and the app predicts the outcome. You can also upload a CSV of new data for batch predictions on hundreds or thousands of rows at once.
10. **Profile data quality** — it gives you a data quality score, memory usage, duplicate detection, and detailed per-column statistics.

### What's New in V14 (All Version History Combined)

This version includes improvements from all previous versions, clubbed together here for convenience:

| Feature | What It Does | Added In |
|---|---|---|
| **Target Type Always Visible** | The Classification/Regression selector is always shown in the sidebar with smart auto-detection based on the selected predictor variable. If you pick a column with 20 or fewer unique numeric values, it assumes Classification. If more than 20, it assumes Regression. You can always override this. | V13, V14 |
| **Gradient Boosting Model** | Added GradientBoostingClassifier/Regressor with tunable parameters: number of trees (n_estimators), maximum depth (max_depth), learning rate, and subsample fraction. Gradient Boosting builds trees sequentially, where each new tree corrects the errors of the previous one. | V14 |
| **KNN Model** | K-Nearest Neighbors (KNeighborsClassifier/Regressor) is now available with tunable neighbors count, weighting scheme, and distance metric. KNN is a simple, intuitive model that classifies a data point based on what its closest neighbors are. | V14 |
| **NaN Handling in Training** | Rows with missing values are now automatically dropped before model training, with a warning message telling you how many rows were removed. | V13, V14 |
| **Predict Tab Redesign** | The prediction interface now shows your original column names and values (not encoded integers). Encoding is applied automatically behind the scenes. | V14 |
| **Model Selection Before Training** | You can select which models to train (all or individual ones) after configuring feature engineering. Only selected models are trained, saving time. | V12 |
| **Before and After Scaling Preview** | Side-by-side statistics comparison showing how your data changes after scaling (Mean, Std, Min, Max, Q1, Median, Q3), with distribution plots. | V10, V12 |
| **String Encoding Fix** | Unencoded string columns are auto-filtered before model training to prevent errors. | V12 |
| **Encoding Selection** | Choose One-Hot, Label, or Auto encoding per categorical column, with smart suggestions based on cardinality. | V9, V10 |
| **Class Imbalance Handling** | SMOTE, Random Oversampling, Random Undersampling, or class weights to handle imbalanced target classes, with before/after visualization. | V9, V10 |
| **Scaling Distribution Plots** | Visual comparison of feature distributions before and after scaling. | V10 |
| **Standardization and Scaling** | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer (Yeo-Johnson), QuantileTransformer (Normal/Uniform). | V9 |

---

## 2. Installation and Running

### What You Need (Prerequisites)

Before running the app, you need Python installed on your computer (Python 3.8 or newer is recommended). Then install all required libraries with one command:

```bash
pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn xgboost statsmodels plotly imbalanced-learn
```

**What each library does:**

- **streamlit** — The web framework that powers the dashboard interface. It turns Python scripts into interactive web apps.
- **pandas** — Data manipulation library. It reads CSVs and Excel files, handles tables (DataFrames), and lets you filter, sort, and transform data.
- **numpy** — Numerical computing library. It handles arrays, mathematical operations, and random number generation.
- **matplotlib** — Static chart library. It creates images of plots (bar charts, scatter plots, histograms, etc.).
- **seaborn** — Statistical visualization library built on matplotlib. It makes prettier charts with less code and has built-in support for statistical plots.
- **scipy** — Scientific computing library. It runs statistical tests like T-Test, Chi-Square, ANOVA, correlation tests, and normality tests.
- **scikit-learn** — The main machine learning library. It provides models (Logistic Regression, Random Forest, SVM, etc.), preprocessing tools (scaling, encoding), and evaluation metrics (accuracy, R-squared, etc.).
- **xgboost** — Extreme Gradient Boosting library. It provides a high-performance gradient boosting model that often wins data science competitions.
- **statsmodels** — Statistical modeling library. It provides OLS (Ordinary Least Squares) regression, VIF (Variance Inflation Factor), and diagnostic tests (Breusch-Pagan, Durbin-Watson).
- **plotly** — Interactive chart library. It creates zoomable, hoverable, downloadable charts as an alternative to static matplotlib plots.
- **imbalanced-learn** — Handles imbalanced datasets. It provides SMOTE, Random Oversampling, and Random Undersampling to fix situations where one class has far more examples than another.

### Running the App

Once everything is installed, run this command in your terminal or command prompt:

```bash
streamlit run universal_data_analysis_v14.py
```

A web browser will open automatically. If it does not, open your browser manually and go to:

- **Local URL:** `http://localhost:8501` — accessible only on your own machine
- **Network URL:** `http://<your-ip>:8501` — accessible from any device on your WiFi or local network (the app shows this link in the sidebar)

### Supported File Formats

- **CSV files** (`.csv`) — The most common data format. The app tries multiple encodings (UTF-8, Latin-1, CP1252, ISO-8859-1) automatically if one does not work.
- **Excel files** (`.xlsx`, `.xls`) — Microsoft Excel spreadsheet files.

Maximum recommended file size is about 2 GB, but performance depends on your computer's memory.

---

## 3. Sidebar Controls

The sidebar is the control panel on the left side of the screen. It contains all the settings that control how the app analyzes your data. Every change you make here affects all the tabs.

### Upload Data

This is where you upload your CSV or Excel file. Click "Browse files" and select your data file. Once uploaded, the sidebar will show:

- The file name and size (in MB)
- The number of rows and columns
- A success message confirming the file loaded correctly

If the file is empty or corrupt, you will see an error message.

### Dataset Size (Sampling)

If your dataset has more than 10,000 rows, the app will offer to sample your data for faster analysis. This is useful because computing correlations and running statistical tests on millions of rows can be slow.

- **Sample for faster analysis** — Check this box to use only a subset of your rows
- **Sample size (rows)** — Use the slider to choose how many rows to sample (between 1,000 and 50,000, or the total size if smaller)

When sampling is enabled, the app runs analysis on the sample, which is much faster while still giving you meaningful results.

### Column Info

This section automatically detects and displays:

- **Rows** — Total number of data rows
- **Columns** — Total number of columns
- **Numeric** — How many columns contain numbers (like age, price, temperature)
- **Categorical** — How many columns contain text/categories (like "Yes/No", "Red/Blue/Green", city names)

### Visualization Style

These settings control the look and feel of all charts in the app:

- **Color Palette** — Choose from: default, viridis, plasma, coolwarm, Set1, Set2, Paired, Dark2, tab10, tab20. Different palettes use different color schemes. "default" uses a standard colorful palette. "viridis" and "plasma" are colorblind-friendly. "coolwarm" goes from blue (low) to red (high).
- **Plot Theme** — Choose the background style of matplotlib plots: default, darkgrid (gray grid on dark background), whitegrid (white background with grid lines — the default), dark, white, ticks.
- **Default plot height** — Slider from 4 to 12 inches. Controls how tall charts are.
- **Default plot width** — Slider from 6 to 20 inches. Controls how wide charts are.
- **Plot backend** — Choose between:
  - **Matplotlib (static)** — Creates static images. Good for reports and publications.
  - **Plotly (interactive)** — Creates interactive charts you can zoom, pan, hover over to see exact values, and download as images. Better for exploration.

### Select Predictor (Target) Variable

This is the most important setting. You must choose which column you want to predict. This column is called the "target" or "dependent variable" or "response variable."

- **Choose target column** — A dropdown showing all column names from your data. Pick the column you want to predict (e.g., "Price", "Survived", "Diagnosis").

### Target Type

After you select a target column, the app automatically detects whether this is a **Classification** or **Regression** problem:

- **Classification** — The target is a category or label (like "Yes/No", "Spam/Not Spam", "Red/Blue/Green"). Also selected automatically if a numeric column has 20 or fewer unique values (like a score from 1-10).
- **Regression** — The target is a continuous number (like price, temperature, salary). Selected automatically if a numeric column has more than 20 unique values.

You can override the auto-detection by clicking a different option. A warning will appear if you switch from the auto-detected type.

**Why this matters:** Classification models predict categories (e.g., "Will this customer churn? Yes or No"). Regression models predict numbers (e.g., "How much will this house sell for?"). Using the wrong type will produce meaningless results.

### Drop Variables

Sometimes you have columns that should not be part of the analysis — like ID numbers, names, or dates that are just labels. Use this multiselect to choose columns to remove from the analysis entirely.

- **Select columns to drop from analysis** — Click and type to search, then click a column name to add it to the drop list. Dropped columns will not appear in any analysis, charts, or models.

### Feature Encoding

This section controls how categorical (text) columns are converted into numbers that machine learning models can understand.

- **Encoding Mode**:
  - **Auto (Smart Suggestion)** — The app analyzes each categorical column and suggests the best encoding method based on the number of unique values (cardinality).
  - **Manual Selection** — You choose the encoding method for each column yourself using multiselect dropdowns.

When in Manual mode, you will see four categories:

- **One-Hot Encoding columns** — Creates a new binary (0/1) column for each category. Best for columns with 15 or fewer unique values (like "Color: Red, Blue, Green" becomes three columns: Color_Red, Color_Blue, Color_Green, each with 0 or 1).
- **Label Encoding columns** — Maps each category to a number (Red=0, Blue=1, Green=2). Best for columns with many unique values or for ordinal data where order matters.
- **Drop (remove from analysis)** — For text fields, IDs, or columns with so many unique values (more than 50) that encoding is not meaningful (like email addresses or free-text descriptions).
- **None (skip encoding)** — Keep the column as-is without encoding.

**Smart suggestions work like this:**
- 2 unique values -> Label Encoding
- 3 to 15 unique values -> One-Hot Encoding
- 16 to 50 unique values -> Label Encoding
- More than 50 unique values -> Drop
- Long text (average length > 30 characters) -> Drop
- Numeric columns -> None (already numeric)

### Feature Scaling

Scaling normalizes your numeric features so they are all on a similar scale. This is important because many machine learning algorithms perform poorly when features have wildly different ranges (e.g., "Age" ranges from 0-100 but "Salary" ranges from 20,000-200,000).

Choose from these scaling methods:

- **StandardScaler (z-score)** — The most common method. Transforms each feature to have a mean of 0 and a standard deviation of 1. Formula: (value - mean) / standard deviation. Best for normally distributed data.
- **MinMaxScaler (0-1)** — Scales all values to fall between 0 and 1. Formula: (value - min) / (max - min). Good for algorithms that require bounded inputs, but sensitive to outliers.
- **RobustScaler (median/IQR)** — Uses the median and interquartile range instead of mean and standard deviation. Much more resistant to outliers. Best when your data has significant extreme values.
- **MaxAbsScaler** — Divides each value by the maximum absolute value. Preserves sparsity (zeros stay as zeros). Good for sparse data like text data represented as numbers.
- **PowerTransformer (Yeo-Johnson)** — Transforms data to be as close to a normal (bell curve) distribution as possible. Handles zero and negative values. Good for heavily skewed data.
- **QuantileTransformer (Normal)** — Maps data to a normal distribution using quantiles. Very robust to outliers. Good when data is far from normal.
- **QuantileTransformer (Uniform)** — Maps data to a uniform distribution (all values equally likely). Good for rank-based algorithms.
- **No Scaling** — No transformation applied. Acceptable for tree-based models (Random Forest, XGBoost, Gradient Boosting) because they are scale-invariant (they do not care about the range of values).

### Class Imbalance (Classification Only)

This section appears only when your target type is Classification. It helps when one category in your target is much more common than another (e.g., 95% "No" and 5% "Yes").

- **None** — Do nothing. Use if your classes are roughly equal.
- **SMOTE** — Synthetic Minority Oversampling Technique. Creates new synthetic (fake but realistic) examples of the minority class using k-Nearest Neighbors. The **SMOTE neighbors** slider (1-10) controls how many neighbors are used. Default is 5.
- **Random Oversampling** — Randomly duplicates minority class examples until both classes have the same count. Simple but can cause overfitting.
- **Random Undersampling** — Randomly removes majority class examples until both classes have the same count. Simple but can lose valuable information.
- **Class Weights Only** — Does not change the data. Instead, tells the model to pay more attention to the minority class during training by assigning higher "importance" (weight) to minority examples. Applied via `class_weight='balanced'`.

### Access Links

The sidebar shows two links at the bottom:

- **Local** — `http://localhost:8501` — Use this to access the app from the same computer it is running on.
- **Network** — `http://<your-ip>:8501` — Use this to access the app from any other device on the same WiFi or local network (like your phone or another computer).

---

## 4. Tab 0 — Correlation Matrix

This tab helps you understand how all your numeric features relate to each other and to the target variable. Correlation is a number between -1 and +1 that measures how two variables move together.

### Full Correlation Matrix

A correlation matrix is a grid (heatmap) where each cell shows the correlation between two numeric features. The color coding makes it easy to spot patterns at a glance:

- **Red (positive)** — When one goes up, the other goes up. A value of +1 means they move perfectly together.
- **Blue (negative)** — When one goes up, the other goes down. A value of -1 means they move perfectly opposite.
- **White/light (near zero)** — No relationship. The variables move independently.

The diagonal is always +1.00 because every feature is perfectly correlated with itself.

The upper triangle is hidden (masked) to avoid redundancy since the matrix is symmetric.

**How to read it:**
- Values close to +1 or -1 are strong correlations
- Values close to 0 are weak or no correlations
- The app uses Pearson correlation, which measures linear relationships specifically

**Pearson Correlation (Pearson r):**
- Measures the strength and direction of the **linear** relationship between two variables
- Ranges from -1 (perfect negative linear relationship) to +1 (perfect positive linear relationship)
- 0 means no linear relationship
- Assumes both variables are approximately normally distributed
- Example: If height and weight have Pearson r = 0.8, it means taller people tend to weigh more, and the relationship is roughly a straight line

### Correlation with Target

This is a horizontal bar chart showing how each numeric feature correlates with the target variable. Bars are color-coded:

- **Red** — Strong correlation (absolute r > 0.7). These features are very strongly related to the target.
- **Blue** — Moderate correlation (absolute r > 0.3). These features have a meaningful relationship with the target.
- **Gray** — Weak correlation (absolute r <= 0.3). These features have little linear relationship with the target.

A table below the chart lists each feature's Pearson r value and absolute r value, sorted by strength.

**What to do with this:**
- Features with strong correlation (red) are likely very useful for prediction
- Features with weak correlation (gray) might not help much, but could still be useful in combination with other features

### Spearman Rank Correlation with Target

Spearman correlation (rho) measures the strength of the **monotonic** relationship between two variables. A monotonic relationship means that as one variable increases, the other tends to increase (or decrease), but not necessarily in a straight line.

**Difference from Pearson:**
- Pearson measures straight-line (linear) relationships
- Spearman measures "always going in the same direction" relationships, even if the rate of change is not constant
- Spearman is more robust to outliers because it uses ranks (order) instead of actual values

**Example:** If you rank students by study time and exam score, and students who study more always score higher (even if the improvement slows down at high study times), Spearman would be high even if Pearson is lower.

The table shows Spearman rho, p-value, and whether the relationship is statistically significant (green = significant, pink = not significant).

### Highly Correlated Feature Pairs

This section identifies pairs of features that are highly correlated with **each other** (not with the target). This is important because:

- When two features are highly correlated, they carry redundant information
- Including both can confuse some models and make interpretation harder
- This is called **multicollinearity**

You can adjust the **Correlation threshold** slider (0.50 to 0.99, default 0.85) to control what counts as "highly correlated." Pairs above this threshold are listed in a table.

**What to do:** If two features are very highly correlated (say 0.95), consider keeping only one of them.

### Suggested Variables to Drop

The app automatically suggests which features to drop based on two criteria:

1. **Highly correlated with another feature** — If a feature is strongly correlated with a feature that is already being kept, the new feature is redundant.
2. **Very low correlation with the target** — If a feature has almost no relationship with the target variable, it is unlikely to help with prediction.

You can adjust two sliders:
- **Max inter-feature correlation to keep** (default 0.85) — Features more correlated than this with an already-kept feature are flagged
- **Min absolute correlation with target to keep** (default 0.05) — Features less correlated than this with the target are flagged

After reviewing the suggestions, you can select features to drop and click "Apply Drop" to remove them from the analysis.

### Variance Inflation Factor (VIF)

VIF measures how much the variance of a regression coefficient is inflated because of correlation with other features. It is another way to detect multicollinearity.

**How to interpret VIF:**
- **VIF = 1** — No correlation with other features. Perfect.
- **VIF 1-5** — Moderate correlation. Usually acceptable.
- **VIF 5-10** — High correlation. Problematic. Consider removing some features.
- **VIF > 10** — Very high correlation. Serious problem. The app will warn you about these features.

**What to do:** If a feature has VIF > 10, it means its information is largely captured by other features. Consider dropping it or combining it with other features.

---

## 5. Tab 1 — Data Cleaning

This tab helps you clean your data by handling missing values and detecting outliers.

### Raw Data Preview

Shows the first 10 rows of your current dataset so you can see what it looks like.

### Row/Column/Missing Summary

Three metric cards at the top show:
- **Rows** — Total number of data rows
- **Columns** — Total number of columns
- **Missing Values** — Total number of cells that are empty or contain NaN (Not a Number)

### Missing Values by Column

If any columns have missing values, a bar chart shows how many values are missing in each column, along with a table listing the exact counts.

**What are missing values?**
Missing values (NaN) occur when data was not collected, was lost, or is not applicable. For example, if a survey question was optional, some respondents might leave it blank.

### Handle Missing Values

Choose how to fill in the missing values:

- **Median (numeric) / Mode (categorical)** — For number columns, replaces missing values with the median (the middle value when sorted). For text columns, replaces with the mode (the most frequently occurring value). **This is usually the best choice** because the median is not affected by outliers.
- **Mean (numeric) / Mode (categorical)** — For number columns, replaces missing values with the mean (average). For text columns, replaces with the mode. The mean can be distorted by outliers, so use this only if your data has no extreme values.
- **Drop Rows** — Removes any row that has even one missing value. This can lose a lot of data if many rows have missing values somewhere. Use this only if you have plenty of data and the missing rows are small in number.

**What is imputation?** Imputation is the process of filling in missing values instead of deleting the rows that have them. The Median and Mean methods are both forms of imputation. The idea is that you replace the missing value with a reasonable estimate based on the rest of the data.

After applying, the cleaned data info shows the new row count, column count, and remaining missing values.

### Export Cleaned Data

Click "Download cleaned data as CSV" to download the cleaned version of your dataset to your computer.

### Outlier Detection (IQR Method)

The app uses the **IQR (Interquartile Range) method** to detect outliers. Here is how it works in plain English:

1. Sort all values in a column from lowest to highest
2. Find **Q1** (the 25th percentile — the value below which 25% of data falls)
3. Find **Q3** (the 75th percentile — the value below which 75% of data falls)
4. Calculate **IQR** = Q3 - Q1 (the range of the middle 50% of data)
5. Any value below Q1 - 1.5 * IQR is an outlier (too low)
6. Any value above Q3 + 1.5 * IQR is an outlier (too high)

**Example:** If Q1 = 25, Q3 = 75, then IQR = 50. The lower fence = 25 - 1.5*50 = -50. The upper fence = 75 + 1.5*50 = 150. Any value below -50 or above 150 would be an outlier.

The table shows for each numeric column:
- Q1, Q3, IQR values
- Lower and Upper fences (the boundaries)
- Number and percentage of outliers

**What are outliers?** Outliers are data points that are unusually high or low compared to the rest of the data. They could be genuine extreme values (a billionaire in an income survey) or errors (a temperature of 999 degrees). The IQR method is a standard statistical technique for identifying these unusual points.

---

## 6. Tab 2 — Statistical Tests

This tab runs statistical tests to determine which features are actually related to the target variable. Think of statistical tests as formal experiments: you start with a hypothesis, run a test, and get a number (p-value) that tells you whether your hypothesis is supported by the data.

### Understanding P-Values (Critical Concept)

Almost every test in this tab produces a **p-value**. Here is what it means in plain English:

- The **p-value** is a number between 0 and 1 that tells you how likely it is that the pattern you see happened by random chance
- A **small p-value** (typically < 0.05) means the pattern is very unlikely to be random — it is probably a real relationship
- A **large p-value** (>= 0.05) means the pattern could easily have happened by chance — it is probably not a real relationship

**The significance level (alpha)** is the threshold you set for what counts as "small enough." The app defaults to 0.05 (5%), but you can change it with the slider (0.01 to 0.10).

- **p-value < alpha** => Statistically significant. The relationship is real (not due to chance).
- **p-value >= alpha** => Not statistically significant. The relationship might be due to chance.

**Important:** "Statistically significant" does not automatically mean the relationship is strong or useful. It just means it is unlikely to be random. A very weak relationship can still be "significant" if you have enough data.

### Classification Target Tests

When your target is a classification variable, the app runs these tests:

#### Chi-Square Test of Independence

**What question it answers:** Is there a relationship between a categorical feature and the target? (Are they independent or associated?)

**How it works:** It creates a contingency table (a cross-tabulation) counting how often each combination of categories occurs, then compares the observed counts to what you would expect if there were no relationship.

**What the result means:**
- **Significant (p < alpha):** The feature and target are associated. Knowing the feature helps predict the target.
- **Not significant (p >= alpha):** The feature and target appear independent. The feature may not help predict the target.

**What to do:** Keep features that show significant associations with the target.

#### Cramer's V (Effect Size)

**What question it answers:** How **strong** is the association between a categorical feature and the target? (Chi-Square tells you if there IS a relationship; Cramer's V tells you HOW STRONG it is.)

**How to interpret the value:**
- **0.0 to 0.1** — Negligible association
- **0.1 to 0.3** — Weak association
- **0.3 to 0.5** — Moderate association
- **0.5 to 1.0** — Strong association

**What to do:** Use Cramer's V alongside Chi-Square. A significant Chi-Square with a high Cramer's V means the feature is both statistically significant AND practically important.

#### T-Test / Welch's T-Test / Mann-Whitney U

**What question it answers:** Do the two groups (defined by the target) have different average values of this numeric feature?

The app automatically chooses the right test:
1. First, it checks if the data in each group is **normally distributed** (bell-shaped) using the Shapiro-Wilk test
2. If both groups are normal -> **Welch's T-Test** (does not assume equal variances)
3. If either group is NOT normal -> **Mann-Whitney U** (a non-parametric alternative that does not assume normality)

**What the result means:**
- **Significant (p < alpha):** The two groups have genuinely different distributions for this feature. The feature is useful for distinguishing between the groups.
- **Not significant (p >= alpha):** The two groups are similar for this feature. The feature may not help distinguish between the groups.

**What to do:** Significant features are good candidates for predicting the target.

#### Kolmogorov-Smirnov (KS) Test

**What question it answers:** Do the two groups have the same overall distribution shape? This is more general than the T-Test — it checks not just the mean but the entire distribution.

**How to interpret:**
- **Significant (p < alpha):** The two groups have different distributions.
- **Not significant (p >= alpha):** The two groups have similar distributions.

**What to do:** A significant KS test means the feature's distribution differs between target classes, suggesting it is useful for prediction.

#### Mutual Information

**What question it answers:** How much does knowing one variable tell you about the other? Unlike correlation, Mutual Information can detect ANY kind of relationship, not just linear ones.

**How to interpret:**
- **0** — The variables are completely independent (knowing one tells you nothing about the other)
- **Higher values** — The variables are more dependent (knowing one gives you more information about the other)

**What to do:** Features with higher Mutual Information scores are more useful for prediction. The app shows a bar chart ranking all features by their Mutual Information score with the target.

### Regression Target Tests

When your target is a regression (continuous numeric) variable, the app runs these tests:

#### Pearson Correlation

**What question it answers:** Is there a linear (straight-line) relationship between a numeric feature and the target?

**How to interpret:**
- **r close to +1** — Strong positive linear relationship (as feature increases, target increases)
- **r close to -1** — Strong negative linear relationship (as feature increases, target decreases)
- **r close to 0** — No linear relationship

**The p-value** tells you if this correlation is statistically significant or could have happened by chance.

**What to do:** Significant features with high absolute r values are strong predictors.

#### Spearman Rank Correlation

**What question it answers:** Is there a monotonic (consistently increasing or decreasing) relationship between a feature and the target? This is less restrictive than Pearson — the relationship does not need to be a straight line.

**What to do:** If Spearman is significant but Pearson is not, the feature has a non-linear monotonic relationship with the target. It is still useful.

#### Kendall's Tau

**What question it answers:** Is there an ordinal association between the feature and the target? Kendall's Tau is similar to Spearman but uses a different calculation method. It is more robust with small sample sizes and handles tied ranks better.

**How to interpret:** Values range from -1 to +1, similar to Pearson and Spearman.

#### One-Way ANOVA / Kruskal-Wallis

**What question it answers:** Do different groups (defined by a categorical feature) have different means of the target variable?

The app automatically chooses:
1. If all groups are normally distributed -> **One-Way ANOVA** (Analysis of Variance)
2. If any group is NOT normal -> **Kruskal-Wallis** (non-parametric alternative)

**What to do:** A significant result means the categorical feature affects the target. For example, if "Department" is a feature and "Salary" is the target, a significant ANOVA means different departments have different average salaries.

#### Shapiro-Wilk Test (Normality)

**What question it answers:** Is the target variable normally distributed (bell-shaped)?

**How to interpret:**
- **Significant (p < alpha):** The data is NOT normally distributed.
- **Not significant (p >= alpha):** The data appears normally distributed.

The app also shows a histogram of the target variable with a red normal distribution curve overlaid so you can visually check.

**Why this matters:** Many statistical tests and models assume normality. If the target is not normal, you might need to transform it (e.g., log transform) or use non-parametric methods.

#### Durbin-Watson Test (Autocorrelation)

**What question it answers:** Are the residuals (prediction errors) correlated with each other? This matters especially for time series data where values at one time point might be related to values at the next.

**How to interpret:**
- **Near 2.0** — No autocorrelation. Good.
- **Below 1.5** — Positive autocorrelation (errors tend to follow each other).
- **Above 2.5** — Negative autocorrelation (errors tend to alternate).

**Why this matters:** Autocorrelation violates the assumptions of many regression models and can make predictions unreliable.

#### Breusch-Pagan Test (Homoscedasticity)

**What question it answers:** Is the variance of the residuals constant across all predicted values? This is called **homoscedasticity**.

**How to interpret:**
- **Significant (p < alpha):** Heteroscedasticity detected — the variance of errors changes as predictions change. This is a problem.
- **Not significant (p >= alpha):** Homoscedasticity holds — the variance is constant. Good.

**Why this matters:** If variance is not constant, your model's predictions will be more accurate for some ranges of data than others, and confidence intervals will be wrong.

#### Mutual Information (Regression)

Same concept as in classification, but uses regression-specific Mutual Information. Measures any kind of statistical dependency between features and the continuous target.

### Summary and Recommendations

At the bottom of this tab, the app summarizes:
- **Significant features to keep** — Features that showed statistically significant relationships with the target across one or more tests.
- **Non-significant features to consider dropping** — Features that did not show significant relationships.

This gives you a data-driven starting point for deciding which features to include in your model.

### Feature Correlation Matrix

A correlation heatmap of just the numeric features (not the target), displayed at the bottom of this tab. This helps you spot multicollinearity among features.

---

## 7. Tab 3 — Visualizations

This tab generates charts to help you see patterns in your data visually.

### Target Variable Distribution

The first set of charts shows the distribution of your target variable:

- **For Classification targets:** A bar chart showing how many rows belong to each class (e.g., 500 "Yes" and 500 "No"). This helps you see if your classes are balanced or imbalanced.
- **For Regression targets:** A histogram showing the distribution of values (e.g., most house prices are around $300K, with fewer at the extremes).

### Feature vs Target Charts

For the first numeric feature, the app also shows:
- **For Classification:** A **box plot** showing how the feature's values differ across target classes
- **For Regression:** A **scatter plot** showing the relationship between the feature and the target

### What is a Box Plot?

A box plot (also called a box-and-whisker plot) shows:
- The **box** spans from Q1 (25th percentile) to Q3 (75th percentile) — the middle 50% of data
- The **line inside the box** is the median (50th percentile)
- The **whiskers** extend to show the range of most data (Q1 - 1.5*IQR to Q3 + 1.5*IQR)
- **Dots beyond the whiskers** are outliers

If the boxes for different classes do not overlap much, the feature is good at separating the classes.

### Feature Distributions

Use the multiselect to choose which features to visualize. The app shows:

- **For numeric features with Classification target:** Box plots showing the distribution of each feature split by target class
- **For numeric features with Regression target:** Scatter plots of each feature against the target
- **For categorical features with Classification target:** Stacked bar charts showing the percentage of each target class within each category
- **For categorical features with Regression target:** Bar charts showing the mean target value for each category

### What is a Scatter Plot?

A scatter plot shows individual data points as dots on an X-Y grid. The x-axis shows one variable, the y-axis shows another. Patterns in the dots reveal relationships:
- Dots forming a line going up = positive correlation
- Dots forming a line going down = negative correlation
- Dots scattered randomly = no correlation

### What is a Histogram?

A histogram groups numeric values into bins (ranges) and shows how many data points fall into each bin as a bar. It shows the shape of the distribution:
- Bell-shaped = normally distributed
- Skewed left = most values are high, with a tail toward low values
- Skewed right = most values are low, with a tail toward high values
- Two peaks = bimodal (might indicate two different groups in your data)

### Pairplot

If you have 8 or fewer features and 500 or fewer rows, the app offers a **pairplot**. A pairplot creates a grid of scatter plots showing every pair of numeric features against each other, with histograms (or density curves) on the diagonal.

- For classification targets, dots are colored by the target class, making it easy to see which feature pairs separate the classes best.
- **Warning:** Pairplots can be slow with many features. The app limits this to 6 features maximum.

### Plot Backend Choice

You can switch between:
- **Matplotlib (static)** — Creates image files. Better for saving and including in reports or presentations.
- **Plotly (interactive)** — Creates interactive charts you can zoom into, hover over for exact values, pan around, and download as PNG. Better for exploration.

---

## 8. Tab 4 — Feature Engineering

This tab lets you configure encoding, scaling, and class imbalance handling before model training. These are the crucial preprocessing steps that turn raw data into something machine learning models can work with.

### Categorical Encoding Preview

Shows a table of all categorical columns with:
- The column name and data type
- Number of unique values
- The encoding method that will be applied (or was suggested)
- Whether your selection matches the app's suggestion

If your encoding differs from the app's suggestion, a warning is shown. This is informational — you can still use your own choice.

### Feature Scaling Preview

After you select a scaling method in the sidebar, this section shows:

1. **Before Scaling** — The raw numeric data with its statistics (Mean, Std, Min, Q1, Median, Q3, Max)
2. **After Scaling** — The same data after the scaling transformation, with the same statistics
3. **Comparison Table** — Side-by-side comparison of each feature's statistics before and after scaling
4. **Distribution Comparison** — Overlay histograms showing how each feature's distribution changed after scaling

You can select which features to compare in the distribution plots.

### Class Imbalance Analysis (Classification Only)

For classification targets, this section shows:

1. **Class Distribution Table** — How many rows belong to each class and what percentage
2. **Pie Chart or Bar Chart** — Visual representation of the class distribution
3. **Imbalance Warning** — The app calculates the ratio of the smallest class to the largest class:
   - **Ratio < 0.3** — Significant imbalance. Strongly recommended to apply SMOTE or oversampling.
   - **Ratio 0.3 to 0.5** — Moderate imbalance. Class weights or resampling may help.
   - **Ratio >= 0.5** — Classes are reasonably balanced.

If you selected SMOTE, Random Oversampling, or Random Undersampling, the app explains what will happen during training.

### Apply Encoding to Dataset

Click this button to apply your encoding selections to the dataset. The app will:

1. Apply One-Hot Encoding to selected columns (creating binary 0/1 columns)
2. Apply Label Encoding to selected columns (converting categories to integers)
3. Drop columns you selected for removal
4. Show you the encoded data preview
5. Allow you to download the encoded data as a CSV

**Important:** Scaling and class imbalance handling are NOT applied here. They are applied during model training (Tab 6) to prevent data leakage.

---

## 9. Tab 5 — Feature Importance

This tab tells you which features matter most for predicting your target variable. It uses two powerful machine learning models to rank features.

### Random Forest Feature Importance

**What is Random Forest?** Imagine you ask 100 experts to each independently make a prediction, then take the majority vote. That is essentially what Random Forest does — it creates many decision trees and averages their predictions. Each tree is trained on a random subset of the data and features.

**How is importance calculated?** For each feature, the app measures how much the model's predictions would worsen if that feature were removed or shuffled. Features that cause the biggest drop in accuracy are the most important.

The app shows a horizontal bar chart ranking all features by their importance score, along with a list of the top 10 features with their exact scores.

### XGBoost Feature Importance (Classification Only)

**What is XGBoost?** XGBoost (eXtreme Gradient Boosting) builds decision trees one at a time, where each new tree specifically tries to correct the mistakes made by the previous trees. It is one of the most powerful and popular machine learning algorithms.

The importance scores from XGBoost may differ from Random Forest because it uses a different method. If both models agree that a feature is important, it is very likely to be a genuinely important feature.

### Cross-Validation Scores (5-Fold)

**What is cross-validation?** Instead of splitting data once into training and test sets, 5-fold cross-validation splits the data into 5 parts (folds). It then trains on 4 parts and tests on the 5th, repeating this 5 times so each part gets to be the test set once. This gives a more reliable estimate of model performance.

The app runs 5-fold cross-validation on all available models and shows:
- **Mean Score** — Average accuracy (classification) or R-squared (regression) across all 5 folds
- **Std** — How much the score varies across folds (lower = more stable)
- **Min** — Worst fold score
- **Max** — Best fold score

Models are sorted by mean score, and the best model is highlighted.

---

## 10. Tab 6 — Model Training and Evaluation

This is the core tab where you train machine learning models and evaluate their performance.

### Configuration

- **Test Size** — What percentage of your data to hold out for testing (default 20%). If you have 1000 rows and test_size = 0.2, the model trains on 800 rows and tests on 200 rows.
- **Random State** — A seed number (default 42) that controls the random train/test split. Using the same random state gives you reproducible results. Change it to get a different random split.
- **Target Type** — Shows whether Classification or Regression models will be used.

### Model Hyperparameters

Each model has expandable sections where you can tune its settings before training. Here is what each parameter means:

#### Logistic Regression (Classification)

**What is it in plain English?** Logistic Regression predicts the probability that something belongs to a particular class. Despite its name, it is a classification algorithm, not a regression algorithm. It draws a decision boundary (a line or surface) to separate classes.

**When to use it:**
- When you need a fast, interpretable baseline model
- When you want to understand the relationship between features and the probability of each class
- When your data is roughly linearly separable

**When NOT to use it:**
- When relationships between features and target are complex and non-linear
- When you need the highest possible accuracy (more complex models usually do better)

**Key parameters:**
- **C (regularization inverse)** — Controls how complex the model is allowed to be. Small C = simple model (less risk of overfitting). Large C = complex model (more risk of overfitting, but fits training data better). Default: 1.0.
- **Max Iterations** — How many times the algorithm iterates to find the best solution. Default: 1000.
- **Penalty** — Type of regularization: L2 (default, penalizes large coefficients), L1 (can zero out features, good for feature selection), Elasticnet (combination of L1 and L2), None (no penalty).
- **Solver** — The optimization algorithm used. lbfgs is the default and works well for most cases.

#### Random Forest (Classification and Regression)

**What is it in plain English?** Random Forest builds many decision trees, each trained on a random subset of the data and features. Each tree votes on the final prediction. Like asking 100 different experts and going with the majority opinion.

**When to use it:**
- Almost always a good choice. It works well on most types of data.
- When you want good accuracy with minimal tuning.
- When you need to know which features are important.
- When your data has complex, non-linear relationships.

**When NOT to use it:**
- When you need a model you can easily explain to non-technical stakeholders (Random Forest is a "black box").
- When you have very little training data (it may overfit).
- When speed of prediction is critical (Random Forest can be slow with many trees).

**Key parameters:**
- **n_estimators (trees)** — Number of trees in the forest. More trees = better performance but slower training. Default: 100.
- **max_depth** — How deep each tree can grow. 0 means no limit (trees grow until all leaves are pure). Smaller values prevent overfitting.
- **min_samples_split** — Minimum number of samples needed to split a node. Higher values prevent overfitting. Default: 2.
- **min_samples_leaf** — Minimum samples in a leaf node. Higher values prevent overfitting. Default: 1.
- **max_features** — How many features each tree considers for splitting. "sqrt" = square root of total features. "log2" = log base 2 of total features. None = all features.

#### SVM / SVR (Classification and Regression)

**What is it in plain English?** Support Vector Machine finds the best possible boundary (hyperplane) that separates different classes with the maximum margin (the widest gap between classes). For regression (SVR), it finds a function that stays within a specified margin of the data points.

**When to use it:**
- When you have high-dimensional data (many features relative to samples).
- When the boundary between classes is complex.
- When you have a clear margin of separation.

**When NOT to use it:**
- When your dataset is very large (SVM is slow on large datasets).
- When your data has a lot of noise or overlapping classes.
- When you need probability estimates (SVM does not naturally provide probabilities, though the app enables probability estimates for ROC curves).

**Key parameters:**
- **C (regularization)** — Like Logistic Regression's C. Small C = wide margin, more misclassifications allowed. Large C = narrow margin, fewer misclassifications on training data (risk of overfitting).
- **Kernel** — The mathematical function used to find the boundary:
  - **rbf** (Radial Basis Function) — The default. Can handle complex, non-linear boundaries.
  - **linear** — Finds a straight-line boundary. Fastest.
  - **poly** — Polynomial kernel. Can find polynomial-shaped boundaries.
  - **sigmoid** — Similar to a neural network activation function.
- **Gamma** — Controls how far the influence of a single training example reaches. "scale" and "auto" are automatic settings. Higher gamma = more complex boundary.
- **Epsilon (SVR only)** — Defines the width of the tube within which predictions are considered acceptable. Points outside the tube contribute to the loss function.

#### XGBoost (Classification and Regression)

**What is it in plain English?** XGBoost is an optimized gradient boosting algorithm that builds trees sequentially. Each new tree focuses on correcting the errors of the previous trees. It is one of the most powerful and widely-used algorithms in data science competitions.

**When to use it:**
- When you want the best possible predictive performance.
- When you have tabular (spreadsheet-like) data.
- When you can spend time tuning hyperparameters.

**When NOT to use it:**
- When you need a simple, interpretable model.
- When speed of training is critical (it is slower than Random Forest).
- When you have very small datasets (it can overfit).

**Key parameters:**
- **n_estimators (rounds)** — Number of boosting rounds (trees). Default: 100.
- **max_depth** — Maximum depth of each tree. Default: 6. Deeper trees can model more complex patterns but overfit more easily.
- **learning_rate** — How much each tree contributes to the final prediction. Smaller values need more trees but often produce better results. Default: 0.3.
- **subsample** — Fraction of training data used for each tree. Values < 1.0 add randomness and reduce overfitting. Default: 1.0 (all data).
- **colsample_bytree** — Fraction of features used for each tree. Similar to subsample but for features. Default: 1.0.

#### Gradient Boosting (Classification and Regression)

**What is it in plain English?** Very similar to XGBoost. It builds trees sequentially, where each tree corrects the errors of the previous one. The main difference is that Gradient Boosting is from scikit-learn while XGBoost is from the XGBoost library with additional optimizations.

**When to use it:**
- When you want strong predictive performance.
- As an alternative to XGBoost (sometimes one works better than the other on specific datasets).

**When NOT to use it:**
- Same limitations as XGBoost.

**Key parameters:**
- **n_estimators (trees)** — Number of boosting stages. Default: 100.
- **max_depth** — Maximum depth of each tree. Default: 3 (shallower than XGBoost's default).
- **learning_rate** — Contribution of each tree. Default: 0.1.
- **subsample** — Fraction of data used for each tree. Default: 1.0.

#### KNN (Classification and Regression)

**What is it in plain English?** K-Nearest Neighbors is the simplest machine learning algorithm. To predict a new data point, it looks at the K closest training examples and takes a vote (classification) or average (regression). "You are what your neighbors are."

**When to use it:**
- When you want a simple, intuitive model.
- When the decision boundary is irregular.
- As a quick baseline to compare against more complex models.

**When NOT to use it:**
- When your dataset is very large (it must store and search through all training data, making predictions slow).
- When you have many features (distance becomes less meaningful in high dimensions — the "curse of dimensionality").
- When features are on very different scales (must scale features first).

**Key parameters:**
- **n_neighbors (K)** — How many neighbors to look at. Small K = complex boundary (risk of overfitting). Large K = smooth boundary (risk of underfitting). Default: 5.
- **weights** — "uniform" = all neighbors have equal vote. "distance" = closer neighbors have more influence.
- **metric** — How to measure distance between points:
  - **euclidean** — Straight-line distance (as the crow flies). The most common choice.
  - **manhattan** — City-block distance (only horizontal and vertical movement).
  - **minkowski** — Generalization that includes both euclidean and manhattan.

### Select Models to Train

Before training, you can select which models to train:
- Check "Select All" to train all available models
- Uncheck it and use the multiselect to choose specific models

This saves time if you only care about certain models.

### Training Process

Click "Train Models" to start. The app will:

1. **Apply encoding** — Uses the encoding settings from the Feature Engineering tab (or auto-applies them if you did not visit that tab)
2. **Drop rows with NaN** — Removes any rows with missing values, with a warning
3. **Train/test split** — Divides data into training and testing sets
4. **Handle class imbalance** (if selected) — Applies SMOTE, oversampling, or undersampling to the training data only
5. **Scale features** — Fits the scaler on training data and transforms both training and test data
6. **Train each selected model**
7. **Evaluate each model** on the test set

### Classification Metrics

#### Accuracy

**What it measures:** The percentage of predictions that were correct.

**How to interpret:** Accuracy = 0.85 means 85% of predictions were correct.

**When accuracy is misleading:** If you have imbalanced classes (95% "No" and 5% "Yes"), a model that always predicts "No" gets 95% accuracy but is useless. Use other metrics alongside accuracy.

#### Confusion Matrix

**What it is:** A table that shows where the model got things right and where it got things wrong.

For a binary classification problem:
- **True Positives (TP)** — Correctly predicted "Yes" (actual = "Yes", predicted = "Yes")
- **True Negatives (TN)** — Correctly predicted "No" (actual = "No", predicted = "No")
- **False Positives (FP)** — Incorrectly predicted "Yes" (actual = "No", predicted = "Yes") — Type I Error
- **False Negatives (FN)** — Incorrectly predicted "No" (actual = "Yes", predicted = "No") — Type II Error

#### Classification Report

Shows for each class:
- **Precision** — Of all the times the model predicted this class, what percentage were actually correct? (TP / (TP + FP))
- **Recall (Sensitivity)** — Of all the actual instances of this class, what percentage did the model correctly identify? (TP / (TP + FN))
- **F1-Score** — The harmonic mean of Precision and Recall. Balances both concerns. Range: 0 to 1 (higher is better).
- **Support** — How many actual instances of this class exist in the test set.

#### ROC Curves and AUC

**What is a ROC Curve?** A ROC (Receiver Operating Characteristic) curve plots the True Positive Rate against the False Positive Rate at different classification thresholds.

- **True Positive Rate (Recall)** = TP / (TP + FN) — Y-axis
- **False Positive Rate** = FP / (FP + TN) — X-axis

The curve starts at (0,0) and ends at (1,1). A perfect classifier goes straight up to (0,1) and then across to (1,1).

**What is AUC?** AUC (Area Under the Curve) summarizes the ROC curve into a single number:
- **AUC = 1.0** — Perfect classifier
- **AUC = 0.5** — Random guessing (no better than a coin flip)
- **AUC < 0.5** — Worse than random (the model is systematically wrong — flip its predictions!)
- **AUC > 0.8** — Generally considered good

**For multiclass problems:** The app creates ROC curves for each class using One-vs-Rest strategy, plus a macro-average AUC.

### Regression Metrics

#### R-squared (R2)

**What it measures:** The proportion of variance in the target that is explained by the model.

**How to interpret:**
- **R2 = 1.0** — Perfect prediction. The model explains all the variance.
- **R2 = 0.7** — The model explains 70% of the variance. Generally good.
- **R2 = 0.0** — The model explains none of the variance. It is no better than just predicting the mean.
- **R2 < 0** — The model is worse than just predicting the mean.

**Important:** R2 always increases when you add more features, even if they are irrelevant. Use Adjusted R2 to account for this.

#### Adjusted R-squared

Same as R2 but penalized for the number of features. It only increases if adding a feature genuinely improves the model. Always prefer Adjusted R2 over R2 when comparing models with different numbers of features.

#### RMSE (Root Mean Squared Error)

**What it measures:** The average magnitude of prediction errors, in the same units as the target variable. It penalizes large errors more heavily (because it squares errors before averaging).

**How to interpret:** If your target is house prices in dollars and RMSE = 50,000, your model's predictions are off by about $50,000 on average (with extra penalty for large errors).

**Lower is better.**

#### MAE (Mean Absolute Error)

**What it measures:** The average absolute difference between predictions and actual values. Like RMSE but does not penalize large errors as heavily.

**How to interpret:** If MAE = 30,000 for house prices, your model's predictions are off by about $30,000 on average.

**Lower is better.** MAE is easier to interpret than RMSE because it is in the original units.

### Actual vs Predicted Plot

A scatter plot where:
- X-axis = actual values from the test set
- Y-axis = what the model predicted
- A red dashed diagonal line represents perfect predictions

Points close to the red line = good predictions. Points far from the line = poor predictions.

### Residual Plot

A scatter plot showing:
- X-axis = predicted values
- Y-axis = residuals (actual - predicted)

**What to look for:**
- Points scattered randomly around zero = good (model captures the pattern)
- Points forming a curved pattern = the model is missing a non-linear relationship
- Points spreading out in a funnel shape = heteroscedasticity (variance changes with predicted value)
- Points consistently above or below zero at certain ranges = systematic bias

### OLS Regression Summary (statsmodels)

For regression models, the app provides a detailed OLS (Ordinary Least Squares) summary, similar to what you would get from statistical software like R or Stata.

#### Coefficients Table

Shows for each feature:
- **Coefficient** — How much the target changes for a one-unit increase in this feature (holding all other features constant). Positive = positive relationship. Negative = negative relationship.
- **Std Error** — How uncertain the coefficient estimate is. Larger = less certain.
- **t-Statistic** — Coefficient divided by its standard error. Larger absolute values indicate more significant features.
- **P-Value** — Whether this feature's coefficient is statistically significant. P < 0.05 means the feature significantly affects the target.
- **95% Confidence Interval** — The range within which the true coefficient likely falls.

#### Model Summary Statistics
- **R-squared (train)** — How well the model fits the training data
- **Adjusted R-squared (train)** — R-squared adjusted for the number of features
- **F-Statistic** — Tests whether the overall model is significant (are any features useful?)
- **AIC (Akaike Information Criterion)** — Measures model quality, balancing fit and complexity. Lower is better.
- **BIC (Bayesian Information Criterion)** — Similar to AIC but more strongly penalizes complexity. Lower is better.
- **Log-Likelihood** — How well the model fits the data. Higher is better.
- **Durbin-Watson** — Tests for autocorrelation in residuals (see Tab 2 section for interpretation).

### Regression Diagnostics

#### Q-Q Plot (Quantile-Quantile Plot)

**What it shows:** Whether the residuals (prediction errors) follow a normal distribution.

**How to read it:** If the points fall along the diagonal line, the residuals are normally distributed. Deviations from the line indicate non-normality:
- Points curving up at the right end = heavy right tail (positive skew)
- Points curving down at the left end = heavy left tail (negative skew)
- S-shaped pattern = light or heavy tails

The app also runs a Shapiro-Wilk test on the residuals and tells you whether they are normally distributed.

#### Scale-Location Plot

**What it shows:** Whether the spread (variance) of residuals is constant across predicted values.

**How to read it:** The red LOWESS line should be roughly horizontal. If it slopes upward or downward, the variance is not constant (heteroscedasticity). The spread of points should be roughly even across the plot.

#### Cook's Distance

**What it shows:** Which individual data points have the most influence on the model's coefficients. Points with high Cook's Distance are "influential" — removing them would significantly change the model.

**How to interpret:**
- Points with Cook's D > 4/n (where n is the number of observations) are considered influential
- The app highlights these points in red on the chart and lists them in a table

**What to do:** If you have many influential points, investigate them. They might be data entry errors, or they might represent genuinely extreme cases that deserve separate analysis.

#### Leverage vs Residuals Plot

**What it shows:** Each point's leverage (how unusual its feature values are) against its standardized residual (how poorly it was predicted).

**How to read it:**
- Points in the upper-right or lower-right corners have both high leverage AND high residuals — these are the most problematic points
- Points between the horizontal lines at +2 and -2 are normal
- Points beyond +2 or -2 have unusually large residuals

#### VIF (Variance Inflation Factor)

Same as in Tab 0, but computed on the training data used for the regression. Features with VIF > 10 indicate problematic multicollinearity that may make coefficient estimates unreliable.

---

## 11. Tab 7 — Predict

This tab lets you use your trained models to make predictions on new data.

### Prerequisites

You must train at least one model in Tab 6 before you can make predictions. The tab stores all the necessary information: the trained model, the encoding pipeline, the scaler, and the feature names.

### Select Model

Choose which trained model to use for prediction from a dropdown.

### Single Prediction (Manual Input)

The app shows input fields for each original feature using the original column names and values from your CSV. You do NOT need to worry about encoding — it is applied automatically behind the scenes.

**How input fields work:**
- **Label-encoded categorical columns** — Show a dropdown with the original category names (e.g., "Red", "Blue", "Green")
- **One-Hot encoded categorical columns** — Show a dropdown with the original category names
- **Numeric columns** — Show a number input with a default value (the median of that column in the training data)
- **Other columns** — Show a text input

After entering all values, click "Predict" to get the result.

**For classification models:**
- The predicted class is displayed prominently
- A table of class probabilities shows the likelihood of each class
- A confidence indicator warns if the prediction has low confidence (< 40%), moderate confidence (40-70%), or high confidence (> 70%)

**For regression models:**
- The predicted numeric value is displayed

### Batch Prediction (CSV Upload)

Instead of entering values one by one, you can upload a CSV file with multiple rows:

1. The CSV must have the same column names as your original training data
2. The app checks for missing columns and warns you
3. It applies the same encoding and scaling pipeline used during training
4. Predictions are added as a new column ("Prediction")
5. For classification, probability columns are also added (e.g., "P(Yes)", "P(No)")
6. You can download the results as a CSV file

---

## 12. Tab 8 — Data Profiling

This tab gives you a comprehensive overview of your data's characteristics and quality.

### Descriptive Statistics

Shows the standard `describe()` output for all columns:
- **count** — Number of non-missing values
- **mean** — Average value
- **std** — Standard deviation (how spread out values are)
- **min** — Minimum value
- **25% (Q1)** — First quartile (25th percentile)
- **50% (Median)** — Middle value
- **75% (Q3)** — Third quartile (75th percentile)
- **max** — Maximum value

For categorical columns, it shows count, number of unique values, the most frequent value (top), and how often it appears (freq).

### Per-Column Profile

A detailed table with one row per column showing:

**For all columns:**
- **Column** — The column name
- **Dtype** — Data type (float64, int64, object, etc.)
- **Non-Null** — Number of non-missing values
- **Missing** — Number of missing values
- **Missing %** — Percentage of missing values
- **Unique** — Number of distinct values

**For numeric columns (additional):**
- **Mean** — Average value
- **Std** — Standard deviation
- **Min, Q1, Median, Q3, Max** — Distribution summary

**Skewness** — Measures the asymmetry of the distribution:
- **Skewness = 0** — Perfectly symmetric (bell curve)
- **Skewness > 0** — Right-skewed (long tail to the right). Most values are low.
- **Skewness < 0** — Left-skewed (long tail to the left). Most values are high.
- **Absolute skewness > 1** — Highly skewed

**Kurtosis** — Measures the "tailedness" of the distribution:
- **Kurtosis = 0** (or 3, depending on convention) — Normal distribution
- **Kurtosis > 0** — Heavy tails (more extreme values than a normal distribution)
- **Kurtosis < 0** — Light tails (fewer extreme values than a normal distribution)

**For categorical columns (additional):**
- **Mode** — The most frequently occurring value
- **Top 5** — The 5 most frequent values and their counts

### Memory Usage

Shows how much memory your dataset is using:
- In kilobytes (KB) and megabytes (MB)
- Useful for knowing if your dataset is too large for certain operations

### Data Quality Score

A single percentage score based on how many cells in your dataset are non-missing:

```
Quality Score = 100 * (1 - missing cells / total cells)
```

- **95% or higher** — Excellent data quality
- **85-94%** — Good data quality
- **70-84%** — Fair data quality — consider handling missing values
- **Below 70%** — Poor data quality — significant missing data

### Duplicate Detection

Shows the number and percentage of completely duplicate rows (rows where every column value is the same as another row). If duplicates exist, you can click "Show duplicate rows" to see them all.

**What to do with duplicates:**
- If they are genuine duplicates (e.g., the same record entered twice), remove them
- If they are legitimate (e.g., two different customers who happen to have the same data), keep them

---

## 13. Glossary of All Terms

### A

**Accuracy** — The percentage of correct predictions out of all predictions. A classification metric. Range: 0 to 1 (0% to 100%).

**AIC (Akaike Information Criterion)** — A measure of model quality that balances how well the model fits the data against its complexity. Lower values indicate a better model. Used to compare different models on the same dataset.

**Alpha (significance level)** — The threshold for deciding if a statistical result is significant. Commonly set to 0.05 (5%). If the p-value is below alpha, the result is "statistically significant."

**ANOVA (Analysis of Variance)** — A statistical test that checks whether the means of three or more groups are different. Used when you have a categorical feature with 3+ categories and want to see if the target varies across groups.

**Autocorrelation** — When values in a dataset are correlated with their own past or future values. Common in time series data. Detected by the Durbin-Watson test.

### B

**Bagging** — Short for Bootstrap Aggregating. A technique where multiple models are trained on random subsets of the data and their predictions are averaged. Random Forest uses bagging.

**Baseline** — A simple reference model used for comparison. For example, always predicting the most common class is a baseline for classification; predicting the mean is a baseline for regression.

**Batch Prediction** — Making predictions on multiple data points at once (as opposed to single prediction). Useful for applying a model to a large dataset.

**Bias** — The error from overly simplistic assumptions in the model. High bias means the model misses relevant patterns (underfitting).

**BIC (Bayesian Information Criterion)** — Similar to AIC but penalizes model complexity more strongly. Lower values are better.

**Boosting** — A technique where models are trained sequentially, with each new model correcting the errors of the previous ones. XGBoost and Gradient Boosting use boosting.

### C

**Categorical Variable** — A variable that represents categories or groups (e.g., color, gender, city). Can be nominal (no order) or ordinal (has order).

**Chi-Square Test** — A statistical test that checks whether two categorical variables are related or independent. Used to find associations between features and classification targets.

**Classification** — A type of machine learning where the goal is to predict a category or class (e.g., spam/not spam, disease/no disease).

**Class Imbalance** — When one category in the target variable has many more examples than another (e.g., 95% "No" and 5% "Yes").

**Class Weight** — A parameter that tells the model to give different importance to different classes. "Balanced" weights give more importance to minority classes.

**Coefficient** — In regression, the amount the target changes for a one-unit increase in a feature (holding other features constant). Also called a parameter or weight.

**Cook's Distance** — A measure of how much the model would change if a particular data point were removed. High Cook's D means the point is influential.

**Correlation** — A statistical measure of how two variables move together. Ranges from -1 (perfectly opposite) to +1 (perfectly together).

**Cross-Validation** — A technique for evaluating model performance by splitting the data into multiple folds, training on some and testing on others. 5-fold cross-validation splits data into 5 parts.

**Cramer's V** — A measure of the strength of association between two categorical variables. Ranges from 0 (no association) to 1 (perfect association).

### D

**Data Leakage** — When information from the test set accidentally leaks into the training process, making model performance appear better than it really a. The app prevents this by splitting data before applying SMOTE or scaling.

**Decision Boundary** — The surface or line that separates different predicted classes. SVM tries to find the best decision boundary.

**Degrees of Freedom (DoF)** — The number of independent pieces of information used in a statistical calculation. In Chi-Square, DoF = (rows - 1) * (columns - 1).

**Durbin-Watson** — A test statistic that checks for autocorrelation in regression residuals. Values near 2 indicate no autocorrelation.

### E

**Encoding** — The process of converting categorical (text) variables into numbers that machine learning models can process. Common methods: One-Hot Encoding, Label Encoding.

**Ensemble** — A technique that combines multiple models to produce better predictions than any single model. Random Forest, XGBoost, and Gradient Boosting are ensemble methods.

**Euclidean Distance** — The straight-line distance between two points. The most common distance metric. Used by KNN by default.

### F

**F-Statistic** — A test statistic used in ANOVA and OLS regression. Tests whether the overall model (or group means) is significant. Large F-values indicate significance.

**Feature** — An input variable used for prediction. Also called a variable, attribute, predictor, or independent variable.

**Feature Engineering** — The process of creating new features or transforming existing ones to improve model performance. Includes encoding, scaling, and class imbalance handling.

**Feature Importance** — A ranking of features by how useful they are for prediction. Computed by Random Forest and XGBoost in this app.

### G

**Gamma** — A parameter in SVM that controls the influence of individual training points. Higher gamma = more complex decision boundary.

**Gradient Boosting** — A boosting technique where each new model is trained to predict the errors (gradients) of the combined previous models.

### H

**Heteroscedasticity** — When the variance of residuals is not constant across predicted values. Violates assumptions of linear regression. Detected by the Breusch-Pagan test.

**Homoscedasticity** — When the variance of residuals is constant across predicted values. The assumption of linear regression. Detected by the Breusch-Pagan test (not significant = homoscedastic).

**Hyperparameter** — A setting you choose before training a model (like number of trees, learning rate, C). Not learned from the data. Tuning hyperparameters can improve model performance.

### I

**IQR (Interquartile Range)** — The range between Q1 (25th percentile) and Q3 (75th percentile). Used to detect outliers: values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are outliers.

### K

**K-Nearest Neighbors (KNN)** — A simple algorithm that classifies a point based on the majority class of its K closest training examples.

**Kolmogorov-Smirnov Test** — A non-parametric test that compares whether two samples come from the same distribution. Used to check if a feature's distribution differs between two target classes.

**Kruskal-Wallis** — A non-parametric alternative to ANOVA. Tests whether three or more groups have the same distribution. Does not assume normality.

**Kurtosis** — A measure of the "tailedness" of a distribution. High kurtosis means more extreme values; low kurtosis means fewer.

### L

**Label Encoding** — Converts each category to a sequential integer (e.g., Red=0, Blue=1, Green=2). Simple but implies an ordering that may not exist.

**Leverage** — A measure of how far a data point's feature values are from the mean. High leverage points have unusual feature combinations.

**Linear Regression** — A regression model that assumes a straight-line relationship between features and the target. Fits: y = b0 + b1*x1 + b2*x2 + ...

**Logistic Regression** — A classification model that predicts the probability of belonging to a class using a sigmoid function. Despite its name, it is used for classification.

### M

**MAE (Mean Absolute Error)** — The average of absolute differences between predictions and actual values. Easier to interpret than RMSE. Lower is better.

**Manhattan Distance** — The sum of absolute differences between coordinates. Also called city-block or L1 distance.

**Mann-Whitney U** — A non-parametric test that compares whether two groups have the same distribution. Used when data is not normally distributed. Alternative to the T-Test.

**Mean** — The average of all values. Sum of all values divided by count.

**Median** — The middle value when data is sorted. More robust to outliers than the mean.

**Metric** — A measurement used to evaluate model performance (accuracy, R2, RMSE, etc.).

**Missing Values** — Data points that are absent or undefined, represented as NaN (Not a Number).

**Mode** — The most frequently occurring value in a dataset.

**Multicollinearity** — When two or more features are highly correlated with each other, making it difficult to determine their individual effects on the target. Detected by VIF.

**Mutual Information** — A measure of the statistical dependency between two variables. Can detect any kind of relationship (not just linear). 0 = independent, higher = more dependent.

### N

**NaN (Not a Number)** — A special value indicating missing or undefined data.

**Non-parametric** — A statistical test or model that does not assume a specific distribution (like normal distribution). More flexible but sometimes less powerful. Examples: Mann-Whitney U, Kruskal-Wallis, Spearman.

**Normal Distribution** — A bell-shaped distribution where most values cluster around the mean. Also called Gaussian distribution. Many statistical tests assume normality.

**Null Hypothesis (H0)** — The default assumption that there is no effect or relationship. Statistical tests try to reject the null hypothesis.

### O

**OLS (Ordinary Least Squares)** — The standard method for fitting a linear regression model. Finds the coefficients that minimize the sum of squared residuals.

**One-Hot Encoding** — Creates a new binary (0/1) column for each category. No ordering is implied. Good for nominal categorical variables with few categories.

**Outlier** — A data point that is unusually far from the rest of the data. Can be genuine extreme values or data errors.

**Overfitting** — When a model learns the training data too well, including its noise, and performs poorly on new data. Characterized by high training accuracy but low test accuracy.

### P

**P-Value** — The probability that the observed result occurred by random chance. Small p-values (typically < 0.05) indicate statistically significant results.

**Pearson Correlation** — A measure of the linear relationship between two numeric variables. Ranges from -1 to +1.

**PowerTransformer** — A preprocessing technique that transforms features to be more normally distributed using the Yeo-Johnson method.

### Q

**Q1 (First Quartile)** — The 25th percentile. 25% of values fall below this point.

**Q3 (Third Quartile)** — The 75th percentile. 75% of values fall below this point.

**Q-Q Plot** — A plot that compares the distribution of residuals to a normal distribution. If points fall on the diagonal line, the residuals are normal.

**QuantileTransformer** — A preprocessing technique that maps data to a normal or uniform distribution using quantiles.

### R

**Random Forest** — An ensemble of decision trees trained on random subsets of data and features. Predictions are based on majority vote (classification) or average (regression).

**Random State** — A seed value that controls randomness in the train/test split and model training. Using the same random state gives reproducible results.

**R-squared (R2)** — The proportion of variance in the target explained by the model. Range: 0 to 1 (can be negative for very bad models). Higher is better.

**Recall** — Of all actual positive cases, what fraction did the model correctly identify? Also called sensitivity or true positive rate.

**Regression** — A type of machine learning where the goal is to predict a continuous numeric value (e.g., price, temperature).

**Regularization** — A technique to prevent overfitting by adding a penalty for model complexity. Controlled by the C parameter in Logistic Regression and SVM.

**Residual** — The difference between the actual value and the predicted value: residual = actual - predicted.

**RMSE (Root Mean Squared Error)** — The square root of the average squared prediction errors. In the same units as the target. Penalizes large errors more than MAE. Lower is better.

### S

**Scaling** — Transforming features to have similar ranges or distributions. Important for algorithms that are sensitive to feature magnitudes (KNN, SVM, Logistic Regression).

**Shapiro-Wilk Test** — A statistical test for normality. Checks whether a sample comes from a normally distributed population.

**SMOTE (Synthetic Minority Oversampling Technique)** — Creates synthetic examples of the minority class by interpolating between existing minority examples using k-Nearest Neighbors.

**Standard Deviation (Std)** — A measure of how spread out values are from the mean. Higher = more spread out.

**StandardScaler** — Transforms features to have mean=0 and standard deviation=1. Also called z-score normalization.

**Statistical Significance** — A result is statistically significant if it is unlikely to have occurred by chance (p-value < alpha).

**SVM (Support Vector Machine)** — A classification algorithm that finds the optimal boundary (hyperplane) to separate classes with maximum margin.

**SVR (Support Vector Regression)** — The regression version of SVM. Finds a function that stays within a specified margin of the data points.

**Skewness** — A measure of the asymmetry of a distribution. Positive = right-skewed. Negative = left-skewed. 0 = symmetric.

### T

**Target Variable** — The variable you are trying to predict. Also called the dependent variable, response variable, or label.

**Test Set** — The portion of data held out for evaluating the model. The model never sees this data during training.

**T-Test** — A statistical test that compares the means of two groups. The app uses Welch's T-Test, which does not assume equal variances.

**Train Set** — The portion of data used to train the model.

**Train/Test Split** — Dividing data into a training set (for model training) and a test set (for evaluation). The app defaults to 80% train, 20% test.

### U

**Underfitting** — When a model is too simple to capture the underlying pattern. Characterized by low accuracy on both training and test data.

### V

**Variance** — The average of squared differences from the mean. Measures how spread out the data is. Standard deviation is the square root of variance.

**VIF (Variance Inflation Factor)** — Measures how much a feature's coefficient variance is inflated due to correlation with other features. VIF > 10 indicates problematic multicollinearity.

### W

**Welch's T-Test** — A variant of the T-Test that does not assume the two groups have equal variances. More robust than the standard T-Test.

---

## 14. Tips and Best Practices

### General Workflow

1. **Upload your data** and check the Data Profiling tab first to understand what you are working with.
2. **Clean your data** in the Data Cleaning tab — handle missing values and check for outliers.
3. **Explore correlations** in the Correlation Matrix tab to identify redundant features and drop them.
4. **Run statistical tests** to find which features are related to your target.
5. **Visualize** your data to spot patterns that numbers might miss.
6. **Configure encoding and scaling** in the Feature Engineering tab.
7. **Check feature importance** to understand which features matter most.
8. **Train models** with different hyperparameters to find the best one.
9. **Make predictions** on new data.

### Handling Missing Values

- **Do not blindly drop all rows with missing values.** You might lose valuable data. Use imputation (median/mean) instead.
- **Median is usually safer than mean** because it is not affected by outliers.
- **If more than 50% of a column is missing**, consider dropping that column entirely rather than imputing.

### Dealing with Outliers

- **Understand why outliers exist** before removing them. A billionaire in income data might be a genuine data point, not an error.
- **Use RobustScaler** if your data has many outliers and you want to use a model that requires scaling.
- **Tree-based models** (Random Forest, XGBoost, Gradient Boosting) handle outliers well without special treatment.

### Choosing the Right Model

- **Start with Logistic/Linear Regression** as a baseline — it is fast, interpretable, and often surprisingly competitive.
- **Random Forest** is a good default choice — it works well on most data with minimal tuning.
- **XGBoost/Gradient Boosting** usually give the best performance but take longer to train and are harder to interpret.
- **SVM** works well for small-to-medium datasets with many features.
- **KNN** is good for quick prototyping but slow on large datasets.

### Encoding and Scaling

- **Always use One-Hot Encoding** for categorical features with few unique values (3-15). It does not introduce false ordering.
- **Use Label Encoding** only for ordinal features (where order matters) or high-cardinality features.
- **Always scale features** for KNN, SVM, Logistic Regression, and neural networks.
- **Do NOT scale features** for tree-based models (Random Forest, XGBoost, Gradient Boosting) — they are scale-invariant.

### Class Imbalance

- **Always check your class distribution** before training. A 95/5 split will produce a misleadingly accurate model.
- **SMOTE** is usually the best choice for handling imbalance, but it can create noisy synthetic examples.
- **Class weights** are the safest option — they do not change the data.
- **Undersampling** can lose valuable information from the majority class.

### Preventing Overfitting

- **Use cross-validation** instead of a single train/test split for more reliable estimates.
- **Reduce model complexity** (lower n_estimators, lower max_depth, higher regularization).
- **Use more training data** if possible.
- **Drop irrelevant features** — they add noise without adding signal.
- **Monitor the gap** between training and test performance. A large gap means overfitting.

### Interpreting Results

- **Statistically significant does not mean practically important.** A correlation of 0.05 can be significant with enough data but might be too weak to be useful.
- **Look at multiple metrics**, not just accuracy or R-squared. Check RMSE, MAE, confusion matrices, and residual plots.
- **No single model is best for all problems.** Try multiple models and compare.
- **Feature importance rankings** can differ between Random Forest and XGBoost. Features that both models agree on are most likely important.

### Saving and Sharing Results

- **Download cleaned data** from Tab 1 for use in other tools.
- **Download encoded data** from Tab 4 for use in other ML pipelines.
- **Download predictions** from Tab 7 to share results with others.
- **Use Plotly charts** (switch in the sidebar) for interactive visualizations you can share.

### Common Pitfalls

- **Forgetting to set the target type.** Classification vs. Regression makes a huge difference. Check the auto-detection.
- **Not dropping ID columns.** Columns like "ID" or "Name" can confuse models. Drop them.
- **Using too many One-Hot columns.** A feature with 100 categories creates 100 new columns, which can overwhelm models.
- **Applying SMOTE before train/test split.** This causes data leakage. The app handles this correctly by splitting first.
- **Ignoring missing values.** Models cannot handle NaN. Always clean before training.
- **Training on the full dataset** and then evaluating on the same data. Always hold out a test set.

### When to Seek Help

- If accuracy is below 50% for binary classification, something is wrong (the model is worse than random).
- If R-squared is negative, the model is worse than just predicting the mean.
- If RMSE is larger than the range of the target variable, the model is not learning useful patterns.
- If all statistical tests show non-significant results, the features may not be predictive of the target.
