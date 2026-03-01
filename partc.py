import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Concrete_Data.csv")
df = pd.read_csv(file_path)

target_col = df.columns[-1]
feature_cols = list(df.columns[:-1])

X_all = df[feature_cols].to_numpy(dtype=float)
y_all = df[target_col].to_numpy(dtype=float)

# test rows 501–630 (1-indexed) -> python slice [500:630]
test_idx = np.arange(500, 630)
train_idx = np.setdiff1d(np.arange(len(df)), test_idx)

X_train_raw = X_all[train_idx]
y_train = y_all[train_idx]
X_test_raw  = X_all[test_idx]
y_test  = y_all[test_idx]

# -----------------------------
# 1) Metrics (same definition as Part B)
# -----------------------------
def mse(y_true, y_pred):
    return float(np.mean((y_pred - y_true)**2))

def r2_variance_explained(y_true, y_pred):
    var = float(np.var(y_true))
    return float(1 - mse(y_true, y_pred)/var) if var != 0 else 0.0

# -----------------------------
# 2) Helper to fit OLS + report performance + p-values
# -----------------------------
def fit_ols_and_report(X_train, y_train, X_test, y_test, feature_cols, label):
    # add intercept column
    Xtr = sm.add_constant(X_train, has_constant="add")
    Xte = sm.add_constant(X_test,  has_constant="add")

    model = sm.OLS(y_train, Xtr).fit()

    ytr_pred = model.predict(Xtr)
    yte_pred = model.predict(Xte)

    perf = {
        "label": label,
        "train_mse": mse(y_train, ytr_pred),
        "train_r2": r2_variance_explained(y_train, ytr_pred),
        "test_mse": mse(y_test, yte_pred),
        "test_r2": r2_variance_explained(y_test, yte_pred),
    }

    # coefficient + t-stat + p-value table
    table = pd.DataFrame({
        "term": ["Intercept"] + feature_cols,
        "coef": model.params,
        "t_stat": model.tvalues,
        "p_value": model.pvalues
    })

    return model, perf, table

# -----------------------------
# 3) Q1 + Q2.1: RAW predictors (Set 2)
# -----------------------------
model_raw, perf_raw, table_raw = fit_ols_and_report(
    X_train_raw, y_train, X_test_raw, y_test, feature_cols, "RAW (Set 2)"
)

# -----------------------------
# 4) Q2.3: STANDARDIZED predictors (Set 1)
#    fit mean/std on TRAIN only
# -----------------------------
mu = X_train_raw.mean(axis=0)
sigma = X_train_raw.std(axis=0)
sigma[sigma == 0] = 1.0

X_train_std = (X_train_raw - mu) / sigma
X_test_std  = (X_test_raw  - mu) / sigma

model_std, perf_std, table_std = fit_ols_and_report(
    X_train_std, y_train, X_test_std, y_test, feature_cols, "STANDARDIZED (Set 1)"
)

# -----------------------------
# 5) Q2.5: LOG(x+1) predictors (Set 3)
# -----------------------------
X_train_log = np.log(X_train_raw + 1.0)
X_test_log  = np.log(X_test_raw  + 1.0)

model_log, perf_log, table_log = fit_ols_and_report(
    X_train_log, y_train, X_test_log, y_test, feature_cols, "LOG(x+1) (Set 3)"
)

# Print performance
print("=== Performance ===")
print(perf_raw)
print(perf_std)
print(perf_log)

# Print p-values (features only)
def feature_pvals(table):
    return table[table["term"] != "Intercept"][["term", "p_value", "t_stat", "coef"]]

print("\n=== RAW p-values ===")
print(feature_pvals(table_raw).to_string(index=False))

print("\n=== STANDARDIZED p-values ===")
print(feature_pvals(table_std).to_string(index=False))

print("\n=== LOG(x+1) p-values ===")
print(feature_pvals(table_log).to_string(index=False))