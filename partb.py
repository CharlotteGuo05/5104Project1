import numpy as np
import pandas as pd
import os

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Concrete_Data.csv")
df = pd.read_csv(file_path)

# Target column, last column in this dataset
target_col = df.columns[-1]
feature_cols = list(df.columns[:-1])

X_all = df[feature_cols].to_numpy(dtype=float)   # shape (1030, 8)
y_all = df[target_col].to_numpy(dtype=float)     # shape (1030,)

# -----------------------------
# 1) Train/Test split (REQUIRED)
#    Test rows 501–630 (inclusive).
#    The project statement uses row numbers (typically 1-indexed).
#    So we use iloc[500:630] in 0-indexed Python.
# -----------------------------
test_start = 500   # row 501 in 1-index
test_end   = 630   # slice end is exclusive -> includes row index 629 (row 630)

test_idx = np.arange(test_start, test_end)
train_idx = np.setdiff1d(np.arange(len(df)), test_idx)

X_train_raw = X_all[train_idx]
y_train = y_all[train_idx]
X_test_raw = X_all[test_idx]
y_test = y_all[test_idx]

print("Train size:", X_train_raw.shape, "Test size:", X_test_raw.shape)

# -----------------------------
# 2) Metrics: MSE + R^2 (project definition)
#    R^2 = 1 - (MSE / Var(y))
# -----------------------------
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred - y_true
    return float(np.mean(err**2))

def r2_variance_explained(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # project defines variance explained using MSE / variance(observed)
    v = float(np.var(y_true))  # population variance (ddof=0)
    if v == 0:
        return 0.0
    return float(1.0 - mse(y_true, y_pred) / v)

# -----------------------------
# 3) Preprocessing
#    Set 1: Standardize (fit on TRAIN only)
#    Set 2: Raw (no preprocessing)
# -----------------------------
def fit_standardizer(X_train: np.ndarray):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    # avoid divide-by-zero (just in case)
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    return mu, sigma_safe

def apply_standardizer(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma

# (optional) normalization if you prefer it instead:
def fit_minmax(X_train: np.ndarray):
    mn = X_train.min(axis=0)
    mx = X_train.max(axis=0)
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    return mn, denom

def apply_minmax(X: np.ndarray, mn: np.ndarray, denom: np.ndarray):
    return (X - mn) / denom

# -----------------------------
# 4) Gradient Descent for Linear Regression
#    Model: y_hat = Xw + b
#    Loss: MSE
# -----------------------------
def gradient_descent_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    iters: int = 5000,
    tol: float = 1e-10,
    verbose: bool = False
):
    """
    X: (n, p) feature matrix
    y: (n,) target
    returns: w (p,), b (scalar), history (list of losses)
    """
    n, p = X.shape
    w = np.zeros(p, dtype=float)
    b = 0.0
    history = []

    prev_loss = None
    for t in range(iters):
        y_hat = X @ w + b
        e = y_hat - y

        # gradients of MSE
        grad_w = (2.0 / n) * (X.T @ e)     # shape (p,)
        grad_b = (2.0 / n) * float(np.sum(e))

        # update
        w -= lr * grad_w
        b -= lr * grad_b

        # track loss
        cur_loss = float(np.mean(e**2))
        history.append(cur_loss)

        # simple stopping criterion if loss stops improving
        if prev_loss is not None and abs(prev_loss - cur_loss) < tol:
            if verbose:
                print(f"Early stop at iter {t}, loss={cur_loss:.6f}")
            break
        prev_loss = cur_loss

        if verbose and (t % 1000 == 0):
            print(f"iter={t}, loss={cur_loss:.6f}")

    return w, b, history

def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b

# -----------------------------
# 5) Runner helpers (univariate + multivariate)
# -----------------------------
def run_univariate_models(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                          lr: float, iters: int):
    """
    Fits 8 separate 1-feature models (one per column).
    Returns a DataFrame of results.
    """
    rows = []
    for j, name in enumerate(feature_cols):
        Xtr = X_train[:, [j]]  # keep as 2D (n,1)
        Xte = X_test[:, [j]]

        w, b, hist = gradient_descent_linear_regression(Xtr, y_train, lr=lr, iters=iters)
        ytr_pred = predict(Xtr, w, b)
        yte_pred = predict(Xte, w, b)

        rows.append({
            "feature": name,
            "w": float(w[0]),
            "b": float(b),
            "train_mse": mse(y_train, ytr_pred),
            "train_r2": r2_variance_explained(y_train, ytr_pred),
            "test_mse": mse(y_test, yte_pred),
            "test_r2": r2_variance_explained(y_test, yte_pred),
            "final_loss": float(hist[-1])
        })
    return pd.DataFrame(rows).sort_values("train_r2", ascending=False)

def run_multivariate_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                           lr: float, iters: int):
    w, b, hist = gradient_descent_linear_regression(X_train, y_train, lr=lr, iters=iters)
    ytr_pred = predict(X_train, w, b)
    yte_pred = predict(X_test, w, b)

    results = {
        "w": float(w[0]),
        "b": float(b),
        "train_mse": mse(y_train, ytr_pred),
        "train_r2": r2_variance_explained(y_train, ytr_pred),
        "test_mse": mse(y_test, yte_pred),
        "test_r2": r2_variance_explained(y_test, yte_pred),
        "final_loss": float(hist[-1]),
        "iters_ran": len(hist)
    }
    coef_table = pd.DataFrame({"feature": feature_cols, "w": w})
    return results, coef_table, hist

# -----------------------------
# 6) PART B: Set 1 (Standardized) + Set 2 (Raw)
#    You may need to tune lr/iters a bit. These are good starting points.
# -----------------------------

# ---- Set 1: Standardized predictors (recommended for GD stability)
mu, sigma = fit_standardizer(X_train_raw)
X_train_std = apply_standardizer(X_train_raw, mu, sigma)
X_test_std  = apply_standardizer(X_test_raw,  mu, sigma)

print("\n=== SET 1: Standardized ===")
uni_std = run_univariate_models(X_train_std, X_test_std, y_train, y_test, lr=0.05, iters=8000)
print("\nTop univariate (std) by train_r2:")
print(uni_std[[]])
print(uni_std[["feature", "w", "b", "train_r2", "test_r2", "train_mse", "test_mse"]].head(8))

print ("----------------------------------------------------------")
multi_std_results, multi_std_coefs, multi_std_hist = run_multivariate_model(
    X_train_std, X_test_std, y_train, y_test, lr=0.05, iters=12000
)
print("\nMultivariate (std) results:")
print(multi_std_results)
print("\nMultivariate (std) coefficients:")
print(multi_std_coefs)

# Check requirement: at least 2 univariate models have positive R^2 on training
pos_count_std = int((uni_std["train_r2"] > 0).sum())
print("\nUnivariate (std) count with train_r2 > 0:", pos_count_std)

# ---- Set 2: Raw predictors (often needs smaller lr)
print("\n=== SET 2: Raw (no preprocessing) ===")
uni_raw = run_univariate_models(X_train_raw, X_test_raw, y_train, y_test, lr=1e-6, iters=200000)
print("\nTop univariate (raw) by train_r2:")
print(uni_raw[["feature", "w", "b", "train_r2", "test_r2", "train_mse", "test_mse"]].head(8))

multi_raw_results, multi_raw_coefs, multi_raw_hist = run_multivariate_model(
    X_train_raw, X_test_raw, y_train, y_test, lr=5e-7, iters=400000
)
print("\nMultivariate (raw) results:")
print(multi_raw_results)
print("\nMultivariate (raw) coefficients:")
print(multi_raw_coefs)

pos_count_raw = int((uni_raw["train_r2"] > 0).sum())
print("\nUnivariate (raw) count with train_r2 > 0:", pos_count_raw)