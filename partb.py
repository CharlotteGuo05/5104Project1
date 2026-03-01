import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(script_dir, "Concrete_Data.csv")
file_path= "/Users/charlotte/Downloads/concrete+compressive+strength/Concrete_Data.xls"
# df = pd.read_csv(file_path)
df=pd.read_excel(file_path, engine='xlrd')

target_col = df.columns[-1]
feature_cols = list(df.columns[:-1])

X_all = df[feature_cols].to_numpy(dtype=float)
y_all = df[target_col].to_numpy(dtype=float)

test_start = 500
test_end   = 630
test_idx = np.arange(test_start, test_end)
train_idx = np.setdiff1d(np.arange(len(df)), test_idx)

X_train_raw = X_all[train_idx]
y_train = y_all[train_idx]
X_test_raw = X_all[test_idx]
y_test = y_all[test_idx]

print("Train size:", X_train_raw.shape, "Test size:", X_test_raw.shape)

# mse and R^2 helper functions
def mse(y_true, y_pred):
    err = y_pred - y_true
    return float(np.mean(err**2))

def r2_variance(y_true, y_pred):
    v = float(np.var(y_true))
    if v == 0:
        return 0.0
    return float(1.0 - mse(y_true, y_pred) / v)

# standadize
def fit_standardizer(X_train):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    return mu, sigma

def apply_standardizer(X, mu, sigma):
    return (X - mu) / sigma

# gradient descent
def gradient_descent_linear_regression(X, y, lr=0.01, iters=5000, tol=1e-10, verbose=False):
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0
    reslut = []

    prev_loss = None
    for t in range(iters):
        y_hat = X @ w + b
        e = y_hat - y
        # gradient
        grad_w = (2 / n) * (X.T @ e)
        grad_b = (2 / n) * float(np.sum(e))
        # update
        w -= lr * grad_w
        b -= lr * grad_b

        cur_loss = float(np.mean(e*e))
        reslut.append(cur_loss)

        if prev_loss is not None and abs(prev_loss - cur_loss) < tol:
            if verbose:
                print(f"Early stop at iter {t}, loss={cur_loss:.6f}")
            break
        prev_loss = cur_loss

        if verbose and (t % 1000 == 0):
            print(f"iter={t}, loss={cur_loss:.6f}")
    return w, b, reslut

def predict(X, w, b):
    return X @ w + b

# univariate model
def univariate_model(X_train, X_test, y_train, y_test, lr, iters):
    rows = []
    for j, name in enumerate(feature_cols):
        Xtr = X_train[:, [j]]
        Xte = X_test[:, [j]]
        w, b, hist = gradient_descent_linear_regression(Xtr, y_train, lr=lr, iters=iters)
        ytr_pred = predict(Xtr, w, b)
        yte_pred = predict(Xte, w, b)

        rows.append({
            "feature": name,
            "w": float(w[0]),
            "b": float(b),
            "train_mse": mse(y_train, ytr_pred),
            "train_r2": r2_variance(y_train, ytr_pred),
            "test_mse": mse(y_test, yte_pred),
            "test_r2": r2_variance(y_test, yte_pred),
            "final_loss": float(hist[-1])
        })
    return pd.DataFrame(rows).sort_values("train_r2", ascending=False)

# multivariate model
def multivariate_model(X_train, X_test, y_train, y_test, lr, iters):
    w, b, hist = gradient_descent_linear_regression(X_train, y_train, lr=lr, iters=iters)
    ytr_pred = predict(X_train, w, b)
    yte_pred = predict(X_test, w, b)

    results = {
        "w": float(w[0]),
        "b": float(b),
        "train_mse": mse(y_train, ytr_pred),
        "train_r2": r2_variance(y_train, ytr_pred),
        "test_mse": mse(y_test, yte_pred),
        "test_r2": r2_variance(y_test, yte_pred),
        "final_loss": float(hist[-1]),
        "iters_ran": len(hist)
    }
    coef_table = pd.DataFrame({"feature": feature_cols, "w": w})
    return results, coef_table, hist

# Set 1
mu, sigma = fit_standardizer(X_train_raw)
X_train_std = apply_standardizer(X_train_raw, mu, sigma)
X_test_std  = apply_standardizer(X_test_raw,  mu, sigma)

# univariate
print("\nSet1: Standardized")
uni_std = univariate_model(X_train_std, X_test_std, y_train, y_test, lr=0.05, iters=8000)
print("\nTop univariate (std) by train_r2:")
print(uni_std[["feature", "w", "b", "train_r2", "test_r2", "train_mse", "test_mse"]].head(8))
print("----------------")
# multivariate
multi_std_results, multi_std_coefs, multi_std_hist = multivariate_model(
    X_train_std, X_test_std, y_train, y_test, lr=0.05, iters=12000
)
print("\nMultivariate (std) results:")
print(multi_std_results)
print("\nMultivariate (std) coefficients:")
print(multi_std_coefs)
print("----------------")


# set 2
print("\nSet 2: Raw data")
# univariate
uni_raw = univariate_model(X_train_raw, X_test_raw, y_train, y_test, lr=1e-6, iters=200000)
print("\nTop univariate (raw) by train_r2:")
print(uni_raw[["feature", "w", "b", "train_r2", "test_r2", "train_mse", "test_mse"]].head(8))
print("----------------")

# multivariate
multi_raw_results, multi_raw_coefs, multi_raw_hist = multivariate_model(
    X_train_raw, X_test_raw, y_train, y_test, lr=5e-7, iters=400000
)
print("\nMultivariate (raw) results:")
print(multi_raw_results)
print("\nMultivariate (raw) coefficients:")
print(multi_raw_coefs)
print("----------------")

# Plot Loss Curve
plt.figure()
plt.plot(multi_std_hist)
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Gradient Descent Convergence (Standardized Multivariate Model)")
plt.grid()
plt.show()