import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product


df = pd.read_csv("winequality-white.csv", sep=";")
X = df.drop(columns=["quality"]).values
y = df["quality"].values.astype(float)

print(f"n total = {len(y)}")

X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=0.20, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=0.125, random_state=0)

n, l_val, l_test = len(y_tr), len(y_va), len(y_te)
print(f"train n={n}, validation l={l_val}, test l={l_test}")

# RandomForestRegressor
rf_grid = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", 1.0],
    "max_samples": [0.7, 0.9, 1.0],
    "bootstrap": [True],
}

# KNeighborsRegressor
knn_grid = {
    "n_neighbors": [1, 3, 5, 7, 11, 15, 21, 31, 41],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
    "leaf_size": [10, 20, 30, 40, 50],
    "algorithm": ["auto", "ball_tree", "kd_tree"],
}

def dict_product(d):
    keys = list(d.keys())
    for vals in product(*[d[k] for k in keys]):
        yield dict(zip(keys, vals))

def grid_search(model_cls, grid, X_tr, y_tr, X_va, y_va):
    best = {"val_mse": np.inf, "params": None, "model": None}
    results = []
    for params in dict_product(grid):
        if model_cls is RandomForestRegressor:
            model = model_cls(random_state=0, n_jobs=-1, **params)
        else:
            model = model_cls(n_jobs=-1, **params)

        model.fit(X_tr, y_tr)
        val_mse = mean_squared_error(y_va, model.predict(X_va))
        results.append((val_mse, params))
        if val_mse < best["val_mse"]:
            best = {"val_mse": val_mse, "params": params, "model": model}
    return best, results

# grid search
for name, cls, grid in [
    ("RandomForestRegressor", RandomForestRegressor, rf_grid),
    ("KNeighborsRegressor", KNeighborsRegressor, knn_grid),
]:
    print(f"\n{name}")
    print(f"grid size = {sum(1 for _ in dict_product(grid))} (val points = {l_val})")
    best, results = grid_search(cls, grid, X_tr, y_tr, X_va, y_va)

    model = cls(random_state=0, n_jobs=-1, **best["params"]) \
        if cls is RandomForestRegressor \
        else cls(**best["params"])
    model.fit(X_tr, y_tr)

    val_mse = mean_squared_error(y_va, model.predict(X_va))
    test_mse = mean_squared_error(y_te, model.predict(X_te))

    print("best params:", best["params"])
    print(f"validation MSE = {val_mse:.4f}")
    print(f"test MSE = {test_mse:.4f}")

    results.sort(key=lambda t: t[0])
    print("top-5 by validation MSE:")
    for mse, params in results[:5]:
        print(f"  {mse:.4f}  {params}")