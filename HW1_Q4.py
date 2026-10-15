import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

df = pd.read_csv("mcs_ds_edited_iter_shuffled.csv")

X = df[["anchor_ratio", "trans_range", "node_density", "iterations"]].values
y = df["ale"].values

models = {
    "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge":            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "SVR_RBF":          make_pipeline(StandardScaler(), SVR(kernel="rbf", C=5.0, epsilon=0.05)),
    "DecisionTree":     DecisionTreeRegressor(max_depth=5, random_state=0),
    "RandomForest":     RandomForestRegressor(n_estimators=300, random_state=0),
    "MLP":              make_pipeline(StandardScaler(),
                                      MLPRegressor(hidden_layer_sizes=(64, 32),
                                                   max_iter=3000, random_state=0)),
}

rng = np.random.RandomState(0)
results = []

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()

for ax, (name, model) in zip(axes, models.items()):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    residuals = y_te - y_hat

    # D'Agostino–Pearson normality test
    stat, p = stats.normaltest(residuals)

    # QQ-plot
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"{name}\nD'Agostino p = {p:.4f}", fontsize=10)
    ax.get_lines()[0].set_markersize(4)

    results.append({
        "Model": name,
        "n_test": len(residuals),
        "resid_mean": residuals.mean(),
        "resid_std": residuals.std(ddof=1),
        "normaltest_stat": stat,
        "p_value": p,
        "normal_at_0.05": p > 0.05,
    })

plt.tight_layout()
plt.savefig("qqplots.png", dpi=120)
plt.show()

summary = pd.DataFrame(results).sort_values("p_value", ascending=False)
print(summary.to_string(index=False))