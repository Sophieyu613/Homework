import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# dset0
dset = np.loadtxt("dset0-1.csv", delimiter=",")
X, y = dset[:, 1:], dset[:, 0].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=13, stratify=y
)

params = {"max_depth": 3, "min_samples_split": 5, "learning_rate": 0.1, "loss": "log_loss"}

for n in [1, 10, 100, 300, 1000]:
    clf = ensemble.GradientBoostingClassifier(n_estimators=n, **params)
    clf.fit(X_train, y_train)
    tr = log_loss(y_train, clf.predict_proba(X_train))
    te = log_loss(y_test,  clf.predict_proba(X_test))
    print(f"dset0  trees={n:>4}  train_dev={tr:.3f}  test_dev={te:.3f}")

# deviance curve (fit 1000 trees once, evaluate test at each iteration)
clf = ensemble.GradientBoostingClassifier(n_estimators=1000, **params).fit(X_train, y_train)
test_score = np.array([log_loss(y_test, p) for p in clf.staged_predict_proba(X_test)])

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("dset0 Deviance")
ax.plot(np.arange(1000) + 1, clf.train_score_, "b-", label="Training Set Deviance")
ax.plot(np.arange(1000) + 1, test_score,     "r-", label="Test Set Deviance")
ax.legend(loc="upper right"); ax.set_xlabel("Boosting Iterations"); ax.set_ylabel("Deviance")
fig.tight_layout(); plt.show()


# dset1
dset = np.loadtxt("dset1-1.csv", delimiter=",")
X, y = dset[:, 1:], dset[:, 0].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=13, stratify=y
)

params = {"max_depth": 3, "min_samples_split": 5, "learning_rate": 0.1, "loss": "log_loss"}

for n in [1, 10, 100, 300, 1000]:
    clf = ensemble.GradientBoostingClassifier(n_estimators=n, **params)
    clf.fit(X_train, y_train)
    tr = log_loss(y_train, clf.predict_proba(X_train))
    te = log_loss(y_test,  clf.predict_proba(X_test))
    print(f"dset1  trees={n:>4}  train_dev={tr:.3f}  test_dev={te:.3f}")

# deviance curve
clf = ensemble.GradientBoostingClassifier(n_estimators=1000, **params).fit(X_train, y_train)
test_score = np.array([log_loss(y_test, p) for p in clf.staged_predict_proba(X_test)])

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("dset1 Deviance")
ax.plot(np.arange(1000) + 1, clf.train_score_, "b-", label="Training Set Deviance")
ax.plot(np.arange(1000) + 1, test_score,     "r-", label="Test Set Deviance")
ax.legend(loc="upper right"); ax.set_xlabel("Boosting Iterations"); ax.set_ylabel("Deviance")
fig.tight_layout(); plt.show()