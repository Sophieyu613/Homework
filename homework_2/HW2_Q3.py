import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)

@dataclass
class Split:
    X: np.ndarray
    y: Optional[np.ndarray] = None

@dataclass
class HW2Dataset:
    train: Split
    val: Split
    test: Split

    @classmethod
    def from_csv(cls, train_path, val_path, test_path):
        tr = np.loadtxt(train_path, delimiter=",", skiprows=1)
        va = np.loadtxt(val_path, delimiter=",", skiprows=1)
        te = np.loadtxt(test_path, delimiter=",", skiprows=1)
        return cls(Split(tr[:, :2], tr[:, 2]),
                   Split(va[:, :2], va[:, 2]),
                   Split(te[:, :2], None))

ds = HW2Dataset.from_csv("train.csv", "val.csv", "test.csv")
Xtr, ytr = ds.train.X, ds.train.y
Xva, yva = ds.val.X,   ds.val.y
Xte      = ds.test.X
mse = lambda a, b: float(np.mean((a - b) ** 2))


# 3a
fig = plt.figure(figsize=(11, 4.5))
ax = fig.add_subplot(1, 2, 1)
sc = ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap="viridis", s=18)
ax.set_title("Training data (colored by y)")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); plt.colorbar(sc, ax=ax, label="y")

ax3 = fig.add_subplot(1, 2, 2, projection="3d")
ax3.scatter(Xtr[:, 0], Xtr[:, 1], ytr, c=ytr, cmap="viridis", s=8)
ax3.set_title("Training data (3D)")
ax3.set_xlabel("x1"); ax3.set_ylabel("x2"); ax3.set_zlabel("y")
plt.tight_layout(); plt.savefig("data_viz.png", dpi=300); plt.show()


# 3b
class RandomKitchenSinks:
    def __init__(self, D=1000, sigma=1.0, ridge_lambda=1e-3, seed=0):
        assert D % 2 == 0
        self.D, self.sigma, self.ridge_lambda = D, sigma, ridge_lambda
        self.rng = np.random.default_rng(seed)

    def _features(self, X):
        if not hasattr(self, "W_"):  # lazy init on first call
            h = self.D // 2
            d = X.shape[1]
            self.W_ = self.rng.normal(0, 1.0 / self.sigma, size=(d, h))  # N(0, I/σ²)
            self.b_ = self.rng.uniform(0, 2*np.pi, size=h)
        Z = X @ self.W_ + self.b_                    # (n, h)
        phi = np.concatenate([np.sin(Z), np.cos(Z)], axis=1)  # (n, D)
        return np.hstack([phi, np.ones((X.shape[0], 1))])     # + intercept

    def fit(self, X, y):
        Phi = self._features(X)
        n, p = Phi.shape
        A = Phi.T @ Phi + self.ridge_lambda * np.eye(p)
        self.coef_ = np.linalg.solve(A, Phi.T @ y)
        return self

    def predict(self, X):
        return self._features(X) @ self.coef_


sigmas = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
rks_train, rks_val = [], []
for s in sigmas:
    m = RandomKitchenSinks(D=1000, sigma=s, ridge_lambda=1e-3).fit(Xtr, ytr)
    rks_train.append(mse(m.predict(Xtr), ytr))
    rks_val.append(mse(m.predict(Xva), yva))

for s, tr, va in zip(sigmas, rks_train, rks_val):
    print(f"sigma={s:5.2f}  train={tr:.4f}  val={va:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(sigmas, rks_train, "o-"); ax[0].set_title("RKS train MSE vs σ"); ax[0].set_xlabel("σ"); ax[0].set_ylabel("MSE")
ax[1].plot(sigmas, rks_val,   "o-"); ax[1].set_title("RKS val MSE vs σ");   ax[1].set_xlabel("σ"); ax[1].set_ylabel("MSE")
plt.tight_layout(); plt.savefig("rks_sweep.png", dpi=300); plt.show()
best_sigma = sigmas[int(np.argmin(rks_val))]
print("[RKS] Best sigma:", best_sigma)


# 3c
class MLP(nn.Module):
    def __init__(self, d_in=2, d_hidden=128, d_out=1, lr=1e-3,
                 batch_size=128, activation="relu", n_epochs=300, device="cpu"):
        super().__init__()
        self.device, self.lr, self.bs = device, lr, batch_size
        self.n_epochs = n_epochs
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), act(),
            nn.Linear(d_hidden, d_out)
        ).to(device)

    def fit(self, X, y):
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        n = X.shape[0]
        self.net.train()
        for ep in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.bs):
                idx = perm[i:i+self.bs]
                opt.zero_grad()
                loss = loss_fn(self.net(X[idx]), y[idx])
                loss.backward(); opt.step()
        return self

    def forward(self, X): return self.net(X)

    @torch.no_grad()
    def predict(self, X):
        X = torch.from_numpy(X).float().to(self.device)
        return self.net(X).cpu().numpy().ravel()

hiddens = [8, 16, 32, 64, 128, 256]
mlp_tr, mlp_va = [], []
for h in hiddens:
    torch.manual_seed(0)
    m = MLP(d_hidden=h, lr=1e-3, batch_size=128, n_epochs=300).fit(Xtr, ytr)
    mlp_tr.append(mse(m.predict(Xtr), ytr))
    mlp_va.append(mse(m.predict(Xva), yva))
    print(f"[MLP] hidden={h:4d}  train={mlp_tr[-1]:.4f}  val={mlp_va[-1]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hiddens, mlp_tr, "o-"); ax[0].set_xscale("log", base=2); ax[0].set_title("MLP train MSE vs hidden"); ax[0].set_xlabel("hidden dim")
ax[1].plot(hiddens, mlp_va, "o-"); ax[1].set_xscale("log", base=2); ax[1].set_title("MLP val MSE vs hidden");   ax[1].set_xlabel("hidden dim")
plt.tight_layout(); plt.savefig("mlp_sweep.png", dpi=300); plt.show()

best_hidden = hiddens[int(np.argmin(mlp_va))]
print("[MLP] best hidden:", best_hidden)


# 3d
rks_best = RandomKitchenSinks(D=1000, sigma=best_sigma, ridge_lambda=1e-3).fit(Xtr, ytr)
mlp_best = MLP(d_hidden=best_hidden, lr=1e-3, batch_size=128, n_epochs=500).fit(Xtr, ytr)

print(f"[FINAL] RKS val MSE: {mse(rks_best.predict(Xva), yva):.4f}")
print(f"[FINAL] MLP val MSE: {mse(mlp_best.predict(Xva), yva):.4f}")

pd.DataFrame({"y": rks_best.predict(Xte)}).to_csv("rks_test_predictions.csv", index=False)
pd.DataFrame({"y": mlp_best.predict(Xte)}).to_csv("mlp_test_predictions.csv", index=False)

def plot_surface_2d(model_fn, xlim=(-3, 3), ylim=(-3, 3), title="",
                    filename="surface.png", grid=200):
    xs = np.linspace(*xlim, grid); ys = np.linspace(*ylim, grid)
    X1, X2 = np.meshgrid(xs, ys)
    Xg = np.stack([X1.ravel(), X2.ravel()], axis=1)
    if isinstance(model_fn, nn.Module):
        with torch.no_grad():
            Z = model_fn(torch.from_numpy(Xg).float()).cpu().numpy().reshape(grid, grid)
    else:
        Z = model_fn(Xg).reshape(grid, grid)
    plt.figure()
    plt.contourf(X1, X2, Z, levels=30)
    plt.title(title); plt.xlabel("x1"); plt.ylabel("x2"); plt.colorbar(label="pred")
    plt.tight_layout(); plt.savefig(filename, dpi=300); plt.show()

plot_surface_2d(rks_best.predict, title="RKS fitted surface", filename="rks_surface.png")
plot_surface_2d(mlp_best, title="MLP fitted surface", filename="mlp_surface.png")