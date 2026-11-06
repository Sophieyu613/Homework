import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_parquet("taxi_trips.parquet")
print(df.shape, df.columns.tolist())

TARGET = "trip_duration"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32).reshape(-1, 1)

# split dataset into training/validation/test sets
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=0)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=0)
print(X_tr.shape, X_val.shape, X_te.shape)

def make_loader(X, y, batch_size=256, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_tr, y_tr, shuffle=True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_te,  y_te)

def gaussian_nll(mu, sigma, y):
    return (torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2).mean()

# baseline models
class ConstantMean(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mean, dtype=torch.float32), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(std, dtype=torch.float32), requires_grad=False)
    def forward(self, x):
        return self.mu.expand(x.shape[0], 1), self.sigma.expand(x.shape[0], 1)

class LinearHomo(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)
        self.log_sigma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.fc(x), torch.exp(self.log_sigma).expand(x.shape[0], 1)

class MLPHomo(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.log_sigma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        h = self.body(x)
        return self.mu_head(h), torch.exp(self.log_sigma).expand(x.shape[0], 1)

# our model
class HeteroMLP(nn.Module):
    def __init__(self, d, hidden=64, depth=2, dropout=0.05):
        super().__init__()
        layers, in_dim = [], d
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden, 1)   # f_mu
        self.logsig_head = nn.Linear(hidden, 1)   # f_sigma
    def forward(self, x):
        h = self.body(x)
        mu = self.mu_head(h)
        sigma = torch.exp(self.logsig_head(h))
        return mu, sigma

def train(model, train_loader, val_loader, epochs=100, lr=1e-3, wd=1e-4, patience=10):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mu, sigma = model(xb)
            loss = gaussian_nll(mu, sigma, yb)
            loss.backward()
            opt.step()
        model.eval()
        va = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, sigma = model(xb)
                va += gaussian_nll(mu, sigma, yb).item() * xb.size(0)
        va /= len(val_loader.dataset)
        if va < best_val - 1e-5:
            best_val, best_state, bad = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

d = X.shape[1]
b0 = ConstantMean(y_tr.mean(), y_tr.std()).to(device)
b1 = train(LinearHomo(d), train_loader, val_loader)
b2 = train(MLPHomo(d), train_loader, val_loader)
model = train(HeteroMLP(d), train_loader, val_loader)

# evaluation
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    mu_all, sig_all, y_all = [], [], []
    for xb, yb in loader:
        mu, sigma = model(xb.to(device))
        mu_all.append(mu.cpu()); sig_all.append(sigma.cpu()); y_all.append(yb)
    mu = torch.cat(mu_all); sig = torch.cat(sig_all); y = torch.cat(y_all)
    nll  = gaussian_nll(mu, sig, y).item()
    rmse = torch.sqrt(((mu - y) ** 2).mean()).item()
    cover = (((y - mu).abs() <= 1.96 * sig).float().mean()).item()
    return {"nll": nll, "rmse": rmse, "coverage_95": cover}

for name, m in [("Constant", b0), ("Linear (homo)", b1),
                ("MLP (homo)", b2), ("MLP (hetero, ours)", model)]:
    r = evaluate(m, test_loader)
    print(f"{name:<22} NLL={r['nll']:.4f}  RMSE={r['rmse']:.4f}  cov95={r['coverage_95']:.3f}")


# 1d
model.eval()

with torch.no_grad():
    x = torch.tensor(X_te, dtype=torch.float32).to(device).unsqueeze(0)
    mu, sigma = model(x)

mu = mu.cpu().numpy().ravel()
sigma = sigma.cpu().numpy().ravel()

z = (45 - mu) / sigma
p = norm.cdf(z)
print(f"P(trip < 45 min) = {p[0]:.4f}")


# 1e
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

b = (y_te.ravel() < 45).astype(int)

# proper scoring rules
brier = brier_score_loss(b, p)
ll = log_loss(b, np.clip(p, 1e-7, 1 - 1e-7))

# classification accuracy at threshold 0.5
b_hat = (p >= 0.5).astype(int)
acc = accuracy_score(b, b_hat)

print(f"Base rate P(Y<45) in test = {b.mean():.4f}")
print(f"Brier score: {brier:.6f}")
print(f"Log-loss: {ll:.6f}")
print(f"Classification accuracy: {acc:.4f}")