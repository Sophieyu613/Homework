import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


data = load_breast_cancer()
X, y = data.data, data.target   # X: (569, 30), y: {0,1}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Standardize
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# Baseline
baseline = LogisticRegression(max_iter=5000, random_state=0)
baseline.fit(X_train_s, y_train)
baseline_acc = baseline.score(X_test_s, y_test)
print(f"sklearn baseline test accuracy: {baseline_acc:.4f}")

# Custom model
class LogisticRegressionTorch(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)   # affine transformation; holds w and b

    def forward(self, x):
        logits = self.linear(x)
        # numerically stable sigmoid; clamp to avoid log(0)
        return torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)

def bce_loss(pred, target):
    eps = 1e-7
    pred = pred.clamp(eps, 1 - eps)   # clip to avoid log(0)
    return -(target * torch.log(pred)
             + (1 - target) * torch.log(1 - pred)).mean()

# Train with SGD
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

Xtr = torch.tensor(X_train_s, dtype=torch.float32, device=device)
ytr = torch.tensor(y_train,   dtype=torch.float32, device=device).view(-1, 1)
Xte = torch.tensor(X_test_s,  dtype=torch.float32, device=device)
yte = torch.tensor(y_test,    dtype=torch.float32, device=device).view(-1, 1)

model = LogisticRegressionTorch(n_features=Xtr.shape[1]).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(Xtr)
    loss = bce_loss(pred, ytr)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch+1:3d} | loss {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    probs = model(Xte)
    preds = (probs >= 0.5).float()

custom_acc = (preds == yte).float().mean().item()
print(f"Custom PyTorch test accuracy: {custom_acc:.4f}")
print(f"sklearn baseline test accuracy: {baseline_acc:.4f}")


# b
class ProbitRegressionTorch(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.normal = torch.distributions.Normal(0.0, 1.0)

    def forward(self, x):
        z = self.linear(x)
        return self.normal.cdf(z).clamp(1e-7, 1 - 1e-7)

torch.manual_seed(0)
probit_model = ProbitRegressionTorch(n_features=Xtr.shape[1]).to(device)
opt = torch.optim.SGD(probit_model.parameters(), lr=0.1)

for epoch in range(200):
    probit_model.train()
    opt.zero_grad()
    loss = bce_loss(probit_model(Xtr), ytr)
    loss.backward()
    opt.step()

probit_model.eval()
with torch.no_grad():
    preds = (probit_model(Xte) >= 0.5).float()
probit_acc = (preds == yte).float().mean().item()
print(f"Probit test accuracy: {probit_acc:.4f}")
