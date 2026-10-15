import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("mcs_ds_edited_iter_shuffled.csv")
X = df[["anchor_ratio", "trans_range", "node_density", "iterations"]].values
y = df["ale"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Standardize
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# Baseline
baseline = LinearRegression()
baseline.fit(X_train_s, y_train)
baseline_mae = mean_absolute_error(y_test, baseline.predict(X_test_s))
print(f"sklearn baseline MAE: {baseline_mae:.4f}")

# Custom model
class LinearRegressionTorch(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

def mse_loss(pred, target):
    return ((target - pred) ** 2).mean()

# Train with SGD
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

Xtr = torch.tensor(X_train_s, dtype=torch.float32, device=device)
ytr = torch.tensor(y_train,   dtype=torch.float32, device=device).view(-1, 1)
Xte = torch.tensor(X_test_s,  dtype=torch.float32, device=device)
yte = torch.tensor(y_test,    dtype=torch.float32, device=device).view(-1, 1)

model = LinearRegressionTorch(n_features=Xtr.shape[1]).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    loss = mse_loss(model(Xtr), ytr)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"epoch {epoch+1:4d} | MSE loss {loss.item():.6f}")

# Evaluation (mean absolute error)
model.eval()
with torch.no_grad():
    y_hat = model(Xte).cpu().numpy().ravel()
custom_mae = np.mean(np.abs(y_test - y_hat))
print(f"Custom PyTorch MAE: {custom_mae:.4f}")
print(f"sklearn baseline MAE: {baseline_mae:.4f}")


# b
def t_loss(pred, target, nu=5):
    residual = target - pred
    distribution = torch.distributions.StudentT(
        df=nu,
        loc=torch.zeros_like(residual),
        scale=torch.ones_like(residual)
    )
    return -distribution.log_prob(residual).mean()

torch.manual_seed(0)

model_t = LinearRegressionTorch(n_features=Xtr.shape[1]).to(device)
optimizer_t = torch.optim.SGD(model_t.parameters(), lr=0.01)

# Train with t-distribution negative log-likelihood loss
n_epochs = 2000
for epoch in range(n_epochs):
    model_t.train()
    optimizer_t.zero_grad()
    loss = t_loss(model_t(Xtr), ytr, nu=5)
    loss.backward()
    optimizer_t.step()

    if (epoch + 1) % 200 == 0:
        print(f"epoch {epoch+1:4d} | t-distribution NLL loss {loss.item():.6f}")

# Evaluation
model_t.eval()
with torch.no_grad():
    y_hat_t = model_t(Xte).cpu().numpy().ravel()
t_mae = mean_absolute_error(y_test, y_hat_t)
print(f"t-loss MAE: {t_mae:.4f}")
print(f"sklearn baseline MAE: {baseline_mae:.4f}")
print(f"Part (a) MSE model MAE: {custom_mae:.4f}")
