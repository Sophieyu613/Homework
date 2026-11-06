import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv("retail.csv")
df["CustomerID"] = df["CustomerID"].astype(str)
df["StockCode"] = df["StockCode"].astype(str)
df = df.dropna(subset=["Quantity", "StockCode", "CustomerID"])
df = df[df["Quantity"] > 0].reset_index(drop=True)

y = df["Quantity"].astype(float).values
X = df[["StockCode", "CustomerID"]].copy()

print("Quantity stats: min %.1f  median %.1f  mean %.1f  max %.1f" %
      (y.min(), np.median(y), y.mean(), y.max()))

# 2a
for c in ["StockCode", "CustomerID"]:
    print(f"{c}: {X[c].nunique()} unique values")

n_stock = X["StockCode"].nunique()
n_cust = X["CustomerID"].nunique()
print("Naive one-hot dimension:", n_stock + n_cust)

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
print(f"Train: {len(X_tr)} rows")
print(f"Validation: {len(X_val)} rows")
print(f"Test: {len(X_te)} rows")


# 2b
te = TargetEncoder(cols=["StockCode", "CustomerID"], smoothing=10)
X_tr_te = te.fit_transform(X_tr, y_tr)
X_val_te = te.transform(X_val)
X_te_te = te.transform(X_te)

# log-target for the ridge model too (heavy tail)
ridge = Ridge(alpha=1.0)
ridge.fit(X_tr_te, np.log1p(y_tr))

pred_val_te = np.expm1(ridge.predict(X_val_te))
pred_te_te = np.expm1(ridge.predict(X_te_te))
pred_te_te = np.clip(pred_te_te, 0, None)

results = {}

def report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    results[name] = (rmse, mae)
    print(f"{name:35s} RMSE={rmse:8.3f}  MAE={mae:7.3f}")

report("Target encoding + Ridge (test)", y_te, pred_te_te)


# 2c
stock_codes = sorted(X_tr["StockCode"].unique())
cust_ids = sorted(X_tr["CustomerID"].unique())
stock_map = {v: i for i, v in enumerate(stock_codes)}
cust_map = {v: i for i, v in enumerate(cust_ids)}

def encode_int(frame, mapping, col, default=0):
    return frame[col].map(mapping).fillna(default).astype("int32").values

def make_inputs(frame):
    return {
        "stock": encode_int(frame, stock_map, "StockCode"),
        "cust":  encode_int(frame,  cust_map,  "CustomerID"),
    }

emb_dim_stock = min(50, (len(stock_codes) + 1) // 2)
emb_dim_cust  = min(50, (len(cust_ids)  + 1) // 2)
print("Embedding dims:", emb_dim_stock, emb_dim_cust)

def build_emb_model(extra_dim=0, name="emb"):
    inp_s = layers.Input(shape=(), name="stock", dtype="int32")
    inp_c = layers.Input(shape=(), name="cust", dtype="int32")
    e_s = layers.Embedding(len(stock_codes) + 1, emb_dim_stock)(inp_s)
    e_c = layers.Embedding(len(cust_ids) + 1, emb_dim_cust)(inp_c)
    e_s = layers.Flatten()(e_s)
    e_c = layers.Flatten()(e_c)
    x = layers.Concatenate()([e_s, e_c])
    if extra_dim:
        inp_x = layers.Input(shape=(extra_dim,), name="extra")
        x = layers.Concatenate()([x, inp_x])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    inputs = [inp_s, inp_c] + ([inp_x] if extra_dim else [])
    return keras.Model(inputs, out, name=name)

emb_model = build_emb_model()
emb_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

hist_emb = emb_model.fit(
    make_inputs(X_tr), np.log1p(y_tr),
    validation_data=(make_inputs(X_val), np.log1p(y_val)),
    epochs=50, batch_size=256, callbacks=[es], verbose=0,
)

pred_emb = np.expm1(emb_model.predict(make_inputs(X_te), verbose=0)).ravel()
pred_emb = np.clip(pred_emb, 0, None)
report("Embedding NN (test)", y_te, pred_emb)


# 2d
raw_tr  = np.nan_to_num(X_tr_te.values,  nan=0.0)
raw_val = np.nan_to_num(X_val_te.values, nan=0.0)
raw_te  = np.nan_to_num(X_te_te.values,  nan=0.0)

scaler   = StandardScaler().fit(raw_tr)
extra_tr  = scaler.transform(raw_tr ).astype("float32")
extra_val = scaler.transform(raw_val).astype("float32")
extra_te  = scaler.transform(raw_te ).astype("float32")

def make_inputs_with_extra(frame, extra):
    d = make_inputs(frame)
    d["extra"] = extra.astype("float32")
    return d

combo_model = build_emb_model(extra_dim=extra_tr.shape[1], name="combo")
combo_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

hist_combo = combo_model.fit(
    make_inputs_with_extra(X_tr, extra_tr), np.log1p(y_tr),
    validation_data=(make_inputs_with_extra(X_val, extra_val), np.log1p(y_val)),
    epochs=80, batch_size=256, callbacks=[es], verbose=0,
)

print(f"best val_loss: {min(hist_combo.history['val_loss']):.4f}, "
      f"stopped at epoch {len(hist_combo.history['loss'])}")

pred_combo = np.expm1(combo_model.predict(
    make_inputs_with_extra(X_te, extra_te), verbose=0)).ravel()

upper = np.quantile(y_tr, 0.999)
pred_combo = np.clip(pred_combo, 0, upper)

report("Embedding + target enc (test)", y_te, pred_combo)

print("\nSummary (test set)")
for k, (r, m) in results.items():
    print(f"{k:35s} RMSE={r:8.3f}  MAE={m:7.3f}")