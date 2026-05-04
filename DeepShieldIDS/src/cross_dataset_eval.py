
from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .feature_harmonization import harmonize_pair
from .smote_balancing import apply_smote
from .train import train_hybrididsnet
from .evaluate import evaluate_model


def train_on_a_test_on_b(name_a, prepared_a, name_b, prepared_b, binary=True, params=None):
    Xa_train, Xb_test, common = harmonize_pair(prepared_a.X_train, prepared_b.X_test)
    Xa_val, _Xb_val, _ = harmonize_pair(prepared_a.X_val, prepared_b.X_val)
    Xa_val = Xa_val.reindex(columns=common, fill_value=0)
    Xb_test = Xb_test.reindex(columns=common, fill_value=0)
    scaler = MinMaxScaler()
    Xa_train = pd.DataFrame(scaler.fit_transform(Xa_train), columns=common)
    Xa_val = pd.DataFrame(scaler.transform(Xa_val), columns=common)
    Xb_test = pd.DataFrame(scaler.transform(Xb_test), columns=common)
    X_train_bal, y_train_bal = apply_smote(Xa_train, prepared_a.y_train)
    run_name = f"cross_{name_a}_to_{name_b}"
    model, _ = train_hybrididsnet(X_train_bal, y_train_bal, Xa_val, prepared_a.y_val, binary=binary, params=params, run_name=run_name)
    metrics = evaluate_model(model, Xb_test, prepared_b.y_test, binary=binary, run_name=run_name)
    metrics["setting"] = f"Train {name_a} -> Test {name_b}"
    metrics["common_features"] = len(common)
    return metrics
