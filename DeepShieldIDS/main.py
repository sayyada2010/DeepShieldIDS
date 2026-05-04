from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
from src.config import CIC_PATH, UNSW_PATH, BINARY_MODE, DEFAULT_MODEL_PARAMS, METRICS_DIR, REPORT_DIR, RANDOM_STATE
from src.utils import ensure_dirs, set_seed, save_json
from src.data_loader import load_csv
from src.preprocessing import prepare_single_dataset
from src.smote_balancing import apply_smote
from src.train import train_hybrididsnet
from src.evaluate import evaluate_model
from src.baselines import run_baselines
from src.cross_dataset_eval import train_on_a_test_on_b
from src.explainability import permutation_feature_importance


def run_dataset(name, df, binary=True, params=None, skip_baselines=False, label_col=None):
    print(f"\n=== Preparing {name} ===")
    prepared = prepare_single_dataset(df, label_col=label_col, binary=binary)
    X_train_bal, y_train_bal = apply_smote(prepared.X_train, prepared.y_train, random_state=RANDOM_STATE)
    print(f"Training samples after SMOTE: {len(X_train_bal)}")
    model, _ = train_hybrididsnet(
        X_train_bal, y_train_bal, prepared.X_val, prepared.y_val,
        binary=binary, params=params, run_name=f"hybrididsnet_{name.lower()}"
    )
    metrics = evaluate_model(model, prepared.X_test, prepared.y_test, binary=binary, run_name=f"hybrididsnet_{name.lower()}")
    metrics["setting"] = f"Independent {name}"
    if not skip_baselines:
        print(f"Running baselines for {name}")
        run_baselines(
            X_train_bal, y_train_bal, prepared.X_test, prepared.y_test,
            METRICS_DIR / f"baseline_{name.lower()}_metrics.csv"
        )
    try:
        permutation_feature_importance(
            X_train_bal, y_train_bal, prepared.X_test, prepared.y_test,
            METRICS_DIR / f"feature_importance_{name.lower()}.csv"
        )
    except Exception as exc:
        print(f"Feature importance skipped: {exc}")
    return prepared, metrics


def main():
    parser = argparse.ArgumentParser(description="DeepShieldIDS complete implementation")
    parser.add_argument("--cic", default=str(CIC_PATH), help="Path to CIC-IDS2017 CSV")
    parser.add_argument("--unsw", default=str(UNSW_PATH), help="Path to UNSW-NB15 CSV")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick tests")
    parser.add_argument("--skip-cross", action="store_true", help="Skip cross-dataset validation")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip ML baselines")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--label-col", default=None, help="Optional label column name if inference fails")
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    ensure_dirs(METRICS_DIR, REPORT_DIR)
    params = DEFAULT_MODEL_PARAMS.copy()
    if args.epochs is not None:
        params["epochs"] = args.epochs
    binary = args.mode == "binary"

    cic_df = load_csv(args.cic, nrows=args.nrows)
    unsw_df = load_csv(args.unsw, nrows=args.nrows)

    cic_prepared, cic_metrics = run_dataset("CIC", cic_df, binary=binary, params=params, skip_baselines=args.skip_baselines, label_col=args.label_col)
    unsw_prepared, unsw_metrics = run_dataset("UNSW", unsw_df, binary=binary, params=params, skip_baselines=args.skip_baselines, label_col=args.label_col)

    rows = [cic_metrics, unsw_metrics]
    if not args.skip_cross:
        print("\n=== Cross-dataset validation ===")
        try:
            rows.append(train_on_a_test_on_b("CIC", cic_prepared, "UNSW", unsw_prepared, binary=binary, params=params))
            rows.append(train_on_a_test_on_b("UNSW", unsw_prepared, "CIC", cic_prepared, binary=binary, params=params))
        except Exception as exc:
            print(f"Cross-dataset validation skipped due to feature harmonization issue: {exc}")

    summary = pd.DataFrame(rows)
    summary.to_csv(METRICS_DIR / "deepshieldids_summary_metrics.csv", index=False)
    save_json(params, REPORT_DIR / "used_hyperparameters.json")
    print("\nSummary metrics:")
    print(summary)
    print(f"\nOutputs saved under: {METRICS_DIR.parent}")


if __name__ == "__main__":
    main()
