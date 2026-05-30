"""
src/model/predict.py
---------------------
Loads the classification model and (optionally) regression models.
Runs classification first; for flights predicted as delayed,
also predicts delay_minutes with an 80% confidence interval.

Usage (CLI)
-----------
  python -m src.model.predict --input data/new_flights.csv --output outputs/predictions.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline.feature_engineering import run as engineer_features
from src.utils.common import get_logger, load_config

logger = get_logger("predict")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(cfg: dict = None):
    """Load classification model + transformers."""
    if cfg is None:
        cfg = load_config()

    root         = Path(__file__).resolve().parents[2]
    model        = joblib.load(root / cfg["paths"]["model_file"])
    tr_path      = root / cfg["paths"]["model_file"].replace("best_model.pkl", "transformers.pkl")
    meta_objects = joblib.load(tr_path) if tr_path.exists() else None

    with open(root / cfg["paths"]["model_meta_file"]) as f:
        meta = json.load(f)

    logger.info("Classification model loaded: %s  (AUC=%.4f)",
                meta["model_name"], meta["test_roc_auc"])
    return model, meta, meta_objects


def load_regression_models(cfg: dict = None):
    """
    Load regression models if they exist.
    Returns (point, lower, upper, meta) or None if not trained yet.
    """
    if cfg is None:
        cfg = load_config()

    root     = Path(__file__).resolve().parents[2]
    out_dir  = root / cfg["paths"]["outputs_dir"]
    req      = ["regression_model.pkl", "regression_lower.pkl",
                "regression_upper.pkl", "regression_meta.json"]

    if not all((out_dir / f).exists() for f in req):
        logger.info("Regression models not found — skipping delay-time prediction.")
        return None

    point = joblib.load(out_dir / "regression_model.pkl")
    lower = joblib.load(out_dir / "regression_lower.pkl")
    upper = joblib.load(out_dir / "regression_upper.pkl")
    with open(out_dir / "regression_meta.json") as f:
        reg_meta = json.load(f)

    logger.info("Regression model loaded: %s  (MAE=%.2f min)",
                reg_meta["model_name"], reg_meta["test_mae_min"])
    return point, lower, upper, reg_meta


# ── Feature preparation ───────────────────────────────────────────────────────

def _prepare_X(df_feat: pd.DataFrame, meta: dict, meta_objects) -> pd.DataFrame:
    """Apply transformers (if any) and return feature matrix for classification."""
    if meta_objects and meta_objects.get("scaler") is not None:
        selected = meta_objects["selected_features"]
        X_sel    = df_feat[selected].fillna(df_feat[selected].median())
        X_sc     = meta_objects["scaler"].transform(X_sel)
        X_pca    = meta_objects["pca"].transform(X_sc)
        return pd.DataFrame(X_pca, columns=meta_objects["pca_cols"],
                            index=df_feat.index)

    features = meta["features"]
    for col in features:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
    return df_feat[features].fillna(0)


# ── Core prediction functions ─────────────────────────────────────────────────

def predict_df(
    model,
    meta: dict,
    df_raw: pd.DataFrame,
    meta_objects=None,
    cfg: dict = None,
    reg_models=None,
) -> pd.DataFrame:
    """
    Run classification (and optionally regression) on raw flight data.

    Columns added to the returned DataFrame
    ----------------------------------------
    delay_probability     : float — P(delayed)
    predicted_delayed     : int   — 1 if delayed, 0 if on time
    predicted_delay_min   : float — point estimate of delay in minutes
                            (NaN for flights predicted as on-time)
    delay_ci_lower_min    : float — 80% CI lower bound (minutes)
    delay_ci_upper_min    : float — 80% CI upper bound (minutes)
    """
    if cfg is None:
        cfg = load_config()

    df_feat = engineer_features(df_raw, cfg)
    X       = _prepare_X(df_feat, meta, meta_objects)

    # ── Classification ────────────────────────────────────────────────────────
    threshold = meta.get("decision_threshold", 0.5)
    proba     = model.predict_proba(X)[:, 1]
    delayed   = (proba >= threshold).astype(int)

    df_feat["delay_probability"]  = proba
    df_feat["predicted_delayed"]  = delayed
    df_feat["predicted_delay_min"]   = np.nan
    df_feat["delay_ci_lower_min"]    = np.nan
    df_feat["delay_ci_upper_min"]    = np.nan

    n_delayed = delayed.sum()
    logger.info("Classified %d/%d flights as delayed (%.1f%%)",
                n_delayed, len(delayed), n_delayed / len(delayed) * 100)

    # ── Regression — only for predicted-delayed flights ───────────────────────
    if reg_models is None:
        reg_models = load_regression_models(cfg)

    if reg_models is not None and n_delayed > 0:
        point_model, lower_model, upper_model, reg_meta = reg_models
        reg_features = reg_meta["features"]
        delay_cap    = reg_meta.get("delay_cap_min", 180)

        delayed_mask = df_feat["predicted_delayed"] == 1
        X_delayed    = df_feat.loc[delayed_mask, reg_features].copy()
        for col in reg_features:
            if col not in X_delayed.columns:
                X_delayed[col] = 0.0
        X_delayed = X_delayed.fillna(0)

        pt = np.clip(point_model.predict(X_delayed), 15, delay_cap)
        lo = np.clip(lower_model.predict(X_delayed), 15, delay_cap)
        hi = np.clip(upper_model.predict(X_delayed), 15, delay_cap)

        # Ensure bounds are sensible
        lo = np.minimum(lo, pt)
        hi = np.maximum(hi, pt)

        df_feat.loc[delayed_mask, "predicted_delay_min"]  = pt.round(1)
        df_feat.loc[delayed_mask, "delay_ci_lower_min"]   = lo.round(1)
        df_feat.loc[delayed_mask, "delay_ci_upper_min"]   = hi.round(1)

        logger.info(
            "Regression: avg predicted delay=%.1f min  "
            "avg 80%% CI=[%.1f, %.1f] min",
            pt.mean(), lo.mean(), hi.mean(),
        )

    return df_feat


def predict_single(
    model, meta: dict, feature_row: pd.DataFrame,
    meta_objects=None, cfg: dict = None, reg_models=None,
) -> dict:
    """
    Predict a single flight (one-row DataFrame already feature-engineered).
    Returns a dict with classification + regression results.
    Used by the webapp.
    """
    if cfg is None:
        cfg = load_config()

    X         = _prepare_X(feature_row, meta, meta_objects)
    threshold = meta.get("decision_threshold", 0.5)
    proba     = float(model.predict_proba(X)[0, 1])
    delayed   = proba >= threshold

    result = {
        "delayed":          delayed,
        "delay_probability":round(proba, 4),
        "predicted_delay_min":  None,
        "delay_ci_lower_min":   None,
        "delay_ci_upper_min":   None,
    }

    if delayed and reg_models is not None:
        point_model, lower_model, upper_model, reg_meta = reg_models
        reg_features = reg_meta["features"]
        delay_cap    = reg_meta.get("delay_cap_min", 180)

        X_reg = feature_row.copy()
        for col in reg_features:
            if col not in X_reg.columns:
                X_reg[col] = 0.0
        X_reg = X_reg[reg_features].fillna(0)

        pt = float(np.clip(point_model.predict(X_reg)[0], 15, delay_cap))
        lo = float(np.clip(lower_model.predict(X_reg)[0], 15, delay_cap))
        hi = float(np.clip(upper_model.predict(X_reg)[0], 15, delay_cap))

        result["predicted_delay_min"] = round(pt, 1)
        result["delay_ci_lower_min"]  = round(min(lo, pt), 1)
        result["delay_ci_upper_min"]  = round(max(hi, pt), 1)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="KLIA delay predictor")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", default="outputs/predictions.csv")
    args = parser.parse_args()

    cfg                    = load_config()
    model, meta, meta_objs = load_model(cfg)
    reg_models             = load_regression_models(cfg)
    df_raw                 = pd.read_csv(args.input)
    df_out                 = predict_df(model, meta, df_raw, meta_objs, cfg, reg_models)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info("Predictions written -> %s", out_path)


if __name__ == "__main__":
    _cli()
