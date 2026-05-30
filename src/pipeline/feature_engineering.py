"""
src/pipeline/feature_engineering.py
-------------------------------------
Stages
------
1. Target variable      — is_delayed (delay >= threshold)
2. Temporal features    — hour, cyclical encoding, peak flags
3. Holiday features     — Malaysian public holidays + eve flag
4. Weather features     — raw cols + heavy_weather composite flag
5. Congestion features  — flights/concurrent departures per hour
6. Aggregate stats      — expanding delay rates + rolling windows (7d, 30d)
7. Aircraft lag         — prev 1-3 legs of same aircraft
8. Target encoding      — leak-free leave-one-out encoding per fold
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.utils.common import get_logger

logger = get_logger("feature_engineering")

WEATHER_COLS = [
    "temperature_2m", "precipitation", "wind_speed_10m", "wind_speed_120m",
    "wind_gusts_10m", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
    "cloud_cover_high",
]


def add_target(df: pd.DataFrame, threshold_min: int = 15) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    df["scheduled_departure_dt"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["scheduled_departure"].astype(str)
    )
    df["actual_departure_dt"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["actual_departure"].astype(str)
    )

    overnight = df["actual_departure_dt"] < df["scheduled_departure_dt"]
    df.loc[overnight, "actual_departure_dt"] += pd.Timedelta(days=1)

    df["delay_minutes"] = (
        (df["actual_departure_dt"] - df["scheduled_departure_dt"])
        .dt.total_seconds() / 60
    )
    df["is_delayed"] = (df["delay_minutes"] >= threshold_min).astype(int)

    logger.info(
        "Target: delayed=%.1f%%  on-time=%.1f%%",
        df["is_delayed"].mean() * 100,
        (1 - df["is_delayed"].mean()) * 100,
    )
    return df


def add_temporal_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    peak_am = cfg["features"]["peak_hours_morning"]
    peak_pm = cfg["features"]["peak_hours_evening"]

    df = df.copy()
    df["hour"]            = df["scheduled_departure_dt"].dt.hour
    df["day_of_week_num"] = df["scheduled_departure_dt"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week_num"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week_num"] / 7)

    df["is_peak_hour"]    = df["hour"].isin(peak_am + peak_pm).astype(int)
    df["is_morning_rush"] = df["hour"].isin(peak_am).astype(int)
    df["is_evening_rush"] = df["hour"].isin(peak_pm).astype(int)
    df["is_weekend"]      = df["day_of_week_num"].isin([5, 6]).astype(int)

    return df


def add_holiday_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    holiday_dates = pd.to_datetime(cfg["features"].get("public_holidays", []))
    holiday_set   = set(holiday_dates.normalize())
    eve_set       = set((holiday_dates - pd.Timedelta(days=1)).normalize())

    flight_date = df["scheduled_departure_dt"].dt.normalize()
    df["is_public_holiday"] = flight_date.isin(holiday_set).astype(int)
    df["is_holiday_eve"]    = flight_date.isin(eve_set).astype(int)

    logger.info(
        "Holiday flags: %d holiday flights  %d eve flights",
        df["is_public_holiday"].sum(),
        df["is_holiday_eve"].sum(),
    )
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    present = [c for c in WEATHER_COLS if c in df.columns]
    df[present] = df[present].fillna(df[present].median())

    if "wind_gusts_10m" in df.columns and "precipitation" in df.columns:
        df["heavy_weather"] = (
            (df["wind_gusts_10m"] > df["wind_gusts_10m"].quantile(0.75)) |
            (df["precipitation"]  > df["precipitation"].quantile(0.75))
        ).astype(int)

    return df


def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = ["date", "hour"]

    df["flights_per_hour"]             = df.groupby(grp)["id"].transform("count")
    df["concurrent_departures"]        = df["flights_per_hour"]
    df["unique_destinations_per_hour"] = df.groupby(grp)["destination"].transform("nunique")

    return df


def add_aggregate_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df      = df.copy().sort_values("scheduled_departure_dt").reset_index(drop=True)
    windows = cfg["features"].get("route_rolling_windows", [7, 30])

    df["airline_delay_rate"] = (
        df.groupby("airline")["is_delayed"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df["route"] = df["airline"] + "_" + df["destination"]

    df["route_delay_rate"] = (
        df.groupby("route")["is_delayed"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df["airline_hour_delay_rate"] = (
        df.groupby(["airline", "hour"])["is_delayed"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # Fix: fill sparse airline×hour combos with airline-level rate, not global median
    df["airline_hour_delay_rate"] = df["airline_hour_delay_rate"].fillna(
        df["airline_delay_rate"]
    )

    df["route_hour_delay_rate"] = (
        df.groupby(["route", "hour"])["is_delayed"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # Fix: fill sparse route×hour combos with route-level rate, not global median
    df["route_hour_delay_rate"] = df["route_hour_delay_rate"].fillna(
        df["route_delay_rate"]
    )

    df["delay_ratio_prev_3_airline"] = (
        df.groupby("airline")["is_delayed"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    for window in windows:
        col = f"route_delay_rate_{window}d"
        df[col] = (
            df.groupby("route")["is_delayed"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # ── NEW: EWM recency-weighted rates (method 2) ────────────────────────────
    # Recent history counts more than old history (span=30 ≈ last month weighted)
    df["airline_delay_rate_ewm"] = (
        df.groupby("airline")["is_delayed"]
        .transform(lambda x: x.shift(1).ewm(span=30, min_periods=5).mean())
    )
    df["route_delay_rate_ewm"] = (
        df.groupby("route")["is_delayed"]
        .transform(lambda x: x.shift(1).ewm(span=30, min_periods=5).mean())
    )
    # Fill sparse EWM values with the expanding mean fallback
    df["airline_delay_rate_ewm"] = df["airline_delay_rate_ewm"].fillna(
        df["airline_delay_rate"]
    )
    df["route_delay_rate_ewm"] = df["route_delay_rate_ewm"].fillna(
        df["route_delay_rate"]
    )

    df.drop(columns=["route"], inplace=True)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise interaction features (method 5).

    LightGBM captures interactions implicitly but these explicit terms
    give the model direct signal for the most predictive combinations.
    All interactions use leak-free aggregate features computed by
    add_aggregate_features(), so no additional leakage is introduced.
    """
    df = df.copy()

    # Busy hour at a chronically late airline is worse than either alone
    df["congestion_x_airline_rate"] = (
        df["flights_per_hour"] * df["airline_delay_rate"]
    )
    # Peak hour amplifies airline-specific delay patterns
    df["peak_x_airline_rate"] = (
        df["is_peak_hour"] * df["airline_delay_rate"]
    )
    # Bad routes get meaningfully worse in bad weather
    df["route_x_weather"] = (
        df["route_delay_rate"] * df["heavy_weather"]
    )
    # Cascading aircraft delay varies by airline's recovery speed
    df["lag_x_airline_rate"] = (
        df["prev_aircraft_delayed_1"] * df["airline_delay_rate"]
    )

    return df


def add_aircraft_lag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("scheduled_departure_dt").reset_index(drop=True)

    for lag in [1, 2, 3]:
        df[f"prev_aircraft_delayed_{lag}"] = (
            df.groupby("aircraft")["is_delayed"].shift(lag)
        )

    df["prev_aircraft_delayed"] = df["prev_aircraft_delayed_1"]
    return df


def target_encode(df: pd.DataFrame, cols: list, n_splits: int = 5) -> pd.DataFrame:
    df          = df.copy()
    kf          = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = df["is_delayed"].mean()

    for col in cols:
        encoded_col     = f"{col}_encoded"
        df[encoded_col] = global_mean

        for train_idx, val_idx in kf.split(df):
            mean_map = df.iloc[train_idx].groupby(col)["is_delayed"].mean()
            df.loc[df.index[val_idx], encoded_col] = (
                df.iloc[val_idx][col].map(mean_map).fillna(global_mean)
            )

    return df


def run(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    logger.info("Starting feature engineering  shape=%s", df_raw.shape)

    threshold   = cfg["data"]["delay_threshold_minutes"]
    encode_cols = cfg["features"]["target_encode_cols"]

    df = (
        df_raw
        .pipe(add_target, threshold_min=threshold)
        .pipe(add_temporal_features, cfg=cfg)
        .pipe(add_holiday_features, cfg=cfg)
        .pipe(add_weather_features)
        .pipe(add_congestion_features)
        .pipe(add_aggregate_features, cfg=cfg)
        .pipe(add_aircraft_lag)
        .pipe(add_interaction_features)       # must come after aggregate + aircraft lag
        .pipe(target_encode, cols=encode_cols)
    )

    lag_cols = [c for c in df.columns if any(
        c.startswith(p) for p in [
            "airline_delay_rate", "route_delay_rate", "airline_hour_delay_rate",
            "route_hour_delay_rate", "delay_ratio_prev_3_airline",
            "prev_aircraft_delayed",
            "congestion_x_airline_rate", "peak_x_airline_rate",
            "route_x_weather"
        ]
    )]
    df[lag_cols] = df[lag_cols].fillna(df[lag_cols].median())

    logger.info("Feature engineering complete  shape=%s", df.shape)
    return df
