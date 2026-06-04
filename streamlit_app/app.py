"""
streamlit_app/app.py  —  KLIA Flight Delay Predictor
Run:  streamlit run streamlit_app/app.py

Changes from previous version
------------------------------
- Probability bar now uses three-zone colouring:
    green  prob <= 31
    yellow 31 < prob < 55
    red    prob >= 55
- Dashboard: prediction accuracy chart replaced with 7-day stacked bar
  (correct / uncertain / incorrect) sourced from PostgreSQL prediction_log table
- Dashboard: model performance metrics moved to bottom
- Prediction volume segment removed from dashboard
- User predictions persisted to PostgreSQL prediction_log table on every submit
"""

import json, os, sys
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib, numpy as np, pandas as pd, requests, streamlit as st
import matplotlib.pyplot as plt
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.common import get_engine, load_config

st.set_page_config(page_title="KLIA Delay Predictor", page_icon="✈",
                   layout="wide", initial_sidebar_state="collapsed")

_env = ROOT / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_rapi_env = os.environ.get("RAPIDAPI_KEY", "")
try:
    RAPIDAPI_KEY = _rapi_env or st.secrets.get("RAPIDAPI_KEY", "")
except Exception:
    RAPIDAPI_KEY = _rapi_env

# ── DB engine — reads st.secrets on cloud, falls back to local config ─────────
def _get_engine_cloud():
    """
    Build a SQLAlchemy engine from st.secrets["postgres"] when running on
    Streamlit Cloud, otherwise delegate to the project's get_engine(CFG).
    """
    try:
        if "postgres" in st.secrets:
            s = st.secrets["postgres"]
            url = (
                f"postgresql+psycopg2://{s['user']}:{s['password']}"
                f"@{s['host']}:{s.get('port', 5432)}/{s['database']}"
            )
            from sqlalchemy import create_engine
            return create_engine(url, pool_pre_ping=True)
    except Exception:
        pass
    return get_engine(CFG)
KLIA_LAT, KLIA_LON = 2.7456, 101.7072

# Verdict thresholds (probability as 0–100)
T_LOW  = 31   # below → Unlikely Delayed (green)
T_HIGH = 35   # above → Likely Delayed   (red)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    cfg = load_config()
    mp  = ROOT / cfg["paths"]["model_file"]
    if not mp.exists():
        return None, None, None, cfg
    model = joblib.load(mp)
    tr    = mp.parent / "transformers.pkl"
    mo    = joblib.load(tr) if tr.exists() else None
    with open(mp.parent / cfg["paths"]["model_meta_file"].split("/")[-1]) as f:
        meta = json.load(f)
    return model, meta, mo, cfg

MODEL, META, META_OBJECTS, CFG = load_model()


def _model_predict_proba(model, X: pd.DataFrame) -> float:
    import lightgbm as lgb
    if isinstance(model, lgb.Booster):
        return float(model.predict(X)[0])
    return float(model.predict_proba(X)[0, 1])


# ── PostgreSQL prediction log ─────────────────────────────────────────────────

def _ensure_log_table():
    """Create prediction_log table if it doesn't exist."""
    try:
        engine = _get_engine_cloud()
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS prediction_log (
                    id               SERIAL PRIMARY KEY,
                    ts               TIMESTAMP NOT NULL DEFAULT NOW(),
                    flight_number    TEXT,
                    airline          TEXT,
                    destination      TEXT,
                    probability      FLOAT,
                    predicted_delayed BOOLEAN,
                    actual_delayed   BOOLEAN DEFAULT NULL
                )
            """))
    except Exception:
        pass

def _log_prediction_db(flight_number: str, airline: str, destination: str,
                       probability: float, predicted_delayed: bool):
    """Write one prediction row to PostgreSQL."""
    try:
        engine = _get_engine_cloud()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO prediction_log
                    (flight_number, airline, destination, probability, predicted_delayed)
                VALUES (:fn, :al, :dest, :prob, :pred)
            """), {"fn": flight_number, "al": airline, "dest": destination,
                   "prob": probability / 100.0, "pred": predicted_delayed})
    except Exception:
        pass

def _load_log_last_7_days() -> pd.DataFrame:
    """
    Return prediction_log rows from the last 7 days.
    actual_delayed is resolved live by joining against the departures table:
      - If actual_departure recorded and > scheduled_departure + 15 min -> TRUE
      - If actual_departure recorded and on time -> FALSE
      - If actual_departure is NULL (not yet departed) -> NULL (pending)
    Falls back to the stored actual_delayed column if no matching departure row found.
    """
    try:
        engine = _get_engine_cloud()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    pl.ts::date AS day,
                    pl.probability,
                    pl.predicted_delayed,
                    COALESCE(
                        CASE
                            WHEN d.actual_departure IS NOT NULL
                            THEN d.actual_departure > d.scheduled_departure + INTERVAL '15 minutes'
                            ELSE NULL
                        END,
                        pl.actual_delayed
                    ) AS actual_delayed
                FROM prediction_log pl
                LEFT JOIN departures d
                    ON UPPER(REPLACE(d.flight_number, ' ', '')) =
                       UPPER(REPLACE(pl.flight_number, ' ', ''))
                   AND d.date = pl.ts::date
                WHERE pl.ts >= NOW() - INTERVAL '7 days'
                ORDER BY pl.ts
            """)).fetchall()
        if rows:
            return pd.DataFrame(rows, columns=["day","probability",
                                               "predicted_delayed","actual_delayed"])
    except Exception:
        pass
    return pd.DataFrame(columns=["day","probability","predicted_delayed","actual_delayed"])

# ── Dashboard data queries (no ttl — cleared by refresh button) ───────────────

@st.cache_data(show_spinner=False)
def _load_delay_by_hour() -> pd.DataFrame:
    """Delay rate % for each hour 0-23, from all departures with actual data."""
    try:
        engine = _get_engine_cloud()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT EXTRACT(HOUR FROM scheduled_departure)::int AS hour,
                       COUNT(*) AS total,
                       SUM(CASE WHEN actual_departure > scheduled_departure + INTERVAL '15 minutes'
                                THEN 1 ELSE 0 END) AS delayed
                FROM departures
                WHERE actual_departure IS NOT NULL
                GROUP BY 1 ORDER BY 1
            """)).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["hour","total","delayed"])
            df["delay_pct"] = df["delayed"] / df["total"] * 100
            return df
    except Exception as e:
        import traceback
        st.error(f"Query error in _load_delay_by_hour: {e}")
        traceback.print_exc()
    return pd.DataFrame(columns=["hour","total","delayed","delay_pct"])

@st.cache_data(show_spinner=False)
def _load_delay_by_dow() -> pd.DataFrame:
    """Delay rate % for each day of week (0=Mon … 6=Sun)."""
    try:
        engine = _get_engine_cloud()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT EXTRACT(DOW FROM date)::int AS dow,
                       COUNT(*) AS total,
                       SUM(CASE WHEN actual_departure > scheduled_departure + INTERVAL '15 minutes'
                                THEN 1 ELSE 0 END) AS delayed
                FROM departures
                WHERE actual_departure IS NOT NULL
                GROUP BY 1 ORDER BY 1
            """)).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["dow","total","delayed"])
            df["delay_pct"] = df["delayed"] / df["total"] * 100
            return df
    except Exception as e:
        import traceback
        st.error(f"Query error in _load_delay_by_dow: {e}")
        traceback.print_exc()
    return pd.DataFrame(columns=["dow","total","delayed","delay_pct"])

@st.cache_data(show_spinner=False)
def _load_top_delayed_airlines(n: int = 5) -> pd.DataFrame:
    """Top n airlines by % delayed (min 30 flights)."""
    try:
        engine = _get_engine_cloud()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT airline,
                       COUNT(*) AS total,
                       SUM(CASE WHEN actual_departure > scheduled_departure + INTERVAL '15 minutes'
                                THEN 1 ELSE 0 END) AS delayed
                FROM departures
                WHERE actual_departure IS NOT NULL AND airline IS NOT NULL
                GROUP BY airline
                HAVING COUNT(*) >= 5
                ORDER BY (SUM(CASE WHEN actual_departure > scheduled_departure + INTERVAL '15 minutes'
                                   THEN 1.0 ELSE 0.0 END) / COUNT(*)) DESC
                LIMIT :n
            """), {"n": n}).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["airline","total","delayed"])
            df["delay_pct"] = df["delayed"] / df["total"] * 100
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["airline","total","delayed","delay_pct"])

@st.cache_data(show_spinner=False)
def _load_top_delayed_routes(n: int = 5) -> pd.DataFrame:
    """Top n routes (destination) by % delayed (min 20 flights)."""
    try:
        engine = _get_engine_cloud()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT destination,
                       COUNT(*) AS total,
                       SUM(CASE WHEN actual_departure > scheduled_departure + INTERVAL '15 minutes'
                                THEN 1 ELSE 0 END) AS delayed
                FROM departures
                WHERE actual_departure IS NOT NULL AND destination IS NOT NULL
                GROUP BY destination
                HAVING COUNT(*) >= 5
                ORDER BY (SUM(CASE WHEN actual_departure > scheduled_departure + INTERVAL '15 minutes'
                                   THEN 1.0 ELSE 0.0 END) / COUNT(*)) DESC
                LIMIT :n
            """), {"n": n}).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["destination","total","delayed"])
            df["delay_pct"] = df["delayed"] / df["total"] * 100
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["destination","total","delayed","delay_pct"])

@st.cache_data(show_spinner=False)
def _load_top_delayed_airline_routes(n: int = 5) -> pd.DataFrame:
    """Top n airline + destination combos by % delayed (min 15 flights)."""
    try:
        engine = _get_engine_cloud()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT airline, destination,
                       COUNT(*) AS total,
                       SUM(CASE WHEN actual_departure > scheduled_departure + INTERVAL '15 minutes'
                                THEN 1 ELSE 0 END) AS delayed
                FROM departures
                WHERE actual_departure IS NOT NULL
                  AND airline IS NOT NULL AND destination IS NOT NULL
                GROUP BY airline, destination
                HAVING COUNT(*) >= 5
                ORDER BY (SUM(CASE WHEN actual_departure > scheduled_departure + INTERVAL '15 minutes'
                                   THEN 1.0 ELSE 0.0 END) / COUNT(*)) DESC
                LIMIT :n
            """), {"n": n}).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["airline","destination","total","delayed"])
            df["delay_pct"] = df["delayed"] / df["total"] * 100
            df["label"] = df["airline"] + "\n→ " + df["destination"]
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["airline","destination","total","delayed","delay_pct","label"])


@st.cache_data(show_spinner=False)
def _load_overall_delay_stats() -> dict:
    """Get overall stats: total flights, delayed flights, no-delay flights."""
    try:
        engine = _get_engine_cloud()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN actual_departure > scheduled_departure + INTERVAL '15 minutes'
                                THEN 1 ELSE 0 END) AS delayed
                FROM departures
                WHERE actual_departure IS NOT NULL
            """)).fetchone()
        if result:
            total, delayed = result[0], result[1] or 0
            not_delayed = total - delayed
            return {
                "total": total,
                "delayed": delayed,
                "not_delayed": not_delayed,
                "delayed_pct": (delayed / total * 100) if total > 0 else 0,
                "not_delayed_pct": (not_delayed / total * 100) if total > 0 else 0,
            }
    except Exception as e:
        import traceback
        st.error(f"Query error in _load_overall_delay_stats: {e}")
        traceback.print_exc()
    return {"total": 0, "delayed": 0, "not_delayed": 0, "delayed_pct": 0, "not_delayed_pct": 0}


def _overall_delay_donut_chart(stats: dict):
    """Donut chart showing % flights delayed vs not delayed."""
    fig, ax = dark_fig(figsize=(7, 4.5))
    
    if stats["total"] == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                color="#8b949e", fontsize=10, transform=ax.transAxes)
        ax.set_title("Overall Delay Distribution", fontsize=10, pad=10)
        fig.tight_layout()
        return fig
    
    sizes = [stats["delayed"], stats["not_delayed"]]
    labels = [f"Delayed\n{stats['delayed_pct']:.1f}%", 
              f"On-Time\n{stats['not_delayed_pct']:.1f}%"]
    colors = ["#e74c3c", "#3fb950"]
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="",
        explode=explode, startangle=90,
        textprops={"fontsize": 9, "color": "#e6edf3", "weight": "600"}
    )
    
    # Draw donut hole
    centre_circle = plt.Circle((0, 0), 0.70, fc="#0d1117", zorder=10)
    ax.add_artist(centre_circle)
    
    # Center text
    ax.text(0, 0, f"{stats['total']}\nFlights", 
            ha="center", va="center", fontsize=11, weight="600",
            color="#e6edf3", zorder=11)
    
    ax.set_title("Overall Delay Distribution", fontsize=10, pad=10)
    fig.tight_layout()
    return fig



# ── Data lookups ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def lookup_db(fn, fd):
    try:
        engine = _get_engine_cloud()
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT airline, destination,
                       COALESCE(aircraft,'Unknown') AS aircraft,
                       TO_CHAR(scheduled_departure,'HH24:MI') AS scheduled_departure
                FROM departures
                WHERE UPPER(REPLACE(flight_number,' ',''))=UPPER(REPLACE(:fn,' ',''))
                  AND date=:dt LIMIT 1
            """), {"fn": fn, "dt": fd}).fetchone()
        return dict(row._mapping) if row else None
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def lookup_aerodatabox(fn, fd):
    if not RAPIDAPI_KEY:
        return None
    try:
        resp = requests.get(
            f"https://aerodatabox.p.rapidapi.com/flights/number/{fn}/{fd}",
            headers={"X-RapidAPI-Key": RAPIDAPI_KEY,
                     "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"},
            timeout=6)
        if resp.status_code != 200:
            return None
        flights = resp.json()
        if isinstance(flights, dict):
            flights = [flights]
        for f in flights:
            dep = f.get("departure", {})
            if dep.get("airport", {}).get("iata", "") in ("KUL", "WMKK"):
                sched = dep.get("scheduledTime", {}).get("local", "")
                return {
                    "airline":             f.get("airline", {}).get("name", fn[:2]),
                    "destination":         f.get("arrival", {}).get("airport", {}).get("name", "Unknown"),
                    "aircraft":            f.get("aircraft", {}).get("model", "Unknown") or "Unknown",
                    "scheduled_departure": sched[11:16] if len(sched) >= 16 else "00:00",
                }
    except Exception:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(fd, hour):
    d = {"temperature_2m":27.0,"precipitation":0.0,"wind_speed_10m":10.0,
         "wind_speed_120m":15.0,"wind_gusts_10m":15.0,"cloud_cover":50.0,
         "cloud_cover_low":20.0,"cloud_cover_mid":20.0,"cloud_cover_high":20.0}
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude":KLIA_LAT,"longitude":KLIA_LON,
                    "hourly":",".join(d.keys()),
                    "start_date":fd,"end_date":fd,"timezone":"Asia/Kuala_Lumpur"},
            timeout=5)
        if r.status_code == 200:
            h = r.json().get("hourly", {})
            if h and "time" in h:
                idx = next((i for i,t in enumerate(h["time"])
                            if datetime.fromisoformat(t).hour==hour), 0)
                for k in d:
                    v = h.get(k, [])
                    if idx < len(v) and v[idx] is not None:
                        d[k] = float(v[idx])
    except Exception:
        pass
    return d

@st.cache_data(ttl=3600, show_spinner=False)
def get_rates(airline, destination, hour):
    r = {
        "airline_delay_rate":0.30, "route_delay_rate":0.30,
        "airline_hour_delay_rate":0.30, "route_hour_delay_rate":0.30,
        "delay_ratio_prev_3_airline":0.30,
        "route_delay_rate_7d":0.30, "route_delay_rate_30d":0.30,
        "airline_encoded":0.30, "destination_encoded":0.30, "aircraft_encoded":0.30,
        "flights_per_hour":5.0, "concurrent_departures":5.0,
        "unique_destinations_per_hour":3.0,
        "prev_aircraft_delayed":0, "prev_aircraft_delayed_1":0,
        "prev_aircraft_delayed_2":0, "prev_aircraft_delayed_3":0,
        "airline_delay_rate_ewm":0.30, "route_delay_rate_ewm":0.30,
    }
    is_del = "CASE WHEN actual_departure>scheduled_departure+INTERVAL '15 minutes' THEN 1.0 ELSE 0.0 END"
    try:
        engine = _get_engine_cloud()
        with engine.connect() as conn:
            def q(sql, p={}):
                v = conn.execute(text(sql), p).scalar()
                return float(v) if v is not None else None

            v = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a)", {"a":airline})
            if v:
                r.update(airline_delay_rate=v, airline_encoded=v,
                         delay_ratio_prev_3_airline=v, airline_delay_rate_ewm=v)
            v = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a) AND EXTRACT(HOUR FROM scheduled_departure)=:h", {"a":airline,"h":hour})
            if v: r["airline_hour_delay_rate"] = v
            v = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a) AND UPPER(destination)=UPPER(:d)", {"a":airline,"d":destination})
            if v:
                r.update(route_delay_rate=v, destination_encoded=v, route_delay_rate_ewm=v)
            v = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a) AND UPPER(destination)=UPPER(:d) AND EXTRACT(HOUR FROM scheduled_departure)=:h", {"a":airline,"d":destination,"h":hour})
            if v: r["route_hour_delay_rate"] = v
            for days, key in [(7,"route_delay_rate_7d"), (30,"route_delay_rate_30d")]:
                v = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a) AND UPPER(destination)=UPPER(:d) AND date>=CURRENT_DATE-INTERVAL '{days} days'", {"a":airline,"d":destination})
                if v: r[key] = v
            v = q(f"SELECT {is_del} FROM departures WHERE UPPER(airline)=UPPER(:a) ORDER BY date DESC,scheduled_departure DESC LIMIT 1", {"a":airline})
            if v:
                iv = int(v)
                r.update(prev_aircraft_delayed=iv, prev_aircraft_delayed_1=iv)
            v = q("SELECT COUNT(*)::float/NULLIF(COUNT(DISTINCT date),0) FROM departures WHERE EXTRACT(HOUR FROM scheduled_departure)=:h", {"h":hour})
            if v: r.update(flights_per_hour=v, concurrent_departures=v)
            v = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a) AND aircraft IS NOT NULL AND aircraft != 'Unknown'", {"a": airline})
            if v: r["aircraft_encoded"] = v
            v_recent = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a) AND date >= CURRENT_DATE - INTERVAL '30 days'", {"a": airline})
            v_alltime = r["airline_delay_rate"]
            r["airline_delay_rate_ewm"] = round(v_recent * 0.6 + v_alltime * 0.4, 4) if v_recent else v_alltime
    except Exception:
        pass
    return r


# ── Feature construction ──────────────────────────────────────────────────────

def build_features(flight, fd, weather, rates):
    sched     = str(flight["scheduled_departure"])[:5]
    dt        = datetime.strptime(f"{fd} {sched}", "%Y-%m-%d %H:%M")
    hour, dow = dt.hour, dt.weekday()
    holidays  = pd.to_datetime(CFG["features"].get("public_holidays", []))
    fday      = pd.Timestamp(fd).normalize()
    h_set     = set(holidays.normalize())
    e_set     = set((holidays - pd.Timedelta(days=1)).normalize())
    peak_am   = CFG["features"]["peak_hours_morning"]
    peak_pm   = CFG["features"]["peak_hours_evening"]
    row = {
        "hour":hour, "day_of_week_num":dow,
        "hour_sin":np.sin(2*np.pi*hour/24), "hour_cos":np.cos(2*np.pi*hour/24),
        "dow_sin":np.sin(2*np.pi*dow/7),    "dow_cos":np.cos(2*np.pi*dow/7),
        "is_peak_hour":int(hour in peak_am+peak_pm),
        "is_morning_rush":int(hour in peak_am), "is_evening_rush":int(hour in peak_pm),
        "is_weekend":int(dow>=5),
        "is_public_holiday":int(fday in h_set),
        "is_holiday_eve":int(fday in e_set),
        **weather,
        "heavy_weather":int(weather["wind_gusts_10m"]>30 or weather["precipitation"]>5),
        **rates,
        "congestion_x_airline_rate": rates["flights_per_hour"] * rates["airline_delay_rate"],
        "peak_x_airline_rate": int(hour in peak_am+peak_pm) * rates["airline_delay_rate"],
        "route_x_weather": rates["route_delay_rate"] * int(
            weather["wind_gusts_10m"]>30 or weather["precipitation"]>5),
    }
    return pd.DataFrame([row])

def apply_model_input(df: pd.DataFrame) -> pd.DataFrame:
    features = META["features"]
    if META_OBJECTS and META_OBJECTS.get("scaler") is not None:
        sel = META_OBJECTS["selected_features"]
        for c in sel:
            if c not in df.columns: df[c] = 0.0
        try:
            X = META_OBJECTS["scaler"].transform(df[sel].fillna(0))
            X = META_OBJECTS["pca"].transform(X)
            return pd.DataFrame(X, columns=META_OBJECTS["pca_cols"])
        except Exception:
            pass
    for c in features:
        if c not in df.columns: df[c] = 0.0
    return df[features].fillna(0)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(flight_number, flight_date, manual_time=None):
    flight = lookup_db(flight_number, flight_date)
    source = "KLIA Database"
    if not flight:
        flight = lookup_aerodatabox(flight_number, flight_date)
        source = "AeroDataBox API"
    if not flight and manual_time:
        flight = {"airline":flight_number[:2].upper(), "destination":"Unknown",
                  "aircraft":"Unknown", "scheduled_departure":manual_time}
        source = "Manual entry"
    if not flight:
        return None, "Flight not found. Try adding a scheduled time."

    sched   = str(flight["scheduled_departure"])[:5]
    hour    = int(sched.split(":")[0]) if ":" in sched else 8
    weather = get_weather(flight_date, hour)
    rates   = get_rates(flight["airline"], flight["destination"], hour)
    df      = build_features(flight, flight_date, weather, rates)
    X       = apply_model_input(df)
    proba   = _model_predict_proba(MODEL, X)
    threshold = META.get("decision_threshold", 0.5)

    return {
        "delayed":      proba >= threshold,
        "probability":  round(proba * 100, 1),
        "flight":       flight_number.upper(),
        "airline":      flight["airline"],
        "destination":  flight["destination"],
        "departure":    sched,
        "source":       source,
        "temp":         f"{weather['temperature_2m']:.0f}°C",
        "rain":         f"{weather['precipitation']:.1f} mm",
        "wind":         f"{weather['wind_gusts_10m']:.0f} km/h",
        "delay_rate":   f"{round(rates['airline_delay_rate']*100,1)}%",
    }, None


# ── Dashboard helpers ─────────────────────────────────────────────────────────

def dark_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.title.set_color('#e6edf3')
    for spine in ax.spines.values():
        spine.set_edgecolor('#21262d')
    ax.grid(axis='y', color='#21262d', linewidth=0.6, linestyle='--')
    return fig, ax

def metric_card(label, value, delta=None, delta_kind="good"):
    delta_html = f'<div class="metric-delta {delta_kind}">{delta}</div>' if delta else ""
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'{delta_html}</div>')

@st.cache_data
def load_dashboard_meta():
    meta_path = ROOT / "models" / "model_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f), False
        except Exception:
            pass
    return {
        "model_version": "v3", "trained_on": "2025-05-30",
        "n_training_rows": 65000, "n_features": 18, "threshold": 0.31,
        "test_roc_auc": 0.7575, "test_f1": 0.566, "test_f2": 0.898,
        "test_precision": 0.481, "test_recall": 0.689, "test_accuracy": 0.712,
        "best_params": {"n_estimators":400,"learning_rate":0.05,"num_leaves":63,
                        "min_child_samples":30,"subsample":0.8,"colsample_bytree":0.8},
        "features": [
            "hour","hour_sin","hour_cos","airline_delay_rate","airline_delay_rate_ewm",
            "airline_encoded","airline_hour_delay_rate","destination_encoded",
            "route_delay_rate","route_delay_rate_7d","route_delay_rate_30d",
            "route_hour_delay_rate","delay_ratio_prev_3_airline","aircraft_encoded",
            "flights_per_hour","congestion_x_airline_rate","wind_gusts_10m","cloud_cover_mid",
        ],
        "feature_importances": [
            0.121,0.098,0.094,0.089,0.082,0.076,0.071,0.065,
            0.058,0.054,0.049,0.043,0.038,0.032,0.028,0.024,0.019,0.017,
        ],
    }, True


def _accuracy_7day_chart():
    """
    Stacked bar chart of prediction accuracy over the last 7 days.
    Segments per day:
      Green  — correct:   actual_delayed recorded AND matches predicted_delayed
      Yellow — uncertain: actual_delayed IS NULL (outcome not yet known)
      Red    — incorrect: actual_delayed recorded AND does NOT match predicted_delayed
    Data sourced from PostgreSQL prediction_log table.
    """
    df_log = _load_log_last_7_days()
    today  = date.today()
    days   = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    labels = [d.strftime("%d %b") for d in days]

    correct_counts   = []
    uncertain_counts = []
    incorrect_counts = []

    for d in days:
        if df_log.empty:
            subset = pd.DataFrame()
        else:
            subset = df_log[df_log["day"] == d]

        if subset.empty:
            correct_counts.append(0)
            uncertain_counts.append(0)
            incorrect_counts.append(0)
            continue

        known   = subset[subset["actual_delayed"].notna()]
        unknown = subset[subset["actual_delayed"].isna()]

        correct   = (known["predicted_delayed"] == known["actual_delayed"]).sum()
        incorrect = (known["predicted_delayed"] != known["actual_delayed"]).sum()
        uncertain = len(unknown)

        correct_counts.append(int(correct))
        uncertain_counts.append(int(uncertain))
        incorrect_counts.append(int(incorrect))

    fig, ax = dark_fig(figsize=(10, 3.8))
    x = list(range(len(labels)))

    b1 = ax.bar(x, correct_counts,
                color='#3fb950', edgecolor='none', label='Correct')
    b2 = ax.bar(x, uncertain_counts, bottom=correct_counts,
                color='#d29922', edgecolor='none', label='Outcome Pending')
    b3 = ax.bar(x, incorrect_counts,
                bottom=[c + u for c, u in zip(correct_counts, uncertain_counts)],
                color='#f85149', edgecolor='none', label='Incorrect')

    # Label totals on bars
    for i, (c, u, w) in enumerate(zip(correct_counts, uncertain_counts, incorrect_counts)):
        total = c + u + w
        if total:
            ax.text(i, total + 0.15, str(total), ha='center', va='bottom',
                    color='#8b949e', fontsize=7.5)

    # Accuracy % line on twin axis (only days with known outcomes)
    ax2 = ax.twinx()
    acc_pts = [(i, c / (c + w) * 100)
               for i, (c, w) in enumerate(zip(correct_counts, incorrect_counts))
               if (c + w) > 0]
    if acc_pts:
        ax2_x, ax2_y = zip(*acc_pts)
        ax2.plot(ax2_x, ax2_y, color='#388bfd', linewidth=1.6, linestyle='-',
                 marker='o', markersize=4, label='Accuracy %', zorder=5)
        for xi, yi in zip(ax2_x, ax2_y):
            ax2.text(xi, yi + 2, f'{yi:.0f}%', ha='center', va='bottom',
                     color='#388bfd', fontsize=7)
    ax2.set_ylim(0, 130)
    ax2.set_ylabel("Accuracy %", fontsize=8, color='#388bfd')
    ax2.tick_params(axis='y', colors='#388bfd', labelsize=7)
    ax2.spines['right'].set_edgecolor('#388bfd')
    for spine in ['top', 'left', 'bottom']:
        ax2.spines[spine].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Predictions", fontsize=8)
    ax.set_title("Prediction Accuracy — Last 7 Days", fontsize=10, pad=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Combine legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, framealpha=0, labelcolor='#8b949e', loc='upper left')
    fig.tight_layout()
    return fig, correct_counts, uncertain_counts, incorrect_counts


def _prediction_volume_chart():
    """Stacked bar: unlikely/could be/likely predictions per day over last 30 days."""
    df_log = _load_log_last_7_days()
    today  = date.today()
    days   = [(today - timedelta(days=i)) for i in range(29, -1, -1)]
    labels = [d.strftime("%d %b") for d in days]

    unlikely, could_be, likely = [], [], []
    for d in days:
        subset = df_log[df_log["day"] == d] if not df_log.empty else pd.DataFrame()
        unlikely.append( int((subset["probability"] <  0.31).sum())                                                        if not subset.empty else 0)
        could_be.append( int(((subset["probability"] >= 0.31) & (subset["probability"] < 0.35)).sum())                     if not subset.empty else 0)
        likely.append(   int((subset["probability"] >= 0.35).sum())                                                        if not subset.empty else 0)

    totals = [u + c + l for u, c, l in zip(unlikely, could_be, likely)]
    window = 7
    ma = [
        sum(totals[max(0, i - window + 1):i + 1]) / len(totals[max(0, i - window + 1):i + 1])
        if totals[i] > 0 or any(totals[max(0, i - window + 1):i + 1])
        else None
        for i in range(len(totals))
    ]

    fig, ax = dark_fig(figsize=(10, 3.8))
    x = list(range(len(labels)))
    ax.bar(x, unlikely, color='#3fb950', edgecolor='none', label='Unlikely Delayed')
    ax.bar(x, could_be, bottom=unlikely, color='#d29922', edgecolor='none', label='Could Be Delayed')
    ax.bar(x, likely,   bottom=[u + c for u, c in zip(unlikely, could_be)],
           color='#f85149', edgecolor='none', label='Likely Delayed')

    for i, (u, c, l) in enumerate(zip(unlikely, could_be, likely)):
        total = u + c + l
        if total:
            ax.text(i, total + 0.1, str(total), ha='center', va='bottom',
                    color='#8b949e', fontsize=7.5)

    # 7-day moving average overlay
    ma_x = [i for i, v in enumerate(ma) if v is not None]
    ma_y = [v for v in ma if v is not None]
    if ma_x:
        ax.plot(ma_x, ma_y, color='#e6edf3', linewidth=1.4, linestyle='--',
                marker='o', markersize=2.5, label='7-day MA', zorder=5)

    ax.set_xticks(x[::5])
    ax.set_xticklabels(labels[::5], fontsize=8)
    ax.set_ylabel("Predictions", fontsize=8)
    ax.set_title("Prediction Volume — Last 30 Days", fontsize=10, pad=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8, framealpha=0, labelcolor='#8b949e', loc='upper left')
    fig.tight_layout()
    return fig


def _route_history_chart(airline, destination, rates):
    airline_30d         = rates.get("airline_delay_rate_ewm", 0.30)
    route_30d           = rates.get("route_delay_rate_30d",   0.30)
    airline_x_route_30d = round((airline_30d + route_30d) / 2, 4)

    # All three values still at the 0.30 fallback means no real DB data was found
    all_default = (
        rates.get("airline_delay_rate_ewm") is None
        and rates.get("route_delay_rate_30d") is None
    ) or (airline_30d == 0.30 and route_30d == 0.30
          and rates.get("airline_delay_rate") == 0.30)

    fig, ax = dark_fig(figsize=(8, 3.2))
    if all_default:
        ax.text(0.5, 0.5, "Insufficient data for this airline & route",
                ha="center", va="center", color="#8b949e",
                fontsize=10, transform=ax.transAxes)
        ax.set_title(f"30-Day Delay History — {airline} · {destination}", fontsize=10, pad=10)
        fig.tight_layout()
        return fig
    labels = [
        f"Route (30d)\n{airline} → {destination}",
        f"Airline (30d)\n{airline}",
        f"Airline×Route (30d)\n{airline} → {destination}",
    ]
    values = [route_30d, airline_30d, airline_x_route_30d]
    fig, ax = dark_fig(figsize=(8, 3.2))
    bars = ax.bar(labels, [v * 100 for v in values],
                  color=["#d29922","#388bfd","#bc8cff"], edgecolor='none', width=0.45)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val*100 + 1.5,
                f'{val*100:.1f}%', ha='center', va='bottom',
                color='#e6edf3', fontsize=9, fontfamily='DM Mono', fontweight='500')
    ax.set_ylim(0, 108)
    ax.set_ylabel("Delay Rate (%)", fontsize=8)
    ax.set_title(f"30-Day Delay History — {airline} · {destination}", fontsize=10, pad=10)
    ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    return fig


def _delay_by_hour_chart(df: pd.DataFrame):
    fig, ax = dark_fig(figsize=(10, 3.6))
    if df.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                color="#8b949e", fontsize=10, transform=ax.transAxes)
    else:
        colors = ["#f85149" if v >= 50 else "#d29922" if v >= 30 else "#3fb950"
                  for v in df["delay_pct"]]
        ax.bar(df["hour"], df["delay_pct"], color=colors, edgecolor="none", width=0.7)
        for _, row in df.iterrows():
            ax.text(row["hour"], row["delay_pct"] + 0.8, f'{row["delay_pct"]:.0f}%',
                    ha="center", va="bottom", color="#8b949e", fontsize=6.5)
        ax.set_xticks(range(0, 24))
        ax.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=7, rotation=45)
        ax.set_ylabel("Delay Rate (%)", fontsize=8)
    ax.set_title("Delay Rate by Hour of Day", fontsize=10, pad=10)
    fig.tight_layout()
    return fig


def _delay_by_dow_chart(df: pd.DataFrame):
    dow_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    fig, ax = dark_fig(figsize=(10, 3.6))
    if df.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                color="#8b949e", fontsize=10, transform=ax.transAxes)
    else:
        labels = [dow_names[d] for d in df["dow"]]
        colors = ["#f85149" if v >= 50 else "#d29922" if v >= 30 else "#3fb950"
                  for v in df["delay_pct"]]
        ax.bar(range(len(labels)), df["delay_pct"], color=colors, edgecolor="none", width=0.55)
        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(i, row["delay_pct"] + 0.8, f'{row["delay_pct"]:.1f}%',
                    ha="center", va="bottom", color="#8b949e", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Delay Rate (%)", fontsize=8)
    ax.set_title("Delay Rate by Day of Week", fontsize=10, pad=10)
    fig.tight_layout()
    return fig


def _top_airlines_chart(df: pd.DataFrame):
    fig, ax = dark_fig(figsize=(10, 3.6))
    if df.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                color="#8b949e", fontsize=10, transform=ax.transAxes)
    else:
        df_s = df.sort_values("delay_pct")
        colors = ["#f85149" if v >= 50 else "#d29922" if v >= 30 else "#3fb950"
                  for v in df_s["delay_pct"]]
        bars = ax.barh(df_s["airline"], df_s["delay_pct"],
                       color=colors, edgecolor="none", height=0.5)
        for bar, (_, row) in zip(bars, df_s.iterrows()):
            ax.text(row["delay_pct"] + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{row["delay_pct"]:.1f}%  ({int(row["total"])} flights)',
                    va="center", color="#8b949e", fontsize=8)
        ax.set_xlabel("Delay Rate (%)", fontsize=8)
        mx = df_s["delay_pct"].max()
        ax.set_xlim(0, mx * 1.45)
    ax.set_title("Top 5 Most Delayed Airlines", fontsize=10, pad=10)
    fig.tight_layout()
    return fig


def _top_routes_chart(df: pd.DataFrame):
    fig, ax = dark_fig(figsize=(10, 3.6))
    if df.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                color="#8b949e", fontsize=10, transform=ax.transAxes)
    else:
        df_s = df.sort_values("delay_pct")
        colors = ["#f85149" if v >= 50 else "#d29922" if v >= 30 else "#3fb950"
                  for v in df_s["delay_pct"]]
        bars = ax.barh(df_s["destination"], df_s["delay_pct"],
                       color=colors, edgecolor="none", height=0.5)
        for bar, (_, row) in zip(bars, df_s.iterrows()):
            ax.text(row["delay_pct"] + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{row["delay_pct"]:.1f}%  ({int(row["total"])} flights)',
                    va="center", color="#8b949e", fontsize=8)
        ax.set_xlabel("Delay Rate (%)", fontsize=8)
        mx = df_s["delay_pct"].max()
        ax.set_xlim(0, mx * 1.45)
    ax.set_title("Top 5 Most Delayed Routes (Destination)", fontsize=10, pad=10)
    fig.tight_layout()
    return fig


def _top_airline_routes_chart(df: pd.DataFrame):
    fig, ax = dark_fig(figsize=(10, 3.6))
    if df.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                color="#8b949e", fontsize=10, transform=ax.transAxes)
    else:
        df_s = df.sort_values("delay_pct")
        colors = ["#f85149" if v >= 50 else "#d29922" if v >= 30 else "#3fb950"
                  for v in df_s["delay_pct"]]
        bars = ax.barh(df_s["label"], df_s["delay_pct"],
                       color=colors, edgecolor="none", height=0.5)
        for bar, (_, row) in zip(bars, df_s.iterrows()):
            ax.text(row["delay_pct"] + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{row["delay_pct"]:.1f}%  ({int(row["total"])} flights)',
                    va="center", color="#8b949e", fontsize=8)
        ax.set_xlabel("Delay Rate (%)", fontsize=8)
        mx = df_s["delay_pct"].max()
        ax.set_xlim(0, mx * 1.45)
    ax.set_title("Top 5 Most Delayed Airline + Route Combos", fontsize=10, pad=10)
    fig.tight_layout()
    return fig


# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

.stApp {
    background-image:
        linear-gradient(rgba(8,12,28,0.62), rgba(8,12,28,0.72)),
        url('https://images.unsplash.com/photo-1542296332-2e4473faf563?w=1600&q=85&auto=format&fit=crop');
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
    font-family: 'DM Sans', sans-serif;
}
.stApp, .stApp p, .stApp label, .stApp span,
.stApp div, h1, h2, h3, .stMarkdown, .stCaption,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: white !important; }
.stTextInput input, .stDateInput input {
    color: #1e293b !important; background: rgba(255,255,255,0.92) !important;
}
.stFormSubmitButton button { color: white !important; }

/* Remove white sticky header/toolbar that covers content when scrolling */
[data-testid="stHeader"],
[data-testid="stHeader"] > *,
header[data-testid="stHeader"],
.stApp > header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[class*="toolbar"],
[class*="StatusWidget"] {
    background: transparent !important;
    background-color: transparent !important;
    backdrop-filter: none !important;
    box-shadow: none !important;
}
[data-testid="stMetricLabel"] p { font-size: 0.72rem !important; }
[data-testid="stMetricValue"]   { font-size: 1.05rem !important; }

/* Transparent full-screen (expand) button on charts */
[data-testid="StyledFullScreenButton"],
[data-testid="StyledFullScreenButton"] button,
button[title="View fullscreen"],
.stPlotlyChart button[title*="fullscreen"],
.stPlotlyChart button[title*="Fullscreen"] {
    background: transparent !important;
    background-color: transparent !important;
    border: 1px solid rgba(139,148,158,0.3) !important;
    color: #8b949e !important;
    fill: #8b949e !important;
}
[data-testid="StyledFullScreenButton"]:hover,
[data-testid="StyledFullScreenButton"] button:hover,
button[title="View fullscreen"]:hover,
.stPlotlyChart button[title*="fullscreen"]:hover,
.stPlotlyChart button[title*="Fullscreen"]:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: #8b949e !important;
    fill: #8b949e !important;
}

/* Transparent Refresh Data button */
[data-testid="stButton"] button {
    background: transparent !important;
    background-color: transparent !important;
    border: 1px solid rgba(139,148,158,0.4) !important;
    color: #c9d1d9 !important;
}
[data-testid="stButton"] button:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: #8b949e !important;
    color: white !important;
}

.metric-card {
    background: rgba(22,27,34,0.92); border: 1px solid #21262d;
    border-radius: 12px; padding: 1.4rem 1.2rem 1.2rem; text-align: center; min-width: 0;
}
.metric-card:hover { border-color: #388bfd; }
.metric-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8b949e; margin-bottom: 0.6rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.metric-value {
    font-family: 'DM Mono', monospace; font-size: 1.65rem;
    font-weight: 500; color: #e6edf3; line-height: 1.1; white-space: nowrap;
}
.metric-delta { font-size: 0.72rem; margin-top: 0.45rem; color: #8b949e; }
.metric-delta.good { color: #3fb950; }
.metric-delta.warn { color: #d29922; }
.metric-delta.bad  { color: #f85149; }

.section-header {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #388bfd;
    margin: 2.5rem 0 1rem 0; padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
}
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.styled-table th {
    font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: #8b949e; padding: 0.6rem 1rem; text-align: left;
    border-bottom: 1px solid #21262d;
}
.styled-table td {
    padding: 0.65rem 1rem; border-bottom: 1px solid #161b22;
    color: #c9d1d9; font-family: 'DM Mono', monospace; font-size: 0.8rem;
}
.styled-table tr:hover td { background: #161b22; }
.route-history-wrap {
    background: rgba(22,27,34,0.88); border: 1px solid #21262d;
    border-radius: 12px; padding: 1.2rem 1.2rem 0.8rem; margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Ensure prediction_log table exists on startup
_ensure_log_table()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_predictor, tab_dashboard = st.tabs(["✈️  Predictor", "📊  Model Dashboard"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ════════════════════════════════════════════════════════════════════════════════

with tab_predictor:
    st.title("✈ KLIA Flight Delay Predictor")
    st.write("Enter your flight number and date. We'll predict whether your flight will be delayed.")

    if MODEL is None:
        st.error("Model not loaded. Run `run.bat` first.")
        st.stop()

    st.divider()

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            flight_number = st.text_input("Flight Number", placeholder="e.g. AK101",
                                           help="IATA code on your boarding pass").upper().strip()
        with c2:
            flight_date = st.date_input("Departure Date", value=date.today(),
                                         min_value=date.today()-timedelta(days=365),
                                         max_value=date.today()+timedelta(days=30))
        manual_time = st.text_input("Scheduled Time",
                                      placeholder="HH:MM (24h) — only needed if flight not in database",
                                      help="Use 24-hour format, e.g. 14:30 for 2:30 PM. Leave blank to auto-fetch.").strip()
        submitted = st.form_submit_button("Check Flight", type="primary",
                                           use_container_width=True)

    if submitted:
        if not flight_number:
            st.warning("Please enter a flight number.")
            st.stop()

        plane_anim = st.empty()
        plane_anim.markdown("""
        <div style="text-align:center;padding:24px 0 16px;">
            <style>
            .plane-fly { display:inline-block;font-size:2.4rem;animation:flyplane 1.6s ease-in-out infinite; }
            @keyframes flyplane {
                0%   { transform:translateX(-18px) rotate(-5deg);opacity:0.6; }
                50%  { transform:translateX(18px)  rotate(5deg); opacity:1;   }
                100% { transform:translateX(-18px) rotate(-5deg);opacity:0.6; }
            }
            </style>
            <div class="plane-fly">✈</div>
            <div style="font-size:0.88rem;color:#64748b;margin-top:8px;font-weight:500;">
                Fetching flight data and live weather…
            </div>
        </div>""", unsafe_allow_html=True)

        result, error = predict(flight_number, flight_date.strftime("%Y-%m-%d"), manual_time or None)
        plane_anim.empty()

        if error:
            st.error(error)
            if not RAPIDAPI_KEY:
                st.info("Add `RAPIDAPI_KEY` to your `.env` for live flight lookup.")
            st.stop()

        prob = result["probability"]

        # Verdict using optimised thresholds
        if prob < T_LOW:
            verdict, icon, color = "Unlikely Delayed", '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>', "#16a34a"
        elif prob < T_HIGH:
            verdict, icon, color = "Could Be Delayed", '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>', "#d97706"
        else:
            verdict, icon, color = "Likely Delayed",   '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>', "#e74c3c"

        st.markdown(f"""
        <div style="background:{color};border-radius:14px;padding:28px 24px;
                    text-align:center;color:white;margin:12px 0;">
            <div style="font-size:1.8rem;font-weight:700;margin-bottom:6px;">{icon} {verdict}</div>
            <div style="font-size:1rem;opacity:0.9;">{prob}% probability of delay</div>
            <div style="font-size:0.82rem;opacity:0.7;margin-top:8px;">
                {result['flight']} &middot; {result['destination']} &middot; {result['departure']}
            </div>
        </div>""", unsafe_allow_html=True)

        # Probability bar — solid colour matches verdict
        bar_color = "#16a34a" if prob < T_LOW else ("#d97706" if prob < T_HIGH else "#e74c3c")

        st.markdown(f"""
        <div style="margin:4px 0 18px;">
            <div style="display:flex;justify-content:space-between;
                        font-size:0.78rem;color:#cbd5e1;margin-bottom:5px;">
                <span>Delay Probability</span>
                <span style="font-weight:600">{prob}%</span>
            </div>
            <div style="background:#21262d;border-radius:20px;height:12px;overflow:hidden;">
                <div style="width:{prob}%;height:100%;background:{bar_color};
                            transition:width 0.6s ease;"></div>
            </div>
            <div style="position:relative;font-size:0.62rem;color:#94a3b8;margin-top:4px;height:1.1em;">
                <span style="position:absolute;left:0;">0%</span>
                <span style="position:absolute;left:{T_LOW}%;color:#3fb950;transform:translateX(-50%);">▏{T_LOW}%</span>
                <span style="position:absolute;left:{T_HIGH}%;color:#d29922;transform:translateX(-50%);">▏{T_HIGH}%</span>
                <span style="position:absolute;right:0;">100%</span>
            </div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Airline",     result["airline"])
        c2.metric("Destination", result["destination"])
        c3.metric("Departure",   result["departure"])

        c4, c5, c6 = st.columns(3)
        c4.metric("Airline Delay Rate", result["delay_rate"])
        c5.metric("Temperature",        result["temp"])
        c6.metric("Weather",            f"{result['rain']} · {result['wind']}")

        st.caption(f"Data source: {result['source']} · Predictions are indicative only.")
        st.markdown(
            "<div style='font-size:0.72rem;color:#8b949e;margin-top:-0.4rem;'>"
            "A delay is defined as departure <strong style='color:#c9d1d9;'>15 or more minutes</strong> "
            "past the scheduled departure time.</div>",
            unsafe_allow_html=True,
        )

        # Persist to PostgreSQL
        _log_prediction_db(
            flight_number, result["airline"], result["destination"],
            result["probability"], result["delayed"]
        )

        # Route history chart
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:600;letter-spacing:0.1em;"
            "text-transform:uppercase;color:#388bfd;margin:1.5rem 0 0.6rem;'>"
            "<svg width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='#388bfd' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;margin-right:5px;'><polyline points='22 7 13.5 15.5 8.5 10.5 2 17'/><polyline points='16 7 22 7 22 13'/></svg>"
            "Airline &amp; Route Delay History</div>",
            unsafe_allow_html=True,
        )
        _rates = get_rates(result["airline"], result["destination"],
                           int(result["departure"].split(":")[0]) if ":" in result["departure"] else 8)
        hist_fig = _route_history_chart(result["airline"], result["destination"], _rates)
        st.pyplot(hist_fig, use_container_width=True)
        plt.close(hist_fig)
        st.markdown(
            "<div style='font-size:0.72rem;color:#8b949e;padding:0.4rem 0 0.2rem;'>"
            "Rates shown as % probability of delay. Sourced from historical departures DB.</div>",
            unsafe_allow_html=True,
        )



# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    dmeta, is_demo = load_dashboard_meta()

    hdr_left, hdr_right = st.columns([6, 1])
    with hdr_left:
        st.markdown("## Model Performance Dashboard")
        badge = (
            '<span style="background:#2d1f00;color:#d29922;border:1px solid #9e6a03;'
            'padding:0.2rem 0.6rem;border-radius:20px;font-size:0.7rem;font-weight:600;">Demo data</span>'
            if is_demo else
            '<span style="background:#0d2d16;color:#3fb950;border:1px solid #238636;'
            'padding:0.2rem 0.6rem;border-radius:20px;font-size:0.7rem;font-weight:600;">Live</span>'
        )
        st.markdown(badge, unsafe_allow_html=True)
    with hdr_right:
        st.markdown("<div style='padding-top:2.1rem;'>", unsafe_allow_html=True)
        if st.button("↺  Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Top row: Prediction Accuracy + Prediction Volume side by side ──────────
    dash_col1, dash_col2 = st.columns(2, gap="medium")

    with dash_col1:
        st.markdown('<div class="section-header">Prediction Accuracy — Last 7 Days</div>',
                    unsafe_allow_html=True)
        acc_fig, correct_counts, uncertain_counts, incorrect_counts = _accuracy_7day_chart()
        st.pyplot(acc_fig, use_container_width=True)
        plt.close(acc_fig)

        total_known     = sum(correct_counts) + sum(incorrect_counts)
        total_correct   = sum(correct_counts)
        total_uncertain = sum(uncertain_counts)
        acc_rate        = (total_correct / total_known * 100) if total_known else 0

        a1, a2, a3 = st.columns(3)
        a1.metric("Correct",         total_correct)
        a2.metric("Pending",         total_uncertain)
        a3.metric("Accuracy",        f"{acc_rate:.1f}%" if total_known else "—")

    with dash_col2:
        st.markdown('<div class="section-header">Prediction Volume — Last 30 Days</div>',
                    unsafe_allow_html=True)
        vol_fig = _prediction_volume_chart()
        st.pyplot(vol_fig, use_container_width=True)
        plt.close(vol_fig)

    # ── Row 2: Delay by Hour + Delay by Day of Week ───────────────────────────
    row2_left, row2_right = st.columns(2, gap="medium")

    with row2_left:
        st.markdown('<div class="section-header">Delay Rate by Hour of Day</div>',
                    unsafe_allow_html=True)
        df_hour = _load_delay_by_hour()
        fig_hour = _delay_by_hour_chart(df_hour)
        st.pyplot(fig_hour, use_container_width=True)
        plt.close(fig_hour)

    with row2_right:
        st.markdown('<div class="section-header">Delay Rate by Day of Week</div>',
                    unsafe_allow_html=True)
        df_dow = _load_delay_by_dow()
        fig_dow = _delay_by_dow_chart(df_dow)
        st.pyplot(fig_dow, use_container_width=True)
        plt.close(fig_dow)

    # ── Row 3: Top Airlines + Top Routes ─────────────────────────────────────
    row3_left, row3_right = st.columns(2, gap="medium")

    with row3_left:
        st.markdown('<div class="section-header">Top 5 Most Delayed Airlines</div>',
                    unsafe_allow_html=True)
        df_airlines = _load_top_delayed_airlines()
        fig_airlines = _top_airlines_chart(df_airlines)
        st.pyplot(fig_airlines, use_container_width=True)
        plt.close(fig_airlines)

    with row3_right:
        st.markdown('<div class="section-header">Top 5 Most Delayed Routes</div>',
                    unsafe_allow_html=True)
        df_routes = _load_top_delayed_routes()
        fig_routes = _top_routes_chart(df_routes)
        st.pyplot(fig_routes, use_container_width=True)
        plt.close(fig_routes)

    # ── Row 4: Top Airline + Route Combos + Overall Delay Distribution (side by side) ───
    row4_left, row4_right = st.columns(2, gap="medium")

    with row4_left:
        st.markdown('<div class="section-header">Top 5 Most Delayed Airline + Route Combos</div>',
                    unsafe_allow_html=True)
        df_combo = _load_top_delayed_airline_routes()
        fig_combo = _top_airline_routes_chart(df_combo)
        st.pyplot(fig_combo, use_container_width=True)
        plt.close(fig_combo)

    with row4_right:
        st.markdown('<div class="section-header">Overall Delay Distribution</div>',
                    unsafe_allow_html=True)
        delay_stats = _load_overall_delay_stats()
        fig_donut = _overall_delay_donut_chart(delay_stats)
        st.pyplot(fig_donut, use_container_width=True)
        plt.close(fig_donut)
        
        # Display summary metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Flights", f"{delay_stats['total']:,}\n(in database)", help="Total flights in database with actual departure times")
        metric_col2.metric("Delayed (>15 min)", f"{delay_stats['delayed']:,}")
        metric_col3.metric("On-Time", f"{delay_stats['not_delayed']:,}")
