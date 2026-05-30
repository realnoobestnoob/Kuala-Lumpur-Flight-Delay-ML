"""
streamlit_app/app.py  —  KLIA Flight Delay Predictor
Run:  streamlit run streamlit_app/app.py

Changes from previous version
------------------------------
- predict_proba() replaced with _model_predict_proba() helper that handles
  both lgb.Booster (focal-loss model from lgb.train()) and sklearn
  LGBMClassifier — focal model uses .predict(), sklearn uses .predict_proba()
- apply_model_input() now applies LOO encoder from meta_objects before
  returning feature matrix, matching the training-time encoding boundary
- get_rates() extended with airline_delay_rate_ewm, route_delay_rate_ewm
- build_features() extended with 3 interaction features:
    congestion_x_airline_rate, peak_x_airline_rate, route_x_weather
  (lag_x_airline_rate and ewm_x_congestion removed — hurt precision in v2)
"""

import json, os, sys
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib, numpy as np, pandas as pd, requests, streamlit as st
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.common import get_engine, load_config

st.set_page_config(page_title="KLIA Delay Predictor", page_icon="✈",
                   layout="centered", initial_sidebar_state="collapsed")

_env = ROOT / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

RAPIDAPI_KEY    = os.environ.get("RAPIDAPI_KEY", "")
KLIA_LAT, KLIA_LON = 2.7456, 101.7072


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
    """
    Return P(delayed) as a float, handling both model types:
      - lgb.Booster   (focal-loss model saved via lgb.train())  → .predict()
      - LGBMClassifier (sklearn API)                            → .predict_proba()
    """
    import lightgbm as lgb
    if isinstance(model, lgb.Booster):
        return float(model.predict(X)[0])
    return float(model.predict_proba(X)[0, 1])


# ── Data lookups ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def lookup_db(fn, fd):
    try:
        engine = get_engine(CFG)
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
        # existing rates
        "airline_delay_rate":0.30, "route_delay_rate":0.30,
        "airline_hour_delay_rate":0.30, "route_hour_delay_rate":0.30,
        "delay_ratio_prev_3_airline":0.30,
        "route_delay_rate_7d":0.30, "route_delay_rate_30d":0.30,
        "airline_encoded":0.30, "destination_encoded":0.30, "aircraft_encoded":0.30,
        "flights_per_hour":5.0, "concurrent_departures":5.0,
        "unique_destinations_per_hour":3.0,
        "prev_aircraft_delayed":0, "prev_aircraft_delayed_1":0,
        "prev_aircraft_delayed_2":0, "prev_aircraft_delayed_3":0,
        # NEW: EWM rates (default to same as expanding mean)
        "airline_delay_rate_ewm":0.30, "route_delay_rate_ewm":0.30,
    }
    is_del = "CASE WHEN actual_departure>scheduled_departure+INTERVAL '15 minutes' THEN 1.0 ELSE 0.0 END"
    try:
        engine = get_engine(CFG)
        with engine.connect() as conn:
            def q(sql, p={}):
                v = conn.execute(text(sql), p).scalar()
                return float(v) if v is not None else None

            # Airline delay rate (expanding mean)
            v = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a)", {"a":airline})
            if v:
                r.update(airline_delay_rate=v, airline_encoded=v,
                         delay_ratio_prev_3_airline=v,
                         airline_delay_rate_ewm=v)   # EWM fallback = expanding mean

            # Airline × hour
            v = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a) AND EXTRACT(HOUR FROM scheduled_departure)=:h", {"a":airline,"h":hour})
            if v: r["airline_hour_delay_rate"] = v

            # Route delay rate
            v = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a) AND UPPER(destination)=UPPER(:d)", {"a":airline,"d":destination})
            if v:
                r.update(route_delay_rate=v, destination_encoded=v,
                         route_delay_rate_ewm=v)     # EWM fallback = expanding mean

            # Route × hour
            v = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a) AND UPPER(destination)=UPPER(:d) AND EXTRACT(HOUR FROM scheduled_departure)=:h", {"a":airline,"d":destination,"h":hour})
            if v: r["route_hour_delay_rate"] = v

            # Rolling windows
            for days, key in [(7,"route_delay_rate_7d"), (30,"route_delay_rate_30d")]:
                v = q(f"SELECT AVG({is_del}) FROM departures WHERE UPPER(airline)=UPPER(:a) AND UPPER(destination)=UPPER(:d) AND date>=CURRENT_DATE-INTERVAL '{days} days'", {"a":airline,"d":destination})
                if v: r[key] = v

            # Previous aircraft lag
            v = q(f"SELECT {is_del} FROM departures WHERE UPPER(airline)=UPPER(:a) ORDER BY date DESC,scheduled_departure DESC LIMIT 1", {"a":airline})
            if v:
                iv = int(v)
                r.update(prev_aircraft_delayed=iv, prev_aircraft_delayed_1=iv)

            # Congestion
            v = q("SELECT COUNT(*)::float/NULLIF(COUNT(DISTINCT date),0) FROM departures WHERE EXTRACT(HOUR FROM scheduled_departure)=:h", {"h":hour})
            if v: r.update(flights_per_hour=v, concurrent_departures=v)

    except Exception:
        pass
    return r


# ── Feature construction ──────────────────────────────────────────────────────

def build_features(flight, fd, weather, rates):
    sched   = str(flight["scheduled_departure"])[:5]
    dt      = datetime.strptime(f"{fd} {sched}", "%Y-%m-%d %H:%M")
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
        # ── NEW: interaction features ─────────────────────────────────────────
        # congestion × airline rate
        "congestion_x_airline_rate": rates["flights_per_hour"] * rates["airline_delay_rate"],
        # peak hour amplifies airline delay pattern
        "peak_x_airline_rate": int(hour in peak_am+peak_pm) * rates["airline_delay_rate"],
        # bad route + bad weather
        "route_x_weather": rates["route_delay_rate"] * int(
            weather["wind_gusts_10m"]>30 or weather["precipitation"]>5
        ),
    }
    return pd.DataFrame([row])


def apply_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare feature matrix for the model.
    Applies LOO encoder from meta_objects if available (fixes leakage boundary).
    Handles both lgb.Booster and sklearn LGBMClassifier.
    """
    features = META["features"]

    # Apply LOO encoder if saved (classification v2+)
    if META_OBJECTS and META_OBJECTS.get("loo_encoder") is not None:
        try:
            df = META_OBJECTS["loo_encoder"].transform(df)
        except Exception:
            pass   # graceful fallback if column mismatch

    # Legacy: scaler + PCA path (classification v1)
    if META_OBJECTS and META_OBJECTS.get("scaler") is not None:
        sel = META_OBJECTS["selected_features"]
        for c in sel:
            if c not in df.columns:
                df[c] = 0.0
        try:
            X = META_OBJECTS["scaler"].transform(df[sel].fillna(0))
            X = META_OBJECTS["pca"].transform(X)
            return pd.DataFrame(X, columns=META_OBJECTS["pca_cols"])
        except Exception:
            pass

    for c in features:
        if c not in df.columns:
            df[c] = 0.0
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

    threshold = META.get("decision_threshold", 0.5)
    proba     = _model_predict_proba(MODEL, X)

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


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.stApp {
    background-image:
        linear-gradient(rgba(10,15,35,0.58), rgba(10,15,35,0.58)),
        url('https://images.unsplash.com/photo-1542296332-2e4473faf563?w=1600&q=85&auto=format&fit=crop');
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
}
/* White text throughout */
.stApp, .stApp p, .stApp label, .stApp span,
.stApp div, h1, h2, h3, .stMarkdown, .stCaption,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: white !important;
}
/* Inputs stay readable */
.stTextInput input, .stDateInput input {
    color: #1e293b !important;
    background: rgba(255,255,255,0.92) !important;
}
/* Buttons */
.stFormSubmitButton button { color: white !important; }
/* Smaller metric labels and values */
[data-testid="stMetricLabel"] p { font-size: 0.72rem !important; }
[data-testid="stMetricValue"]   { font-size: 1.05rem !important; }
</style>
""", unsafe_allow_html=True)

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

    manual_time = st.text_input("Scheduled Time (optional)",
                                  placeholder="HH:MM — only needed if flight not in database",
                                  help="Leave blank to auto-fetch").strip()

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
        .plane-fly {
            display:inline-block;
            font-size:2.4rem;
            animation: flyplane 1.6s ease-in-out infinite;
        }
        @keyframes flyplane {
            0%   { transform: translateX(-18px) rotate(-5deg); opacity:0.6; }
            50%  { transform: translateX(18px)  rotate(5deg);  opacity:1;   }
            100% { transform: translateX(-18px) rotate(-5deg); opacity:0.6; }
        }
        </style>
        <div class="plane-fly">✈</div>
        <div style="font-size:0.88rem;color:#64748b;margin-top:8px;font-weight:500;">
            Fetching flight data and live weather…
        </div>
    </div>""", unsafe_allow_html=True)

    result, error = predict(flight_number, flight_date.strftime("%Y-%m-%d"),
                            manual_time or None)
    plane_anim.empty()

    if error:
        st.error(error)
        if not RAPIDAPI_KEY:
            st.info("Add `RAPIDAPI_KEY` to your `.env` for live flight lookup.")
        st.stop()

    prob     = result["probability"]
    if prob < 40:
        verdict = "Unlikely Delayed"
        icon    = "✅"
        color   = "#16a34a"
    elif prob < 65:
        verdict = "Could Be Delayed"
        icon    = "⚠️"
        color   = "#d97706"
    else:
        verdict = "Likely Delayed"
        icon    = "🔴"
        color   = "#e74c3c"

    st.markdown(f"""
    <div style="background:{color};border-radius:14px;padding:28px 24px;
                text-align:center;color:white;margin:12px 0;">
        <div style="font-size:1.8rem;font-weight:700;margin-bottom:6px;">{icon} {verdict}</div>
        <div style="font-size:1rem;opacity:0.9;">{prob}% probability of delay</div>
        <div style="font-size:0.82rem;opacity:0.7;margin-top:8px;">
            {result['flight']} &middot; {result['destination']} &middot; {result['departure']}
        </div>
    </div>""", unsafe_allow_html=True)

    # Single solid colour bar: green < 40%, orange 40–65%, red > 65%
    if prob < 40:
        bar_color = "#16a34a"
    elif prob < 65:
        bar_color = "#d97706"
    else:
        bar_color = "#dc2626"

    st.markdown(f"""
    <div style="margin:4px 0 18px;">
        <div style="display:flex;justify-content:space-between;
                    font-size:0.78rem;color:#cbd5e1;margin-bottom:5px;">
            <span>Delay Probability</span><span style="font-weight:600">{prob}%</span>
        </div>
        <div style="background:#e2e8f0;border-radius:20px;height:12px;overflow:hidden;">
            <div style="width:{prob}%;height:100%;border-radius:20px;
                        background:{bar_color};transition:width 0.6s ease;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.62rem;color:#94a3b8;margin-top:3px;">
            <span>0% On Time</span><span>100% Certain Delay</span>
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
