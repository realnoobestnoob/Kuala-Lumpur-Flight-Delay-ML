# KLIA Flight Delay Predictor
A machine learning system that predicts flight departure delays at Kuala Lumpur International Airport (KLIA). Get real-time delay predictions and explore historical delay patterns through a dashboard.

## What It Does

**Predict flight delays** — Input a flight number and date to get an instant probability estimate of departure delays (>15 minutes late).

**Analyze delay patterns** — Interactive dashboards showing:
- Delay rates by hour of day and day of week
- Most delayed airlines and routes
- 7-day prediction accuracy tracking
- 30-day prediction volume trends
- Overall delay distribution across all flights

## Live Demo

Try it here: [KLIA Flight Delay Predictor](https://kliadelaypredictor.streamlit.app/)

## Data Sources

- **Flight schedules & actuals** — Personal Neon database
- **Weather data** — [Open-Meteo API](https://open-meteo.com)
- **Flight details** — [AeroDataBox API](https://aerodatabox.com)

## Tech Stack

- **Model** — LightGBM
- **Backend** — PostgreSQL (Neon), SQLAlchemy
- **Frontend** — Streamlit + Plotly/Matplotlib
- **APIs** — Open-Meteo, AeroDataBox, RapidAPI
- **Deployment** — Streamlit Cloud

## Model Performance

- **Accuracy** — AUC: 0.759, F1: 0.683
- **Features** — 25+ engineered features
- **Training data** — >60,000+ historical departures

## Project Structure

```
├── streamlit_app/
│   └── app.py                    # Main Streamlit dashboard
├── src/
│   ├── model/
│   │   └── predictor.py          # LightGBM model inference
│   ├── utils/
│   │   └── common.py             # DB helpers & config
│   └── etl/
│       └── import_flights_fixed.py  # ETL pipeline for Google Sheets
├── models/
│   └── delay_predictor.pkl       # Trained LightGBM model
├── requirements.txt
├── .env.example
└── README.md
```

## Contributing

Contributions welcome! Feel free to open issues or submit PRs for:
- Model improvements
- New data sources
- UI/UX enhancements
- Bug fixes

## ⚠️ Disclaimer

Predictions are indicative only and should not be used for critical decisions. Always verify with official airport sources for real flight information.
