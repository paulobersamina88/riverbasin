
# REVISED app.py (fixed JSON parsing + safer file handling)

import json
import io
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(
    page_title="PH River Basin Rainfall Hazard Dashboard",
    page_icon="🌧️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_GEOJSON = DATA_DIR / "major_river_basins_simplified.geojson"
DEFAULT_SAMPLE = DATA_DIR / "sample_basin_rainfall.csv"
DEFAULT_TYPHOON = DATA_DIR / "sample_typhoon_track.csv"

PH_CENTER = [12.7, 122.3]
MAP_ZOOM = 5.5
MANILA_TZ = "Asia/Manila"
REQUEST_TIMEOUT = 25


# -----------------------------
# SAFE LOADERS
# -----------------------------
@st.cache_data(show_spinner=False)
def load_geojson(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GeoJSON missing: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_sample_data(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sample CSV missing: {p}")
    return pd.read_csv(p)


def normalize_geojson(uploaded_file) -> dict:
    if uploaded_file is not None:
        return json.load(uploaded_file)

    if DEFAULT_GEOJSON.exists():
        return load_geojson(str(DEFAULT_GEOJSON))

    st.error("Missing default basin GeoJSON. Please upload one.")
    st.stop()


# -----------------------------
# GEOJSON → DF
# -----------------------------
@st.cache_data(show_spinner=False)
def geojson_to_basin_df(geojson_dict: dict) -> pd.DataFrame:
    rows = []
    for feat in geojson_dict.get("features", []):
        p = feat.get("properties", {})
        rows.append({
            "basin_name": p.get("basin_name", "Unknown Basin"),
            "region": p.get("region", ""),
            "lat": pd.to_numeric(p.get("lat"), errors="coerce"),
            "lon": pd.to_numeric(p.get("lon"), errors="coerce"),
            "threshold24_mm": pd.to_numeric(p.get("threshold24_mm", 100), errors="coerce"),
            "threshold72_mm": pd.to_numeric(p.get("threshold72_mm", 200), errors="coerce"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["threshold24_mm"] = df["threshold24_mm"].fillna(100.0)
    df["threshold72_mm"] = df["threshold72_mm"].fillna(200.0)
    return df


# -----------------------------
# WEATHER FETCH
# -----------------------------
def fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_openmeteo_forecast(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=precipitation&forecast_days=7"
    )
    return fetch_json(url)


def fetch_openmeteo_historical(lat, lon, start, end):
    url = (
        f"https://historical-forecast-api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
        "&hourly=precipitation"
    )
    return fetch_json(url)


# -----------------------------
# FIXED LIVE FUNCTION
# -----------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def get_live_forecast_for_basins(basin_df_json: str):
    # FIX: safe JSON parsing
    basin_df = pd.read_json(io.StringIO(basin_df_json))

    if basin_df.empty:
        return pd.DataFrame(), pd.Timestamp.now(), []

    out = []
    failures = []

    today = pd.Timestamp.now().date()
    start = str(today - pd.Timedelta(days=3))
    end = str(today)

    for _, row in basin_df.iterrows():
        try:
            forecast = fetch_openmeteo_forecast(row["lat"], row["lon"])
            historical = fetch_openmeteo_historical(row["lat"], row["lon"], start, end)

            rain = pd.Series(forecast["hourly"]["precipitation"]).fillna(0)
            hist = pd.Series(historical["hourly"]["precipitation"]).fillna(0)

            out.append({
                "basin_name": row["basin_name"],
                "forecast_rain_24h_mm": rain[:24].sum(),
                "forecast_rain_72h_mm": rain[:72].sum(),
                "antecedent_rain_72h_mm": hist[-72:].sum(),
            })

        except Exception as e:
            failures.append(f"{row['basin_name']}: {e}")
            out.append({
                "basin_name": row["basin_name"],
                "forecast_rain_24h_mm": np.nan,
                "forecast_rain_72h_mm": np.nan,
                "antecedent_rain_72h_mm": np.nan,
            })

    return pd.DataFrame(out), pd.Timestamp.now(), failures


# -----------------------------
# MAIN
# -----------------------------
st.title("🌧️ River Basin Dashboard (Fixed Version)")

with st.sidebar:
    data_mode = st.radio("Forecast source", ["Sample data", "Live Open-Meteo"])
    uploaded_geojson = st.file_uploader("Upload basin GeoJSON", type=["geojson"])

    # DEBUG
    st.write("GeoJSON exists:", DEFAULT_GEOJSON.exists())
    st.write("Sample exists:", DEFAULT_SAMPLE.exists())

geojson_data = normalize_geojson(uploaded_geojson)
basin_master = geojson_to_basin_df(geojson_data)

if data_mode == "Sample data":
    basin_values = load_sample_data(str(DEFAULT_SAMPLE))
else:
    basin_json = basin_master.to_json(orient="records")
    basin_values, _, failures = get_live_forecast_for_basins(basin_json)

    if failures:
        st.warning("Some basins failed:")
        st.write(failures)

st.dataframe(basin_values)
