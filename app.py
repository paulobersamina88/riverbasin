
import json
from io import StringIO
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

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_GEOJSON = DATA_DIR / "major_river_basins_simplified.geojson"
DEFAULT_SAMPLE = DATA_DIR / "sample_basin_rainfall.csv"
DEFAULT_TYPHOON = DATA_DIR / "sample_typhoon_track.csv"

PH_CENTER = [12.7, 122.3]
MAP_ZOOM = 5.5

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data(show_spinner=False)
def load_geojson(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_sample_data(path: Path):
    return pd.read_csv(path)

def geojson_to_basin_df(geojson_dict):
    rows = []
    for feat in geojson_dict["features"]:
        p = feat["properties"]
        rows.append(
            {
                "basin_name": p["basin_name"],
                "region": p.get("region", ""),
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "threshold24_mm": p.get("threshold24_mm", 100),
                "threshold72_mm": p.get("threshold72_mm", 200),
            }
        )
    return pd.DataFrame(rows)

# -----------------------------
# WEATHER
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_openmeteo_forecast(lat, lon):
    # 7 days forecast, hourly precipitation
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=precipitation"
        "&forecast_days=7"
        "&timezone=Asia%2FManila"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_openmeteo_historical(lat, lon, start_date, end_date):
    url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=precipitation"
        "&timezone=Asia%2FManila"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def summarize_precip_from_hourly(payload):
    hourly = payload.get("hourly", {})
    arr = hourly.get("precipitation", [])
    if not arr:
        return 0.0, 0.0
    rain24 = float(np.nansum(arr[:24]))
    rain72 = float(np.nansum(arr[:72]))
    return rain24, rain72

def summarize_historical_72h(payload):
    hourly = payload.get("hourly", {})
    arr = hourly.get("precipitation", [])
    if not arr:
        return 0.0
    return float(np.nansum(arr[-72:]))

def live_forecast_for_basins(basin_df):
    out = []
    today = pd.Timestamp.now(tz="Asia/Manila").date()
    start_date = str(today - pd.Timedelta(days=3))
    end_date = str(today)
    progress = st.progress(0, text="Fetching live basin forecasts...")
    for i, row in basin_df.iterrows():
        try:
            forecast = fetch_openmeteo_forecast(row["lat"], row["lon"])
            rain24, rain72 = summarize_precip_from_hourly(forecast)
            hist = fetch_openmeteo_historical(row["lat"], row["lon"], start_date, end_date)
            ant72 = summarize_historical_72h(hist)
        except Exception:
            rain24, rain72, ant72 = np.nan, np.nan, np.nan
        out.append(
            {
                "basin_name": row["basin_name"],
                "region": row["region"],
                "lat": row["lat"],
                "lon": row["lon"],
                "threshold24_mm": row["threshold24_mm"],
                "threshold72_mm": row["threshold72_mm"],
                "forecast_rain_24h_mm": rain24,
                "forecast_rain_72h_mm": rain72,
                "antecedent_rain_72h_mm": ant72,
                "river_stage_factor": 1.0,
                "dam_release_factor": 1.0,
            }
        )
        progress.progress((i + 1) / len(basin_df), text=f"Fetched {i+1}/{len(basin_df)} basins")
    progress.empty()
    return pd.DataFrame(out)

# -----------------------------
# HAZARD LOGIC
# -----------------------------
def grade_color(level):
    return {
        "Low": "#2ecc71",
        "Moderate": "#f1c40f",
        "High": "#e67e22",
        "Severe": "#e74c3c",
        "Extreme": "#8e0000",
        "No Data": "#9aa0a6",
    }.get(level, "#9aa0a6")

def compute_effective_rain(row, window="24h"):
    if window == "24h":
        rain = row["forecast_rain_24h_mm"]
        threshold = row["threshold24_mm"]
    else:
        rain = row["forecast_rain_72h_mm"]
        threshold = row["threshold72_mm"]

    ant_factor = 1 + min((row.get("antecedent_rain_72h_mm", 0) or 0) / 300.0, 0.35)
    river_factor = row.get("river_stage_factor", 1.0) or 1.0
    dam_factor = row.get("dam_release_factor", 1.0) or 1.0
    effective_rain = rain * ant_factor * river_factor * dam_factor
    ratio = effective_rain / threshold if threshold and threshold > 0 else np.nan
    return effective_rain, ratio, threshold

def classify_ratio(ratio):
    if pd.isna(ratio):
        return "No Data"
    if ratio >= 1.50:
        return "Extreme"
    if ratio >= 1.00:
        return "Severe"
    if ratio >= 0.70:
        return "High"
    if ratio >= 0.40:
        return "Moderate"
    return "Low"

def add_hazard_columns(df, window="24h"):
    effective = []
    ratios = []
    thresholds = []
    levels = []
    for _, row in df.iterrows():
        e, r, t = compute_effective_rain(row, window=window)
        effective.append(e)
        ratios.append(r)
        thresholds.append(t)
        levels.append(classify_ratio(r))
    out = df.copy()
    out["selected_window"] = window
    out["threshold_selected_mm"] = thresholds
    out["effective_rain_mm"] = effective
    out["hazard_ratio"] = ratios
    out["hazard_level"] = levels
    return out

def advisory_text(row):
    basin = row["basin_name"]
    level = row["hazard_level"]
    window = row["selected_window"]
    rain = row["effective_rain_mm"]
    threshold = row["threshold_selected_mm"]
    if level in ["Extreme", "Severe"]:
        return (
            f"{level} rainfall hazard for {basin} River Basin. "
            f"Effective {window} rainfall is {rain:.1f} mm versus threshold {threshold:.1f} mm. "
            "Possible flooding in low-lying and flood-prone communities. Coordinate with LGUs and monitor official advisories."
        )
    if level == "High":
        return (
            f"High rainfall hazard for {basin} River Basin. "
            f"Effective {window} rainfall is {rain:.1f} mm. Localized flooding is possible in vulnerable areas."
        )
    if level == "Moderate":
        return (
            f"Moderate rainfall hazard for {basin} River Basin. Continue monitoring rainfall and river conditions."
        )
    if level == "No Data":
        return f"No forecast data available for {basin} River Basin at the moment."
    return f"Low rainfall hazard currently indicated for {basin} River Basin."

# -----------------------------
# MAPS
# -----------------------------
def merge_geojson_with_metrics(geojson_dict, hazard_df):
    metric_map = hazard_df.set_index("basin_name").to_dict(orient="index")
    merged = json.loads(json.dumps(geojson_dict))
    for feat in merged["features"]:
        basin = feat["properties"]["basin_name"]
        metrics = metric_map.get(basin, {})
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                v = float(v)
            feat["properties"][k] = v
    return merged

def style_function(feature):
    level = feature["properties"].get("hazard_level", "No Data")
    return {
        "fillColor": grade_color(level),
        "color": "#333333",
        "weight": 1.0,
        "fillOpacity": 0.65,
    }

def add_typhoon_overlay(m, track_df):
    if track_df is None or track_df.empty:
        return
    points = track_df[["lat", "lon"]].dropna().values.tolist()
    if len(points) >= 2:
        folium.PolyLine(points, weight=3, tooltip="Typhoon Track").add_to(m)
    for _, r in track_df.iterrows():
        popup = (
            f"<b>{r.get('name','Typhoon')}</b><br>"
            f"{r.get('datetime','')}<br>"
            f"Wind: {r.get('wind_kph','')} kph"
        )
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=5,
            weight=1,
            fill=True,
            fill_opacity=0.9,
            popup=popup,
        ).add_to(m)

def build_map(geojson_merged, track_df=None):
    m = folium.Map(location=PH_CENTER, zoom_start=MAP_ZOOM, tiles="cartodbpositron")
    tooltip = folium.GeoJsonTooltip(
        fields=["basin_name", "region", "hazard_level", "effective_rain_mm", "threshold_selected_mm"],
        aliases=["River Basin", "Region", "Hazard", "Effective Rain (mm)", "Threshold (mm)"],
        localize=True,
        sticky=False,
    )
    popup = folium.GeoJsonPopup(
        fields=[
            "basin_name",
            "hazard_level",
            "forecast_rain_24h_mm",
            "forecast_rain_72h_mm",
            "antecedent_rain_72h_mm",
            "effective_rain_mm",
            "threshold_selected_mm",
        ],
        aliases=[
            "River Basin",
            "Hazard",
            "24h Forecast (mm)",
            "72h Forecast (mm)",
            "Antecedent 72h (mm)",
            "Effective Rain (mm)",
            "Threshold (mm)",
        ],
        localize=True,
    )
    folium.GeoJson(
        geojson_merged,
        name="Basins",
        style_function=style_function,
        tooltip=tooltip,
        popup=popup,
        highlight_function=lambda x: {"weight": 3, "fillOpacity": 0.8},
    ).add_to(m)

    # Add markers at basin centroids
    for feat in geojson_merged["features"]:
        p = feat["properties"]
        lat = p.get("lat")
        lon = p.get("lon")
        if lat is None or lon is None:
            continue
        folium.CircleMarker(
            [lat, lon],
            radius=6,
            weight=1,
            color="#111",
            fill=True,
            fill_color=grade_color(p.get("hazard_level", "No Data")),
            fill_opacity=1.0,
            popup=f"{p.get('basin_name')}<br>Hazard: {p.get('hazard_level')}",
            tooltip=p.get("basin_name"),
        ).add_to(m)

    add_typhoon_overlay(m, track_df)
    folium.LayerControl().add_to(m)
    return m

# -----------------------------
# UI
# -----------------------------
st.title("🌧️ Philippine River Basin Rainfall Hazard Dashboard")
st.caption(
    "Lightweight, Project NOAH-inspired basin rainfall hazard screening for the Philippines."
)

with st.sidebar:
    st.header("Controls")
    app_mode = st.radio("Mode", ["Version 1.1", "Version 2"], index=0)
    data_mode = st.radio("Forecast source", ["Sample data", "Live Open-Meteo"], index=0)
    rain_window = st.radio("Hazard window", ["24h", "72h"], horizontal=True)
    uploaded_geojson = st.file_uploader("Optional: upload your own basin GeoJSON", type=["geojson", "json"])

# Load basin geometry
if uploaded_geojson is not None:
    geojson_data = json.load(uploaded_geojson)
else:
    geojson_data = load_geojson(DEFAULT_GEOJSON)

basin_master = geojson_to_basin_df(geojson_data)

# Load forecast data
if data_mode == "Sample data":
    basin_data = load_sample_data(DEFAULT_SAMPLE)
else:
    with st.spinner("Downloading live forecast data by basin centroid..."):
        basin_data = live_forecast_for_basins(basin_master)

# Merge missing threshold/location fields if needed
basin_data = basin_master.drop(columns=["region"], errors="ignore").merge(
    basin_data.drop(columns=["lat", "lon", "threshold24_mm", "threshold72_mm"], errors="ignore"),
    on="basin_name",
    how="left",
)
basin_data["region"] = basin_data.get("region").fillna(basin_master.set_index("basin_name").loc[basin_data["basin_name"], "region"].values)

# V2 controls
typhoon_track_df = None
if app_mode == "Version 2":
    with st.sidebar:
        st.subheader("Version 2 modifiers")
        apply_stage = st.checkbox("Apply elevated river stage factor", value=True)
        river_stage_default = st.slider("Default river stage factor", 1.0, 1.5, 1.05, 0.05)
        apply_dam = st.checkbox("Apply dam release factor", value=False)
        dam_release_default = st.slider("Default dam release factor", 1.0, 1.4, 1.0, 0.05)
        use_typhoon_overlay = st.checkbox("Show typhoon track overlay", value=False)
        upload_typhoon = st.file_uploader("Upload typhoon track CSV", type=["csv"])

    if apply_stage:
        basin_data["river_stage_factor"] = river_stage_default
    if apply_dam:
        basin_data["dam_release_factor"] = dam_release_default

    if use_typhoon_overlay:
        if upload_typhoon is not None:
            typhoon_track_df = pd.read_csv(upload_typhoon)
        elif DEFAULT_TYPHOON.exists():
            typhoon_track_df = pd.read_csv(DEFAULT_TYPHOON)

# Compute hazard
hazard_df = add_hazard_columns(basin_data, window=rain_window)
hazard_df["alert_text"] = hazard_df.apply(advisory_text, axis=1)

# Banner
critical = hazard_df[hazard_df["hazard_level"].isin(["Extreme", "Severe"])].sort_values(
    "hazard_ratio", ascending=False
)
if not critical.empty:
    names = ", ".join(critical["basin_name"].head(6).tolist())
    st.error(f"Alert: {len(critical)} basin(s) are currently at Severe/Extreme rainfall hazard in the selected {rain_window} view: {names}")
else:
    st.success(f"No basin is currently at Severe/Extreme rainfall hazard in the selected {rain_window} view.")

left, right = st.columns([1.8, 1.0])

with left:
    merged_geojson = merge_geojson_with_metrics(geojson_data, hazard_df)
    folium_map = build_map(merged_geojson, typhoon_track_df)
    map_state = st_folium(folium_map, width=None, height=680)

with right:
    st.subheader("Top 10 highest-risk basins")
    top10 = (
        hazard_df.sort_values("hazard_ratio", ascending=False)[
            ["basin_name", "region", "hazard_level", "effective_rain_mm", "threshold_selected_mm", "hazard_ratio"]
        ]
        .head(10)
        .copy()
    )
    top10["hazard_ratio"] = (top10["hazard_ratio"] * 100).round(0).astype("Int64").astype(str) + "%"
    st.dataframe(top10, use_container_width=True, hide_index=True)

    all_options = ["All"] + hazard_df.sort_values("basin_name")["basin_name"].tolist()
    selected = st.selectbox("Basin details", all_options, index=0)

    clicked_basin = None
    if map_state and map_state.get("last_active_drawing"):
        clicked_basin = map_state["last_active_drawing"]["properties"].get("basin_name")
    if clicked_basin:
        st.info(f"Clicked basin: {clicked_basin}")
        selected = clicked_basin

    if selected != "All":
        row = hazard_df.loc[hazard_df["basin_name"] == selected].iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("Hazard", row["hazard_level"])
        c2.metric("Threshold used", f'{row["threshold_selected_mm"]:.1f} mm')
        c1.metric("Forecast 24h", f'{row["forecast_rain_24h_mm"]:.1f} mm')
        c2.metric("Forecast 72h", f'{row["forecast_rain_72h_mm"]:.1f} mm')
        c1.metric("Antecedent 72h", f'{row["antecedent_rain_72h_mm"]:.1f} mm')
        c2.metric("Effective rain", f'{row["effective_rain_mm"]:.1f} mm')
        st.text_area("Suggested alert text", row["alert_text"], height=140)
        st.caption(
            "Rainfall hazard level is a screening indicator based on basin rainfall totals and modifiers, "
            "and does not yet represent calibrated flood depth or inundation extent."
        )
    else:
        counts = hazard_df["hazard_level"].value_counts().reindex(
            ["Extreme", "Severe", "High", "Moderate", "Low", "No Data"], fill_value=0
        )
        st.bar_chart(counts)

st.markdown("---")
st.markdown("### Basin forecast table")
show_cols = [
    "basin_name",
    "region",
    "forecast_rain_24h_mm",
    "forecast_rain_72h_mm",
    "antecedent_rain_72h_mm",
    "threshold24_mm",
    "threshold72_mm",
    "river_stage_factor",
    "dam_release_factor",
    "effective_rain_mm",
    "hazard_level",
]
st.dataframe(
    hazard_df[show_cols].sort_values("effective_rain_mm", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")
st.markdown(
    """
**Notes**
- Included basin polygons are simplified placeholders for demo use only.
- Live mode uses **Open-Meteo** basin-centroid precipitation as a lightweight forecast source.
- Replace the sample GeoJSON with official river basin boundaries for production.
"""
)
