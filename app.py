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

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_GEOJSON = DATA_DIR / "major_river_basins_simplified.geojson"
DEFAULT_SAMPLE = DATA_DIR / "sample_basin_rainfall.csv"
DEFAULT_TYPHOON = DATA_DIR / "sample_typhoon_track.csv"

PH_CENTER = [12.7, 122.3]
MAP_ZOOM = 5.5
MANILA_TZ = "Asia/Manila"
REQUEST_TIMEOUT = 25


# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data(show_spinner=False)
def load_geojson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_sample_data(path: str):
    return pd.read_csv(path)


def normalize_geojson(uploaded_file) -> dict:
    if uploaded_file is None:
        return load_geojson(str(DEFAULT_GEOJSON))
    return json.load(uploaded_file)


@st.cache_data(show_spinner=False)
def geojson_to_basin_df(geojson_dict: dict) -> pd.DataFrame:
    rows = []
    for feat in geojson_dict.get("features", []):
        p = feat.get("properties", {})
        rows.append(
            {
                "basin_name": p.get("basin_name", "Unknown Basin"),
                "region": p.get("region", ""),
                "lat": pd.to_numeric(p.get("lat"), errors="coerce"),
                "lon": pd.to_numeric(p.get("lon"), errors="coerce"),
                "threshold24_mm": pd.to_numeric(p.get("threshold24_mm", 100), errors="coerce"),
                "threshold72_mm": pd.to_numeric(p.get("threshold72_mm", 200), errors="coerce"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["basin_name", "region", "lat", "lon", "threshold24_mm", "threshold72_mm"]
        )

    df["threshold24_mm"] = df["threshold24_mm"].fillna(100.0)
    df["threshold72_mm"] = df["threshold72_mm"].fillna(200.0)
    return df


# -----------------------------
# WEATHER
# -----------------------------
def fetch_json(url: str) -> dict:
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_openmeteo_forecast(lat: float, lon: float) -> dict:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=precipitation"
        "&forecast_days=7"
        f"&timezone={MANILA_TZ.replace('/', '%2F')}"
    )
    return fetch_json(url)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_openmeteo_historical(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=precipitation"
        f"&timezone={MANILA_TZ.replace('/', '%2F')}"
    )
    return fetch_json(url)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def get_live_forecast_for_basins(basin_df_json: str):
    # Critical fix: parse the JSON string as an in-memory buffer, not as a filepath/URL.
    basin_df = pd.read_json(io.StringIO(basin_df_json))
    if basin_df.empty:
        return pd.DataFrame(), pd.Timestamp.now(tz=MANILA_TZ), []

    out = []
    failures = []
    today = pd.Timestamp.now(tz=MANILA_TZ).date()
    start_date = str(today - pd.Timedelta(days=3))
    end_date = str(today)

    for _, row in basin_df.iterrows():
        basin = row["basin_name"]
        lat = row.get("lat")
        lon = row.get("lon")

        if pd.isna(lat) or pd.isna(lon):
            failures.append(f"{basin}: missing centroid coordinates")
            out.append(
                {
                    "basin_name": basin,
                    "forecast_rain_24h_mm": np.nan,
                    "forecast_rain_72h_mm": np.nan,
                    "antecedent_rain_72h_mm": np.nan,
                }
            )
            continue

        try:
            forecast = fetch_openmeteo_forecast(float(lat), float(lon))
            historical = fetch_openmeteo_historical(float(lat), float(lon), start_date, end_date)
            rain24, rain72 = summarize_precip_from_hourly(forecast)
            ant72 = summarize_historical_72h(historical)
            out.append(
                {
                    "basin_name": basin,
                    "forecast_rain_24h_mm": rain24,
                    "forecast_rain_72h_mm": rain72,
                    "antecedent_rain_72h_mm": ant72,
                }
            )
        except Exception as exc:
            failures.append(f"{basin}: {type(exc).__name__}: {exc}")
            out.append(
                {
                    "basin_name": basin,
                    "forecast_rain_24h_mm": np.nan,
                    "forecast_rain_72h_mm": np.nan,
                    "antecedent_rain_72h_mm": np.nan,
                }
            )

    return pd.DataFrame(out), pd.Timestamp.now(tz=MANILA_TZ), failures


# -----------------------------
# PROCESSING
# -----------------------------
def summarize_precip_from_hourly(payload: dict):
    hourly = payload.get("hourly", {})
    arr = pd.to_numeric(pd.Series(hourly.get("precipitation", [])), errors="coerce").fillna(0.0).to_numpy()
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.nansum(arr[:24])), float(np.nansum(arr[:72]))


def summarize_historical_72h(payload: dict):
    hourly = payload.get("hourly", {})
    arr = pd.to_numeric(pd.Series(hourly.get("precipitation", [])), errors="coerce").fillna(0.0).to_numpy()
    if arr.size == 0:
        return np.nan
    return float(np.nansum(arr[-72:]))


def prepare_sample_data(sample_df: pd.DataFrame) -> pd.DataFrame:
    renamed = sample_df.copy()
    expected = {
        "forecast_rain_24h_mm": 0.0,
        "forecast_rain_72h_mm": 0.0,
        "antecedent_rain_72h_mm": 0.0,
        "river_stage_factor": 1.0,
        "dam_release_factor": 1.0,
    }
    legacy_map = {
        "rain_mm_24h": "forecast_rain_24h_mm",
        "rain_mm_72h": "forecast_rain_72h_mm",
    }
    renamed = renamed.rename(columns=legacy_map)
    for col, default in expected.items():
        if col not in renamed.columns:
            renamed[col] = default
    return renamed


def safe_merge_master_with_values(basin_master: pd.DataFrame, basin_values: pd.DataFrame) -> pd.DataFrame:
    master_cols = ["basin_name", "region", "lat", "lon", "threshold24_mm", "threshold72_mm"]
    value_cols = [c for c in basin_values.columns if c not in master_cols[1:]]
    merged = basin_master[master_cols].merge(
        basin_values[value_cols],
        on="basin_name",
        how="left",
    )

    defaults = {
        "forecast_rain_24h_mm": np.nan,
        "forecast_rain_72h_mm": np.nan,
        "antecedent_rain_72h_mm": 0.0,
        "river_stage_factor": 1.0,
        "dam_release_factor": 1.0,
    }
    for col, default in defaults.items():
        if col not in merged.columns:
            merged[col] = default

    for col in defaults:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged


# -----------------------------
# HAZARD LOGIC
# -----------------------------
def grade_color(level: str) -> str:
    return {
        "Low": "#2ecc71",
        "Moderate": "#f1c40f",
        "High": "#e67e22",
        "Severe": "#e74c3c",
        "Extreme": "#8e0000",
        "No Data": "#9aa0a6",
    }.get(level, "#9aa0a6")


def compute_effective_rain(row: pd.Series, window="24h"):
    rain = row["forecast_rain_24h_mm"] if window == "24h" else row["forecast_rain_72h_mm"]
    threshold = row["threshold24_mm"] if window == "24h" else row["threshold72_mm"]

    if pd.isna(rain) or pd.isna(threshold) or threshold <= 0:
        return np.nan, np.nan, threshold

    ant72 = row.get("antecedent_rain_72h_mm", 0.0)
    river_factor = row.get("river_stage_factor", 1.0)
    dam_factor = row.get("dam_release_factor", 1.0)

    ant_factor = 1 + min(max((0 if pd.isna(ant72) else ant72) / 300.0, 0), 0.35)
    river_factor = 1.0 if pd.isna(river_factor) else river_factor
    dam_factor = 1.0 if pd.isna(dam_factor) else dam_factor

    effective_rain = float(rain) * float(ant_factor) * float(river_factor) * float(dam_factor)
    ratio = effective_rain / float(threshold)
    return effective_rain, ratio, threshold


def classify_ratio(ratio: float) -> str:
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


def add_hazard_columns(df: pd.DataFrame, window="24h") -> pd.DataFrame:
    out = df.copy()
    computed = out.apply(lambda row: compute_effective_rain(row, window=window), axis=1, result_type="expand")
    out["effective_rain_mm"] = computed[0]
    out["hazard_ratio"] = computed[1]
    out["threshold_selected_mm"] = computed[2]
    out["selected_window"] = window
    out["hazard_level"] = out["hazard_ratio"].apply(classify_ratio)
    return out


def advisory_text(row: pd.Series) -> str:
    basin = row["basin_name"]
    level = row["hazard_level"]
    window = row["selected_window"]
    rain = row["effective_rain_mm"]
    threshold = row["threshold_selected_mm"]

    if level in ["Extreme", "Severe"]:
        return (
            f"{level} rainfall hazard for {basin} River Basin. Effective {window} rainfall is "
            f"{rain:.1f} mm versus threshold {threshold:.1f} mm. Possible flooding in low-lying and "
            "flood-prone communities. Coordinate with LGUs and monitor official advisories."
        )
    if level == "High":
        return (
            f"High rainfall hazard for {basin} River Basin. Effective {window} rainfall is {rain:.1f} mm. "
            "Localized flooding is possible in vulnerable areas."
        )
    if level == "Moderate":
        return f"Moderate rainfall hazard for {basin} River Basin. Continue monitoring rainfall and river conditions."
    if level == "No Data":
        return f"No forecast data available for {basin} River Basin at the moment."
    return f"Low rainfall hazard currently indicated for {basin} River Basin."


# -----------------------------
# MAPS
# -----------------------------
def merge_geojson_with_metrics(geojson_dict: dict, hazard_df: pd.DataFrame) -> dict:
    metric_map = hazard_df.set_index("basin_name").to_dict(orient="index")
    merged = json.loads(json.dumps(geojson_dict))
    for feat in merged.get("features", []):
        basin = feat.get("properties", {}).get("basin_name")
        metrics = metric_map.get(basin, {})
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                v = float(v)
            feat.setdefault("properties", {})[k] = v
    return merged


def style_function(feature):
    level = feature.get("properties", {}).get("hazard_level", "No Data")
    return {
        "fillColor": grade_color(level),
        "color": "#333333",
        "weight": 1.0,
        "fillOpacity": 0.65,
    }


def add_typhoon_overlay(m, track_df: pd.DataFrame | None):
    if track_df is None or track_df.empty:
        return

    clean = track_df.copy()
    clean["lat"] = pd.to_numeric(clean.get("lat"), errors="coerce")
    clean["lon"] = pd.to_numeric(clean.get("lon"), errors="coerce")
    clean = clean.dropna(subset=["lat", "lon"])
    if clean.empty:
        return

    points = clean[["lat", "lon"]].values.tolist()
    if len(points) >= 2:
        folium.PolyLine(points, weight=3, tooltip="Typhoon Track").add_to(m)

    for _, r in clean.iterrows():
        popup = (
            f"<b>{r.get('name', 'Typhoon')}</b><br>"
            f"{r.get('datetime', '')}<br>"
            f"Wind: {r.get('wind_kph', '')} kph"
        )
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=5,
            weight=1,
            fill=True,
            fill_opacity=0.9,
            popup=popup,
        ).add_to(m)


def build_map(geojson_merged: dict, track_df: pd.DataFrame | None = None):
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

    for feat in geojson_merged.get("features", []):
        p = feat.get("properties", {})
        lat = p.get("lat")
        lon = p.get("lon")
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
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


def get_clicked_basin(map_state: dict | None):
    if not map_state:
        return None
    for key in ["last_active_drawing", "last_object_clicked_tooltip", "last_object_clicked_popup"]:
        value = map_state.get(key)
        if isinstance(value, dict):
            props = value.get("properties", {})
            if props.get("basin_name"):
                return props.get("basin_name")
    return None


# -----------------------------
# UI
# -----------------------------
st.title("🌧️ Philippine River Basin Rainfall Hazard Dashboard")
st.caption("Lightweight, Project NOAH-inspired basin rainfall hazard screening for the Philippines.")

with st.sidebar:
    st.header("Controls")
    app_mode = st.radio("Mode", ["Version 1.1", "Version 2"], index=1)
    data_mode = st.radio("Forecast source", ["Sample data", "Live Open-Meteo"], index=1)
    rain_window = st.radio("Hazard window", ["24h", "72h"], horizontal=True)
    auto_refresh = st.checkbox("Auto-refresh every 30 minutes", value=False)
    uploaded_geojson = st.file_uploader("Optional: upload your own basin GeoJSON", type=["geojson", "json"])

if auto_refresh:
    st.markdown(
        "<meta http-equiv='refresh' content='1800'>",
        unsafe_allow_html=True,
    )

geojson_data = normalize_geojson(uploaded_geojson)
basin_master = geojson_to_basin_df(geojson_data)

if basin_master.empty:
    st.error("No basin features were found in the GeoJSON. Please upload a valid basin GeoJSON.")
    st.stop()

last_updated = None
failures = []

if data_mode == "Sample data":
    basin_values = prepare_sample_data(load_sample_data(str(DEFAULT_SAMPLE)))
    last_updated = pd.Timestamp.now(tz=MANILA_TZ)
else:
    with st.spinner("Downloading live forecast data by basin centroid..."):
        basin_json = basin_master.to_json(orient="records")
        basin_values, last_updated, failures = get_live_forecast_for_basins(basin_json)

basin_data = safe_merge_master_with_values(basin_master, basin_values)

# Version 2 modifiers
use_typhoon_overlay = False
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

hazard_df = add_hazard_columns(basin_data, window=rain_window)
hazard_df["alert_text"] = hazard_df.apply(advisory_text, axis=1)

# Header metrics
m1, m2, m3 = st.columns(3)
m1.metric("Basins monitored", f"{len(hazard_df)}")
m2.metric("Data source", data_mode)
m3.metric("Last updated", last_updated.strftime("%Y-%m-%d %I:%M %p") if last_updated is not None else "-")

if failures and data_mode == "Live Open-Meteo":
    with st.expander("Show basins with live-data fetch issues"):
        st.write("Some basins could not be refreshed and were left as no-data.")
        st.write(failures)

critical = hazard_df[hazard_df["hazard_level"].isin(["Extreme", "Severe"])].sort_values(
    "hazard_ratio", ascending=False
)
if not critical.empty:
    names = ", ".join(critical["basin_name"].head(6).tolist())
    st.error(
        f"Alert: {len(critical)} basin(s) are currently at Severe/Extreme rainfall hazard in the selected {rain_window} view: {names}"
    )
else:
    st.success(f"No basin is currently at Severe/Extreme rainfall hazard in the selected {rain_window} view.")

left, right = st.columns([1.8, 1.0])

with left:
    merged_geojson = merge_geojson_with_metrics(geojson_data, hazard_df)
    folium_map = build_map(merged_geojson, typhoon_track_df if use_typhoon_overlay else None)
    map_state = st_folium(folium_map, width=None, height=680, returned_objects=["last_active_drawing"])

with right:
    st.subheader("Top 10 highest-risk basins")
    top10 = hazard_df.sort_values("hazard_ratio", ascending=False).head(10).copy()
    top10 = top10[
        ["basin_name", "region", "hazard_level", "effective_rain_mm", "threshold_selected_mm", "hazard_ratio"]
    ]
    top10["hazard_ratio"] = (top10["hazard_ratio"] * 100).round(0).astype("Int64").astype(str) + "%"
    st.dataframe(top10, use_container_width=True, hide_index=True)

    options = ["All"] + hazard_df.sort_values("basin_name")["basin_name"].tolist()
    selected = st.selectbox("Basin details", options, index=0)

    clicked_basin = get_clicked_basin(map_state)
    if clicked_basin:
        st.info(f"Clicked basin: {clicked_basin}")
        selected = clicked_basin

    if selected != "All":
        row = hazard_df.loc[hazard_df["basin_name"] == selected].iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("Hazard", row["hazard_level"])
        c2.metric("Threshold used", f"{row['threshold_selected_mm']:.1f} mm" if pd.notna(row["threshold_selected_mm"]) else "-")
        c1.metric("Forecast 24h", f"{row['forecast_rain_24h_mm']:.1f} mm" if pd.notna(row["forecast_rain_24h_mm"]) else "-")
        c2.metric("Forecast 72h", f"{row['forecast_rain_72h_mm']:.1f} mm" if pd.notna(row["forecast_rain_72h_mm"]) else "-")
        c1.metric("Antecedent 72h", f"{row['antecedent_rain_72h_mm']:.1f} mm" if pd.notna(row["antecedent_rain_72h_mm"]) else "-")
        c2.metric("Effective rain", f"{row['effective_rain_mm']:.1f} mm" if pd.notna(row["effective_rain_mm"]) else "-")
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
    hazard_df[show_cols].sort_values("effective_rain_mm", ascending=False, na_position="last"),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")
st.markdown(
    """
**Notes**
- Included basin polygons are simplified placeholders for demo use only.
- Live mode uses Open-Meteo basin-centroid precipitation as a lightweight forecast source.
- Replace the sample GeoJSON with official river basin boundaries for production.
- This is near-real-time rainfall hazard screening, not a full hydrologic-hydraulic flood model.
"""
)
