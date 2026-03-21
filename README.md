# Philippine River Basin Rainfall Hazard Dashboard (Streamlit)

A lightweight, Project NOAH-inspired Streamlit app for basin-level rainfall hazard screening in the Philippines.

## What's included
- **Version 1.1**
  - Philippines basin map with color gradient
  - click/select a major river basin
  - 24h / 72h hazard toggle
  - top 10 highest-risk basins
  - alert banner for basins exceeding thresholds
  - sample data mode for offline demos

- **Version 2**
  - optional live forecast fetch using **Open-Meteo**
  - antecedent rainfall effect
  - basin-specific thresholds
  - manual river stage and dam release factor
  - optional typhoon track overlay from uploaded CSV
  - generated alert messages for LGU / public use

## Important note
The included GeoJSON polygons are **simplified placeholders** for demonstration only.
For production use, replace `data/major_river_basins_simplified.geojson` with an official basin boundary shapefile/GeoJSON.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Upload the folder to GitHub.
2. Create a new Streamlit app from the repo.
3. Main file path: `app.py`

## Files
- `app.py` - main dashboard
- `data/sample_basin_rainfall.csv` - sample rainfall inputs
- `data/major_river_basins_simplified.geojson` - placeholder basin polygons
- `data/sample_typhoon_track.csv` - optional overlay demo

## Live-data note
Open-Meteo is used as a lightweight no-key forecast source for the demo app.
You can later swap this with PAGASA, your own model outputs, or a custom forecast API.

## Example typhoon CSV columns
```csv
name,datetime,lat,lon,wind_kph
Typhoon Demo,2026-03-21 00:00,11.2,127.5,95
Typhoon Demo,2026-03-21 12:00,12.0,126.8,105
```