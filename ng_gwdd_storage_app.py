# NG USA — GWDD (All Cities) + EIA Natural Gas Storage (Injection/Withdrawal)
# Streamlit app (Windows friendly). Save as: ng_gwdd_storage_app.py
# Run: streamlit run ng_gwdd_storage_app.py --server.port 8051

from __future__ import annotations

import os
import math
import datetime as dt
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Helpers
# -----------------------------
def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def c_to_f(c: float) -> float:
    return (c * 9.0 / 5.0) + 32.0


@st.cache_data(ttl=60 * 30, show_spinner=False)  # 30 minutes
def fetch_open_meteo_daily(lat: float, lon: float, days: int = 10) -> pd.DataFrame:
    """
    Fetch daily temperatures from Open-Meteo.
    We request daily mean temp in Celsius then convert to Fahrenheit for GWDD calc.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "forecast_days": int(days),
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    daily = js.get("daily", {})
    times = daily.get("time", [])
    temps_c = daily.get("temperature_2m_mean", [])
    if not times or not temps_c or len(times) != len(temps_c):
        raise ValueError("Open-Meteo returned empty daily temperature data.")

    df = pd.DataFrame({"date": pd.to_datetime(times), "temp_c": temps_c})
    df["temp_f"] = df["temp_c"].map(c_to_f)
    return df


def compute_gwdd(df_daily: pd.DataFrame, base_f: float = 65.0) -> pd.DataFrame:
    """
    GWDD = max(0, base_temp_F - mean_temp_F) for each day.
    """
    out = df_daily.copy()
    out["gwdd"] = (base_f - out["temp_f"]).clip(lower=0)
    return out[["date", "temp_f", "gwdd"]]


def safe_float(x):
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


# -----------------------------
# EIA Storage (via SeriesID API)
# -----------------------------
EIA_SERIES_TOTAL_R48 = "NG.NW2_EPG0_SWO_R48_BCF.W"  # Total Lower 48, Working gas in storage (BCF), weekly


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)  # 6 hours
def fetch_eia_series(series_id: str, api_key: str) -> pd.DataFrame:
    """
    Fetch EIA data using the SeriesID endpoint (EIA API v2).
    Endpoint format:
      https://api.eia.gov/v2/seriesid/{SERIES_ID}?api_key=YOUR_KEY

    Returns DataFrame with:
      date (datetime), value (float)
    """
    if not api_key:
        raise ValueError("Missing EIA API key. Put it in the sidebar (or set EIA_API_KEY env var).")

    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    r = requests.get(url, params={"api_key": api_key}, timeout=30)
    # If key is wrong, EIA often returns 403/400 with message
    r.raise_for_status()
    js = r.json()

    # Typical v2 shape:
    # { "response": { "data": [ {"period":"2026-01-03","value":1234}, ... ] } }
    resp = js.get("response") or {}
    data = resp.get("data")

    # Some edge cases (older shapes) might be:
    # { "series": [ {"data":[["20260103",1234], ... ] } ] }
    if isinstance(data, list):
        rows = []
        for row in data:
            period = row.get("period") or row.get("date")
            val = row.get("value")
            if period is None:
                continue
            # period might be YYYY-MM-DD or YYYYMMDD
            p = str(period)
            if len(p) == 8 and p.isdigit():
                d = dt.datetime.strptime(p, "%Y%m%d").date()
            else:
                d = pd.to_datetime(p).date()
            rows.append((d, safe_float(val)))
        df = pd.DataFrame(rows, columns=["date", "value"])
    else:
        series = js.get("series") or []
        rows = []
        for s in series:
            for p, v in s.get("data", []):
                p = str(p)
                if len(p) == 8 and p.isdigit():
                    d = dt.datetime.strptime(p, "%Y%m%d").date()
                else:
                    d = pd.to_datetime(p).date()
                rows.append((d, safe_float(v)))
        df = pd.DataFrame(rows, columns=["date", "value"])

    if df.empty:
        raise ValueError("EIA returned no data for this series. Check series id / API key.")

    df = df.dropna(subset=["value"]).drop_duplicates(subset=["date"]).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return df


def project_next_week_storage(stor: pd.DataFrame) -> dict:
    """
    Simple projection for next week's storage:
    - Recent trend: average of last 4 weekly changes
    - Seasonal: median weekly change for same ISO week in prior years (up to 5 years)
    Projection = 70% recent + 30% seasonal (if seasonal available)
    """
    if stor is None or stor.empty or "value" not in stor.columns:
        return {"ok": False, "reason": "No storage data"}

    s = stor.dropna(subset=["value"]).sort_values("date").copy()
    if len(s) < 3:
        return {"ok": False, "reason": "Not enough history"}

    # Ensure weekly_change exists
    if "weekly_change" not in s.columns:
        s["weekly_change"] = s["value"].diff()

    last_row = s.iloc[-1]
    last_date = pd.to_datetime(last_row["date"])
    last_storage = float(last_row["value"])

    # Recent avg (last 4 changes)
    recent = s["weekly_change"].dropna().tail(4)
    recent_avg = float(recent.mean()) if len(recent) else float("nan")

    # Seasonal median for next week's ISO week
    next_date = last_date + pd.Timedelta(days=7)
    iso_week = int(next_date.isocalendar().week)
    iso_year = int(next_date.isocalendar().year)

    s2 = s.copy()
    iso = s2["date"].dt.isocalendar()
    s2["iso_week"] = iso.week.astype(int)
    s2["iso_year"] = iso.year.astype(int)

    seasonal_pool = s2[(s2["iso_week"] == iso_week) & (s2["iso_year"] < iso_year)]
    # keep last ~5 years
    seasonal_pool = seasonal_pool.sort_values("date").tail(5 * 1)  # ~5 samples (weekly)
    seasonal_vals = seasonal_pool["weekly_change"].dropna()

    seasonal_median = float(seasonal_vals.median()) if len(seasonal_vals) else float("nan")

    # Blend
    if pd.isna(seasonal_median) and pd.isna(recent_avg):
        return {"ok": False, "reason": "Cannot compute projection"}

    if pd.isna(seasonal_median):
        proj_change = recent_avg
        method = "Recent 4-week avg"
    elif pd.isna(recent_avg):
        proj_change = seasonal_median
        method = "Seasonal median"
    else:
        proj_change = 0.7 * recent_avg + 0.3 * seasonal_median
        method = "70% recent + 30% seasonal"

    proj_storage = last_storage + proj_change
    direction = "Injection" if proj_change > 0 else "Withdrawal" if proj_change < 0 else "Flat"

    return {
        "ok": True,
        "last_date": last_date,
        "next_date": next_date,
        "last_storage": last_storage,
        "proj_change": float(proj_change),
        "proj_storage": float(proj_storage),
        "direction": direction,
        "method": method,
    }



# -----------------------------
# Default Cities (you can edit in UI)
# -----------------------------
DEFAULT_CITIES = pd.DataFrame(
    [
        ("New York, NY", 40.7128, -74.0060, 1.00),
        ("Chicago, IL", 41.8781, -87.6298, 0.95),
        ("Boston, MA", 42.3601, -71.0589, 0.70),
        ("Philadelphia, PA", 39.9526, -75.1652, 0.70),
        ("Washington, DC", 38.9072, -77.0369, 0.65),
        ("Atlanta, GA", 33.7490, -84.3880, 0.60),
        ("Detroit, MI", 42.3314, -83.0458, 0.60),
        ("Minneapolis, MN", 44.9778, -93.2650, 0.55),
        ("Denver, CO", 39.7392, -104.9903, 0.45),
        ("Dallas, TX", 32.7767, -96.7970, 0.50),
        ("Houston, TX", 29.7604, -95.3698, 0.55),
        ("Phoenix, AZ", 33.4484, -112.0740, 0.35),
        ("Los Angeles, CA", 34.0522, -118.2437, 0.45),
        ("San Francisco, CA", 37.7749, -122.4194, 0.35),
        ("Seattle, WA", 47.6062, -122.3321, 0.40),
        ("Miami, FL", 25.7617, -80.1918, 0.30),
    ],
    columns=["city", "lat", "lon", "weight"],
)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NG USA GWDD + Storage", layout="wide")

st.title("NG USA — GWDD (All Cities) + EIA Storage (Injection/Withdrawal)")

with st.sidebar:
    st.header("GWDD Settings")
    base_f = st.number_input("Base Temp (°F)", min_value=30.0, max_value=80.0, value=65.0, step=0.5)
    days = st.slider("Forecast Days", min_value=3, max_value=16, value=10, step=1)

    st.subheader("Cities (edit list)")
    cities_df = st.data_editor(
        DEFAULT_CITIES,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "city": st.column_config.TextColumn("City"),
            "lat": st.column_config.NumberColumn("Lat"),
            "lon": st.column_config.NumberColumn("Lon"),
            "weight": st.column_config.NumberColumn("Weight", help="Higher weight = more demand impact"),
        },
        key="cities_editor",
    )

    st.divider()
    st.header("EIA Storage Settings")
    api_key = st.text_input(
        "EIA API Key",
        value=os.getenv("EIA_API_KEY", ""),
        type="password",
        help="Get a key from EIA. Tip: you can set an environment variable EIA_API_KEY.",
    )

tabs = st.tabs(["GWDD Dashboard", "EIA Storage Dashboard"])

# -----------------------------
# Tab 1: GWDD
# -----------------------------
with tabs[0]:
    st.subheader("GWDD (City-wise + Overall Weighted)")

    if cities_df.empty:
        st.error("City list is empty. Add at least 1 city in the sidebar.")
    else:
        # Normalize weights (avoid divide-by-zero)
        w_sum = float(cities_df["weight"].fillna(0).sum())
        if w_sum <= 0:
            st.warning("All weights are 0. Setting equal weights.")
            cities_df["weight"] = 1.0
            w_sum = float(cities_df["weight"].sum())

        # Fetch and compute GWDD for each city
        city_frames: List[pd.DataFrame] = []
        overall_rows: Dict[pd.Timestamp, float] = {}

        with st.spinner("Fetching weather + calculating GWDD..."):
            for _, row in cities_df.iterrows():
                try:
                    cdf = fetch_open_meteo_daily(float(row["lat"]), float(row["lon"]), days=days)
                    gw = compute_gwdd(cdf, base_f=base_f)
                    gw["city"] = str(row["city"])
                    gw["weight"] = float(row["weight"])
                    city_frames.append(gw)
                except Exception as e:
                    st.warning(f"City failed: {row.get('city','?')} — {e}")

        if not city_frames:
            st.error("No city data fetched. Check internet and city lat/lon.")
        else:
            all_city = pd.concat(city_frames, ignore_index=True)

            # Overall weighted GWDD by date
            all_city["w_gwdd"] = all_city["gwdd"] * all_city["weight"]
            overall = (
                all_city.groupby("date", as_index=False)
                .agg(weighted_sum=("w_gwdd", "sum"), weight_sum=("weight", "sum"))
            )
            overall["GWDD_overall_weighted"] = overall["weighted_sum"] / overall["weight_sum"]
            overall = overall[["date", "GWDD_overall_weighted"]].sort_values("date")

            # Top metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Cities", int(all_city["city"].nunique()))
            c2.metric("Forecast Days", int(days))
            c3.metric("Base (°F)", f"{base_f:g}")

            st.markdown("### Overall Weighted GWDD (Daily)")
            st.dataframe(overall, use_container_width=True, hide_index=True)
            st.line_chart(overall.set_index("date")["GWDD_overall_weighted"])

            st.markdown("### City-wise GWDD (Daily)")
            pivot = all_city.pivot_table(index="date", columns="city", values="gwdd", aggfunc="mean").sort_index()
            st.dataframe(pivot, use_container_width=True)
            st.line_chart(pivot)

# -----------------------------
# Tab 2: EIA Storage
# -----------------------------
with tabs[1]:
    st.subheader("EIA Weekly Natural Gas Storage (BCF) + Weekly Injection/Withdrawal")

    try:
        stor = fetch_eia_series(EIA_SERIES_TOTAL_R48, api_key=api_key)
        stor = compute_weekly_change(stor)

        latest = stor.dropna(subset=["value"]).iloc[-1]
        prev = stor.dropna(subset=["value"]).iloc[-2] if len(stor) >= 2 else None

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Series", "Total Lower 48 (BCF)")
        m2.metric("Latest Storage (BCF)", f"{latest['value']:.0f}", help=str(latest["date"].date()))
        if prev is not None:
            delta = float(latest["value"] - prev["value"])
            m3.metric("Weekly Change (BCF)", f"{delta:+.0f}", help="Positive=injection, negative=withdrawal")
        else:
            m3.metric("Weekly Change (BCF)", "n/a")

        st.markdown("### Storage (BCF)")
        st.dataframe(stor.tail(104), use_container_width=True, hide_index=True)  # last ~2 years
        st.line_chart(stor.set_index("date")["value"])

        st.markdown("### Weekly Injection/Withdrawal (BCF)")
        ch = stor.dropna(subset=["weekly_change"]).set_index("date")["weekly_change"]
        st.line_chart(ch)

        st.caption(
            "Note: This uses the EIA SeriesID API. If you get 400/403, your API key is wrong or blocked. "
            "Keep your key private."
        )

    except Exception as e:
        st.error(f"EIA Storage fetch failed: {e}")
        st.info("Fix: enter your EIA API key in the sidebar. Also check your internet connection.")
