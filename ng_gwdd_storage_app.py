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
import numpy as np


# -----------------------------
# Helpers
# -----------------------------
def fetch_eia_storage_us_total(api_key):
    """
    Fetch latest US working gas in storage (BCF)
    Safely handles cases when only one data point is returned.
    """
    url = (
        "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
        f"?api_key={api_key}"
        "&frequency=weekly"
        "&data[0]=value"
        "&facets[series][]=NG.NW2_EPG0_SWO_R48_BCF"
        "&sort[0][column]=period"
        "&sort[0][direction]=desc"
        "&length=2"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()

    data = r.json()["response"]["data"]
    df = pd.DataFrame(data)

    if df.empty:
        return None, None

    latest = float(df.iloc[0]["value"])

    # 👇 SAFE CHECK
    if len(df) > 1:
        previous = float(df.iloc[1]["value"])
        weekly_change = latest - previous
    else:
        weekly_change = None  # EIA returned only one week

    return latest, weekly_change



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


def compute_weekly_change(storage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds weekly injection/withdrawal (difference) column.
    Positive = injection, Negative = withdrawal.
    """
    df = storage_df.copy().sort_values("date")
    df["weekly_change"] = df["value"].diff()
    return df


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

tabs = st.tabs(["GWDD Dashboard", "EIA Storage Dashboard", "Signals & Summary"])

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


# -----------------------------
# Signals & Summary (NEW)
# -----------------------------
def _linear_slope(x, y) -> float:
    """Simple least-squares slope. Returns 0 if not enough points."""
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 2:
            return 0.0
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)
    except Exception:
        return 0.0


def _compute_overall_weighted_gwdd(cities_df: pd.DataFrame, base_f: float, days: int) -> pd.DataFrame:
    """Reuses the same logic as GWDD tab to build overall weighted GWDD daily series."""
    overall_rows = []
    for _, row in cities_df.iterrows():
        try:
            cdf = fetch_open_meteo_daily(float(row["lat"]), float(row["lon"]), days=days)
            if cdf.empty:
                continue
            gwdd = compute_gwdd(cdf, base_f=base_f)
            for d, g in zip(gwdd["date"].tolist(), gwdd["gwdd"].tolist()):
                overall_rows.append((d, float(g) * float(row["weight"])))
        except Exception:
            continue

    if not overall_rows:
        return pd.DataFrame(columns=["date", "gwdd_overall_weighted"])

    overall = pd.DataFrame(overall_rows, columns=["date", "weighted_gwdd"])
    overall = overall.groupby("date", as_index=False)["weighted_gwdd"].sum()
    overall = overall.rename(columns={"weighted_gwdd": "gwdd_overall_weighted"}).sort_values("date")
    return overall


def _compute_ng_signal(overall_gwdd_df: pd.DataFrame, storage_df_with_change: pd.DataFrame) -> dict:
    """
    Simple heuristic signal:
      - GWDD slope over forecast window (higher GWDD => higher demand => bullish)
      - Latest EIA weekly change (negative = withdrawal = bullish, positive = injection = bearish)
      - Divergence alert when GWDD and storage change point opposite ways.
    """
    out = {
        "gwdd_slope": 0.0,
        "gwdd_trend": "Neutral",
        "latest_storage_bcf": None,
        "latest_weekly_change_bcf": None,
        "signal": "Neutral",
        "divergence": False,
        "notes": [],
    }

    # GWDD trend (use slope)
    if overall_gwdd_df is not None and not overall_gwdd_df.empty:
        y = overall_gwdd_df["gwdd_overall_weighted"].astype(float).to_numpy()
        x = np.arange(len(y), dtype=float)
        slope = _linear_slope(x, y)
        out["gwdd_slope"] = slope

        # thresholds (tuned to avoid noise)
        if slope > 0.15:
            out["gwdd_trend"] = "Rising (colder / higher demand)"
        elif slope < -0.15:
            out["gwdd_trend"] = "Falling (warmer / lower demand)"
        else:
            out["gwdd_trend"] = "Flat / Mixed"

    # Storage (latest)
    if storage_df_with_change is not None and not storage_df_with_change.empty:
        s = storage_df_with_change.sort_values("date")
        out["latest_storage_bcf"] = float(s["value"].iloc[-1])
        if "weekly_change" in s.columns and len(s) >= 2:
            out["latest_weekly_change_bcf"] = float(s["weekly_change"].iloc[-1])

    # Build bullish/bearish summary
    gwdd_up = out["gwdd_slope"] > 0.15
    gwdd_down = out["gwdd_slope"] < -0.15
    chg = out["latest_weekly_change_bcf"]

    if chg is None:
        # GWDD-only
        if gwdd_up:
            out["signal"] = "Bullish (GWDD rising)"
        elif gwdd_down:
            out["signal"] = "Bearish (GWDD falling)"
        else:
            out["signal"] = "Neutral"
    else:
        # Combined
        withdrawal = chg < 0
        injection = chg > 0

        if withdrawal and gwdd_up:
            out["signal"] = "Bullish (withdrawal + colder trend)"
        elif injection and gwdd_down:
            out["signal"] = "Bearish (injection + warmer trend)"
        elif withdrawal and gwdd_down:
            out["signal"] = "Mixed (withdrawal but GWDD falling)"
            out["divergence"] = True
        elif injection and gwdd_up:
            out["signal"] = "Mixed (injection but GWDD rising)"
            out["divergence"] = True
        else:
            out["signal"] = "Neutral"

    if out["divergence"]:
        out["notes"].append("GWDD trend and latest storage change are pointing opposite ways (divergence).")

    return out


def _project_next_week_storage(storage_df_with_change: pd.DataFrame, method: str = "avg4") -> dict:
    """
    Project next week's storage level using recent average weekly change.
    method: avg4 / avg8 / last
    """
    out = {
        "latest_date": None,
        "latest_storage_bcf": None,
        "projection_method": method,
        "projected_change_bcf": None,
        "projected_next_storage_bcf": None,
    }
    if storage_df_with_change is None or storage_df_with_change.empty:
        return out

    s = storage_df_with_change.sort_values("date").copy()
    out["latest_date"] = s["date"].iloc[-1]
    out["latest_storage_bcf"] = float(s["value"].iloc[-1])

    changes = s["weekly_change"].dropna().astype(float)
    if changes.empty:
        return out

    if method == "last":
        proj_change = float(changes.iloc[-1])
    elif method == "avg8":
        proj_change = float(changes.tail(8).mean())
    else:  # avg4
        proj_change = float(changes.tail(4).mean())

    out["projected_change_bcf"] = proj_change
    out["projected_next_storage_bcf"] = float(out["latest_storage_bcf"] + proj_change)
    return out


with tabs[2]:
    st.header("Signals & Summary")
    st.caption("Bullish/Bearish summary + GWDD vs EIA divergence alert + projected next-week storage level.")

    # Build fresh (cached) data needed for the signal
    overall = _compute_overall_weighted_gwdd(cities_df, base_f=base_f, days=days)

    storage_with_change = None
    if api_key and str(api_key).strip():
        try:
            stor_sig = fetch_eia_series(EIA_SERIES_TOTAL_R48, api_key=str(api_key).strip())
            storage_with_change = compute_weekly_change(stor_sig)
        except Exception as e:
            storage_with_change = None
            st.warning(f"EIA fetch failed for signals: {e}")

    signal = _compute_ng_signal(overall, storage_with_change)

    # Top signal box
    sig_text = signal["signal"]
    if sig_text.lower().startswith("bullish"):
        st.success(f"Signal: **{sig_text}**")
    elif sig_text.lower().startswith("bearish"):
        st.error(f"Signal: **{sig_text}**")
    else:
        st.info(f"Signal: **{sig_text}**")

    # Divergence alert
    if signal["divergence"]:
        st.warning("⚠️ **GWDD ↔ EIA Divergence Alert**: GWDD trend and the latest storage change are conflicting.")

    # Key numbers
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("GWDD Trend", signal["gwdd_trend"])
    with c2:
        if signal["latest_weekly_change_bcf"] is None:
            st.metric("Latest weekly change (BCF)", "N/A")
        else:
            st.metric("Latest weekly change (BCF)", f"{signal['latest_weekly_change_bcf']:.0f}")
    with c3:
        if signal["latest_storage_bcf"] is None:
            st.metric("Latest storage (BCF)", "N/A")
        else:
            st.metric("Latest storage (BCF)", f"{signal['latest_storage_bcf']:.0f}")

    if signal["notes"]:
        st.write("Notes:")
        for n in signal["notes"]:
            st.write(f"- {n}")

    st.divider()

    st.subheader("Projected storage level for the following week")
    if storage_with_change is None or storage_with_change.empty:
        st.info("Enter a valid EIA API key to enable storage projection.")
    else:
        method = st.selectbox("Projection method", ["avg4", "avg8", "last"], index=0)
        proj = _project_next_week_storage(storage_with_change, method=method)

        use_manual = st.checkbox("Manual override (enter expected weekly change)", value=False)
        if use_manual:
            proj_change = st.number_input(
                "Expected weekly change (BCF). Positive=injection, negative=withdrawal",
                value=float(proj["projected_change_bcf"] or 0.0),
                step=1.0
            )
            proj_change = float(proj_change)
        else:
            proj_change = float(proj["projected_change_bcf"] or 0.0)

        projected_next = float(proj["latest_storage_bcf"] + proj_change)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Latest storage (BCF)", f"{proj['latest_storage_bcf']:.0f}")
        with d2:
            st.metric("Projected change (BCF)", f"{proj_change:+.0f}")
        with d3:
            st.metric("Projected next-week storage (BCF)", f"{projected_next:.0f}")

        st.dataframe(
            pd.DataFrame(
                [{
                    "latest_date": proj["latest_date"],
                    "latest_storage_bcf": proj["latest_storage_bcf"],
                    "projection_method": ("manual" if use_manual else method),
                    "projected_change_bcf": proj_change,
                    "projected_next_storage_bcf": projected_next,
                }]
            ),
            use_container_width=True,
        )

    st.divider()

    st.subheader("Full summary (all-in-one)")
    summary_parts = []
    summary_parts.append(f"GWDD: {signal['gwdd_trend']} (slope {signal['gwdd_slope']:+.2f}/day)")
    if signal["latest_weekly_change_bcf"] is not None:
        chg = signal["latest_weekly_change_bcf"]
        summary_parts.append(f"EIA weekly change: {chg:+.0f} BCF")
    if signal["latest_storage_bcf"] is not None:
        summary_parts.append(f"Latest storage: {signal['latest_storage_bcf']:.0f} BCF")
    summary_parts.append(f"Signal: {signal['signal']}")
    if signal["divergence"]:
        summary_parts.append("ALERT: GWDD ↔ EIA divergence")

    st.write(" • ".join(summary_parts))

