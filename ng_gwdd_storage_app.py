# NG USA — GWDD + EIA Natural Gas Storage (Injection/Withdrawal)
# Streamlit app (Windows friendly). Save as: ng_gwdd_storage_app.py
# Run: streamlit run ng_gwdd_storage_app.py --server.port 8051

from __future__ import annotations

import os
import math
import json
from pathlib import Path
import datetime as dt
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional RSS news feeds
try:
    import feedparser
except Exception:
    feedparser = None

# Optional market data (front-month NG)
try:
    import yfinance as yf
except Exception:
    yf = None


# -----------------------------
# Fast HTTP (connection reuse)
# -----------------------------
_HTTP = requests.Session()


def _http_get(url: str, params=None, timeout: int = 10):
    """Small wrapper for faster, re-used HTTP connections."""
    return _HTTP.get(url, params=params, timeout=timeout)


@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_yahoo_last_price(ticker: str) -> float | None:
    """Fetch last available close price for a Yahoo Finance ticker using yfinance.
    Returns None if yfinance not installed or no data."""
    if yf is None:
        return None
    try:
        df = yf.download(ticker, period="7d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        # prefer Close
        col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else df.columns[-1])
        val = df[col].dropna().iloc[-1]
        return float(val)
    except Exception:
        return None



# -----------------------------
# NG price projection (simple model) — next 5 business days
# -----------------------------
@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_ng_history_daily(ticker: str = "NG=F", period: str = "3mo") -> pd.DataFrame:
    """Fetch daily close history for NG using yfinance (if available).
    Returns DataFrame with columns: date, close."""
    if yf is None:
        return pd.DataFrame(columns=["date", "close"])
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "close"])
        close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        out = close.dropna().reset_index()
        out.columns = ["date", "close"]
        out["date"] = pd.to_datetime(out["date"])
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        return out.dropna(subset=["close"]).sort_values("date")
    except Exception:
        return pd.DataFrame(columns=["date", "close"])


def project_ng_prices_next_5_days(hist: pd.DataFrame) -> pd.DataFrame:
    """Very simple projection (NOT a guaranteed forecast):
    - uses last ~14 trading days slope (linear trend) on daily closes
    - builds +/- band using recent daily return volatility.
    Output: date, projected, low, high"""
    if hist is None or hist.empty or "close" not in hist.columns:
        return pd.DataFrame(columns=["date", "projected", "low", "high"])
    h = hist.dropna(subset=["close"]).copy().sort_values("date")
    if len(h) < 10:
        return pd.DataFrame(columns=["date", "projected", "low", "high"])

    # Trend slope from last 14 points (or available)
    n = int(min(14, len(h)))
    recent = h.tail(n)
    y = recent["close"].to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0
    last = float(y[-1])

    # Vol band from last 20 daily returns
    r = h["close"].pct_change().dropna()
    vol = float(r.tail(20).std()) if len(r) >= 5 else 0.02  # fallback 2%

    # Next 5 business days
    start = pd.Timestamp(h["date"].iloc[-1]).normalize()
    dates = pd.bdate_range(start + pd.Timedelta(days=1), periods=5)

    rows = []
    for i, d in enumerate(dates, start=1):
        proj = last + slope * i
        band = abs(last) * vol * np.sqrt(i)
        rows.append({"date": d, "projected": proj, "low": proj - band, "high": proj + band})

    return pd.DataFrame(rows)



# -----------------------------
# Helpers
# -----------------------------

# --- EIA forecast storage (for correct Surprise: Actual - Forecast for the SAME report week)
_FORECAST_STORE = Path("eia_forecasts.json")

def _load_eia_forecasts() -> dict:
    try:
        if _FORECAST_STORE.exists():
            return json.loads(_FORECAST_STORE.read_text())
    except Exception:
        pass
    return {}

def _save_eia_forecasts(data: dict) -> None:
    try:
        _FORECAST_STORE.write_text(json.dumps(data, indent=2))
    except Exception:
        # If write fails (read-only env), fall back to session_state
        st.session_state["_eia_forecasts_fallback"] = data

def _get_saved_forecast(report_date: dt.date) -> float | None:
    key = report_date.isoformat()
    data = _load_eia_forecasts()
    if key in data:
        try:
            return float(data[key])
        except Exception:
            return None
    # fallback
    fb = st.session_state.get("_eia_forecasts_fallback", {})
    if isinstance(fb, dict) and key in fb:
        try:
            return float(fb[key])
        except Exception:
            return None
    return None

def _set_saved_forecast(report_date: dt.date, forecast_bcf: float) -> None:
    key = report_date.isoformat()
    data = _load_eia_forecasts()
    data[key] = float(forecast_bcf)
    _save_eia_forecasts(data)
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

    r = _http_get(url, timeout=10)
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
    Fast + resilient: short timeout, silent failures -> empty DataFrame (keeps app snappy).
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "forecast_days": int(days),
        "timezone": "UTC",
    }

    try:
        r = _http_get(url, params=params, timeout=8)
        r.raise_for_status()
        js = r.json()

        daily = js.get("daily", {})
        times = daily.get("time", [])
        temps_c = daily.get("temperature_2m_mean", [])
        if not times or not temps_c or len(times) != len(temps_c):
            return pd.DataFrame(columns=["date", "temp_c", "temp_f"])

        df = pd.DataFrame({"date": pd.to_datetime(times), "temp_c": temps_c})
        df["temp_f"] = df["temp_c"].map(c_to_f)
        return df
    except Exception:
        # Silent fail: keeps the UI fast when Open-Meteo has transient SSL/network issues
        return pd.DataFrame(columns=["date", "temp_c", "temp_f"])


@st.cache_data(ttl=60 * 30, show_spinner=False)  # 30 minutes
def fetch_open_meteo_daily_model(lat: float, lon: float, days: int = 14, model: str | None = None) -> pd.DataFrame:
    """
    Fetch daily mean temperature from Open-Meteo with an optional model selector.
    Fast + resilient: short timeout, silent failures -> empty DataFrame.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "forecast_days": int(days),
        "timezone": "UTC",
    }
    if model:
        params["models"] = model

    try:
        r = _http_get(url, params=params, timeout=8)
        r.raise_for_status()
        js = r.json()

        daily = js.get("daily", {})
        times = daily.get("time", [])
        temps_c = daily.get("temperature_2m_mean", [])
        if not times or not temps_c or len(times) != len(temps_c):
            return pd.DataFrame(columns=["date", "temp_c", "temp_f"])

        df = pd.DataFrame({"date": pd.to_datetime(times), "temp_c": temps_c})
        df["temp_f"] = df["temp_c"].map(c_to_f)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "temp_c", "temp_f"])



@st.cache_data(ttl=60 * 30, show_spinner=False)  # 30 minutes
def fetch_open_meteo_daily_history(lat: float, lon: float, past_days: int = 7, forecast_days: int = 10) -> pd.DataFrame:
    """
    Fetch daily mean temperatures from Open-Meteo including recent ACTUALS (past_days) and FORECAST (forecast_days).
    Returns columns: date, temp_c, temp_f, kind ('Actual'/'Forecast')
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "past_days": int(past_days),
        "forecast_days": int(forecast_days),
        "timezone": "UTC",
    }

    try:
        r = _http_get(url, params=params, timeout=8)
        r.raise_for_status()
        js = r.json()

        daily = js.get("daily", {})
        times = daily.get("time", [])
        temps_c = daily.get("temperature_2m_mean", [])
        if not times or not temps_c or len(times) != len(temps_c):
            return pd.DataFrame(columns=["date", "temp_c", "temp_f", "kind"])

        df = pd.DataFrame({"date": pd.to_datetime(times), "temp_c": temps_c})
        df["temp_f"] = df["temp_c"].map(c_to_f)

        # Split Actual vs Forecast using today's UTC date (Open-Meteo returns UTC daily)
        today_utc = pd.Timestamp.utcnow().normalize()
        df["kind"] = np.where(df["date"] < today_utc, "Actual", "Forecast")
        return df[["date", "temp_c", "temp_f", "kind"]]
    except Exception:
        return pd.DataFrame(columns=["date", "temp_c", "temp_f", "kind"])

def _try_models_for_city(lat: float, lon: float, days: int, model_candidates: list[str], label: str) -> pd.DataFrame:
    """
    Try a list of Open-Meteo model codes and return the first that works.
    Resilient behavior:
      - If a model returns empty data, try the next model.
      - If all models fail/empty, fall back to default selection.
      - If still empty, return an empty DataFrame (do NOT raise) to keep UI fast.
    """
    for mc in model_candidates:
        try:
            df = fetch_open_meteo_daily_model(lat, lon, days=days, model=mc)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue

    # Fallback (no model param)
    try:
        df = fetch_open_meteo_daily_model(lat, lon, days=days, model=None)
        return df if df is not None else pd.DataFrame(columns=["date", "temp_c", "temp_f"])
    except Exception:
        return pd.DataFrame(columns=["date", "temp_c", "temp_f"])

def build_model_weighted_gwdd(cities_df: pd.DataFrame, base_f: float, days: int, model_key: str) -> pd.DataFrame:
    """
    Build gas-weighted GWDD outlook (day 1-14) for a given model using the existing city weights.
    Returns: date, gwdd
    """
    # Model code candidates (Open-Meteo model names can vary; we try a few common ones)
    model_map = {
        "gfs_op": ["gfs_seamless", "gfs"],
        "gfs_ens": ["gfs_ensemble", "gefs"],
        "ecmwf_ens": ["ecmwf_ensemble", "ecmwf"],
    }
    candidates = model_map.get(model_key, [])
    rows = []
    for _, row in cities_df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        w = float(row.get("weight", 1.0))
        df_t = _try_models_for_city(lat, lon, days=days, model_candidates=candidates, label=model_key)
        gw = compute_gwdd(df_t, base_f=base_f)
        gw["w_gwdd"] = gw["gwdd"] * w
        rows.append(gw[["date", "w_gwdd"]])
    if not rows:
        return pd.DataFrame(columns=["date", "gwdd"])
    out = pd.concat(rows, ignore_index=True).groupby("date", as_index=False)["w_gwdd"].sum()
    out = out.rename(columns={"w_gwdd": "gwdd"}).sort_values("date")
    return out



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
# Front-month Natural Gas price (Yahoo: NG=F)
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)  # 1 hour
def fetch_front_month_ng_history(start: str = "2020-01-01") -> pd.DataFrame:
    """Fetch continuous front-month NG futures (NG=F) daily history via yfinance.
    Returns columns: date, close
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    df = yf.download("NG=F", start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError("No data returned for NG=F (check internet / ticker availability).")

    # If yfinance returns MultiIndex columns, try to pull the Close field safely.
    close_df = None
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer Close level if available
        if "Close" in df.columns.get_level_values(0):
            close_df = df.xs("Close", axis=1, level=0, drop_level=False)
        elif "Adj Close" in df.columns.get_level_values(0):
            close_df = df.xs("Adj Close", axis=1, level=0, drop_level=False)

    if close_df is not None:
        # Take first column (single ticker) and name it close
        close_series = close_df.iloc[:, 0]
        out = close_series.to_frame(name="close").reset_index()
    else:
        # Single-level columns (common case)
        col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else df.columns[0])
        out = df[[col]].rename(columns={col: "close"}).reset_index()

    # Robustly name the date column (yfinance can return Date/Datetime/index)
    if "Date" in out.columns:
        out = out.rename(columns={"Date": "date"})
    elif "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "date"})
    elif "index" in out.columns:
        out = out.rename(columns={"index": "date"})
    else:
        out = out.rename(columns={out.columns[0]: "date"})

    # If duplicate columns create a DataFrame for 'date'/'close', pick first.
    if "date" in out.columns and isinstance(out["date"], pd.DataFrame):
        out["date"] = out["date"].iloc[:, 0]
    if "close" in out.columns and isinstance(out["close"], pd.DataFrame):
        out["close"] = out["close"].iloc[:, 0]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    # Drop timezone if present
    try:
        out["date"] = out["date"].dt.tz_localize(None)
    except Exception:
        pass

    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    return out


def build_yearly_seasonality(df: pd.DataFrame, year_start: int = 2020, year_end: int = 2026) -> pd.DataFrame:
    """Build a MM-DD indexed table with each year as a column (seasonality overlay)."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()

    # Minimal safety: ensure a usable "date" column exists (handles index/Date/Datetime/MultiIndex)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = ["_".join([str(x) for x in tup if x not in (None, "")]).strip() for tup in d.columns]
    if "date" not in d.columns:
        if "Date" in d.columns:
            d = d.rename(columns={"Date": "date"})
        elif "Datetime" in d.columns:
            d = d.rename(columns={"Datetime": "date"})
        else:
            d = d.reset_index()
            if "date" not in d.columns:
                if "Date" in d.columns:
                    d = d.rename(columns={"Date": "date"})
                elif "Datetime" in d.columns:
                    d = d.rename(columns={"Datetime": "date"})
                elif "index" in d.columns:
                    d = d.rename(columns={"index": "date"})
                else:
                    d = d.rename(columns={d.columns[0]: "date"})
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])

    d["year"] = d["date"].dt.year
    d = d[(d["year"] >= year_start) & (d["year"] <= year_end)]
    if d.empty:
        return pd.DataFrame()
    d["mmdd"] = d["date"].dt.strftime("%m-%d")
    piv = d.pivot_table(index="mmdd", columns="year", values="close", aggfunc="last")
    # Sort mmdd in calendar order
    piv = piv.reindex(sorted(piv.index), axis=0)
    piv = piv.sort_index()
    return piv

# -----------------------------
# EIA Storage (via SeriesID API)
# -----------------------------
EIA_SERIES_TOTAL_R48 = "NG.NW2_EPG0_SWO_R48_BCF.W"  # Total Lower 48, Working gas in storage (BCF), weekly


# -----------------------------
# NOAA 8–14 Day Outlook (IMAGE ONLY — SAFE)
# -----------------------------
NOAA_814_TEMP_MAP = "https://www.cpc.ncep.noaa.gov/products/predictions/814day/814temp.new.gif"
NOAA_814_DISCUSSION = "https://www.cpc.ncep.noaa.gov/products/predictions/814day/fxus06.html"



@st.cache_data(ttl=60 * 15, show_spinner=False)  # 15 min (faster updates after EIA release)
def fetch_eia_series(series_id: str, api_key: str) -> pd.DataFrame:
    """
    Fetch EIA data for a single Series ID.
    Tries multiple endpoints for compatibility (API v2 SeriesID + legacy v1 fallback).
    Returns DataFrame with:
      date (datetime), value (float)
    """
    if not api_key or not str(api_key).strip():
        raise ValueError("Missing EIA API key. Put it in the sidebar (or set EIA_API_KEY env var).")

    series_id = str(series_id).strip()
    api_key = str(api_key).strip()

    # Endpoints to try (EIA has multiple compatible patterns over time)
    attempts = [
        ("v2_path", f"https://api.eia.gov/v2/seriesid/{series_id}", {"api_key": api_key}),
        ("v2_query", "https://api.eia.gov/v2/seriesid/", {"api_key": api_key, "series_id": series_id}),
        ("v2_query_noslash", "https://api.eia.gov/v2/seriesid", {"api_key": api_key, "series_id": series_id}),
        # Legacy v1 fallback (may still work for some keys/series)
        ("v1_legacy", "https://api.eia.gov/series/", {"api_key": api_key, "series_id": series_id}),
    ]

    last_err: Exception | None = None
    js = None

    for _name, url, params in attempts:
        try:
            r = _http_get(url, params=params, timeout=30)
            r.raise_for_status()
            js = r.json()
            # Basic sanity: must have some expected keys
            if isinstance(js, dict) and ("response" in js or "series" in js):
                break
        except Exception as e:
            last_err = e
            js = None

    if js is None:
        raise RuntimeError(f"EIA request failed for series '{series_id}'. Last error: {last_err}")

    data_rows = None

    # API v2 expected shape:
    # { "response": { "data": [ {"period":"2026-01-03","value":"1234"}, ... ] } }
    if isinstance(js, dict) and "response" in js and isinstance(js["response"], dict):
        resp = js["response"]
        if "data" in resp and isinstance(resp["data"], list):
            data_rows = resp["data"]

    # Legacy v1 expected shape:
    # { "series": [ { "data": [ ["20260103", 1234], ... ] } ] }
    if data_rows is None and isinstance(js, dict) and "series" in js and isinstance(js["series"], list) and js["series"]:
        s0 = js["series"][0]
        if isinstance(s0, dict) and "data" in s0 and isinstance(s0["data"], list):
            # Convert v1 format to v2-like rows
            converted = []
            for item in s0["data"]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    converted.append({"period": item[0], "value": item[1]})
            data_rows = converted

    if not data_rows:
        raise RuntimeError(f"No data returned from EIA for series '{series_id}'. Check your API key and series id.")

    df = pd.DataFrame(data_rows)

    # Normalize column names
    if "period" in df.columns:
        df = df.rename(columns={"period": "date"})
    if "value" not in df.columns:
        # Some routes may use "data" or other naming - fail clearly
        raise RuntimeError(f"Unexpected EIA response columns for '{series_id}': {list(df.columns)}")

    # Parse date formats (weekly often 'YYYY-MM-DD', legacy sometimes 'YYYYMMDD')
    def _parse_date(x: str) -> dt.datetime:
        x = str(x)
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y"):
            try:
                return dt.datetime.strptime(x, fmt)
            except Exception:
                continue
        # last resort: pandas
        return pd.to_datetime(x, errors="coerce").to_pydatetime()

    df["date"] = df["date"].apply(_parse_date)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
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
# EIA report forecast (date + actual auto-update)
# -----------------------------
@st.cache_data(ttl=60 * 5, show_spinner=False)  # refresh fast near release time
def fetch_eia_latest_storage_and_change(api_key: str) -> dict:
    """Returns latest period date, storage level and the latest weekly change (actual)."""
    stor = fetch_eia_series(EIA_SERIES_TOTAL_R48, api_key=str(api_key).strip())
    stor = compute_weekly_change(stor)
    stor = stor.dropna(subset=["value"]).sort_values("date")
    latest = stor.iloc[-1]
    prev = stor.iloc[-2] if len(stor) >= 2 else None
    actual_change = float(latest["value"] - prev["value"]) if prev is not None else None
    return {
        "latest_period_date": latest["date"].date(),
        "latest_storage_bcf": float(latest["value"]),
        "actual_weekly_change_bcf": actual_change,
    }

def estimate_next_eia_report_date(today: dt.date | None = None) -> dt.date:
    """EIA NG storage report is typically Thursday.
    This returns the next Thursday date (local calendar)."""
    if today is None:
        today = dt.date.today()
    THU = 3  # Monday=0
    days_ahead = (THU - today.weekday()) % 7
    # If today is Thursday, treat it as 'next' (user can refresh after release).
    return today + dt.timedelta(days=days_ahead)


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
# Global demand regions (US NG burn + key LNG import markets)
# You can edit weights/locations inside the GWDD tab.
# -----------------------------
GLOBAL_DEMAND_REGIONS = pd.DataFrame(
    [
        ("US Northeast (NYC)", 40.7128, -74.0060, 1.00),
        ("US Midwest (Chicago)", 41.8781, -87.6298, 0.90),
        ("US South Central (Dallas)", 32.7767, -96.7970, 0.85),
        ("US Southeast (Atlanta)", 33.7490, -84.3880, 0.75),
        ("US West (Los Angeles)", 34.0522, -118.2437, 0.55),
        ("Mexico (Monterrey)", 25.6866, -100.3161, 0.70),
        ("UK (London)", 51.5074, -0.1278, 0.55),
        ("Netherlands (Amsterdam)", 52.3676, 4.9041, 0.50),
        ("Germany (Berlin)", 52.5200, 13.4050, 0.55),
        ("Japan (Tokyo)", 35.6762, 139.6503, 0.55),
        ("South Korea (Seoul)", 37.5665, 126.9780, 0.50),
        ("China (Shanghai)", 31.2304, 121.4737, 0.50),
    ],
    columns=["region", "lat", "lon", "weight"],
)



# -----------------------------
# Global NG use countries (representative demand centers)
# (Used for Global GWDD daily table + signal)
# -----------------------------
GLOBAL_NG_COUNTRIES = pd.DataFrame(
    [
        ("United States", "Chicago, IL", 41.8781, -87.6298, 1.00),
        ("Canada", "Toronto, ON", 43.6532, -79.3832, 0.35),
        ("Mexico", "Monterrey", 25.6866, -100.3161, 0.25),
        ("United Kingdom", "London", 51.5074, -0.1278, 0.30),
        ("Germany", "Berlin", 52.5200, 13.4050, 0.30),
        ("France", "Paris", 48.8566, 2.3522, 0.25),
        ("Italy", "Milan", 45.4642, 9.1900, 0.22),
        ("Spain", "Madrid", 40.4168, -3.7038, 0.18),
        ("Netherlands", "Amsterdam", 52.3676, 4.9041, 0.18),
        ("Japan", "Tokyo", 35.6762, 139.6503, 0.28),
        ("South Korea", "Seoul", 37.5665, 126.9780, 0.20),
        ("China", "Shanghai", 31.2304, 121.4737, 0.30),
        ("India", "Delhi", 28.6139, 77.2090, 0.20),
    ],
    columns=["country", "city", "lat", "lon", "weight"],
)



# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="NG USA GWDD + Storage", layout="wide")
    # Show loading overlay only on FIRST app open (not on reruns / auto-refresh / tab clicks)
    if "app_loaded" not in st.session_state:
        st.session_state["app_loaded"] = False

    # --- Netflix-style full-screen loading overlay (removed automatically when app finishes rendering) ---
    _loader = st.empty()
    if not st.session_state["app_loaded"]:
        _loader.markdown(
            """
            <style>
            #netflix-loader {
                position: fixed;
                inset: 0;
                background: #000;
                z-index: 999999;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }
            #netflix-loader .ring {
                width: 64px;
                height: 64px;
                border: 6px solid #333;
                border-top-color: #fff;
                border-radius: 50%;
                animation: spin 0.9s linear infinite;
                margin-bottom: 18px;
            }
            #netflix-loader .title {
                color: #fff;
                font-size: 28px;
                font-weight: 700;
                text-align: center;
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
            }
            #netflix-loader .sub {
                color: rgba(255,255,255,0.55);
                font-size: 13px;
                text-align: center;
                margin-top: 8px;
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
            </style>
    
            <div id="netflix-loader">
              <div class="ring"></div>
              <div class="title">Please wait… we pull fresh data for you</div>
              <div class="sub">Loading live weather, storage, and signals</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        _loader.empty()


    st.title("NG USA — GWDD (All Cities) + EIA Storage (Injection/Withdrawal)")

    with st.sidebar:
        st.header("GWDD Settings")
        base_c = st.number_input("Base Temp (°C)", min_value=-5.0, max_value=30.0, value=18.0, step=0.5)
        base_f = c_to_f(base_c)  # internal (GWDD uses Fahrenheit for GWDD math)
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
        # Ensure required columns exist even if user edits/removes them in the UI
        # (Prevents US GWHDD / GWDD calculations from silently returning empty.)
        if 'lon' not in cities_df.columns or 'lat' not in cities_df.columns:
            cities_df = cities_df.merge(DEFAULT_CITIES[['city','lat','lon']], on='city', how='left', suffixes=('', '_default'))
            if 'lat_default' in cities_df.columns:
                cities_df['lat'] = cities_df['lat'].fillna(cities_df['lat_default'])
                cities_df = cities_df.drop(columns=['lat_default'])
            if 'lon_default' in cities_df.columns:
                cities_df['lon'] = cities_df['lon'].fillna(cities_df['lon_default'])
                cities_df = cities_df.drop(columns=['lon_default'])
        if 'weight' not in cities_df.columns:
            cities_df['weight'] = 1.0
        cities_df['weight'] = pd.to_numeric(cities_df['weight'], errors='coerce').fillna(1.0)


        st.divider()
        st.header("EIA Storage Settings")
        api_key = st.text_input(
            "EIA API Key",
            value=st.secrets.get("EIA_API_KEY", os.getenv("EIA_API_KEY", "")),
            type="password",
            help="Get a key from EIA. Tip: you can set an environment variable EIA_API_KEY.",
        )

        # Keep key in session_state so other tabs (Drivers) can read it reliably
        st.session_state["EIA_API_KEY"] = str(api_key).strip() if api_key is not None else ""

        # Market expectation for the *next* EIA storage report (manual entry).
        # Negative = expected withdrawal, Positive = expected injection.
        eia_market_forecast_bcf = st.number_input(
            "Next EIA report market forecast (BCF)",
            value=-107.0,
            step=1.0,
            help="Enter the market/analyst estimate for the upcoming EIA storage change. Example: -107 = expected 107 Bcf withdrawal.",
        )
    auto_refresh = st.checkbox("Auto refresh every 1 minute", value=False)
    if auto_refresh:
        # Browser meta-refresh (no extra packages needed)
        st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)


    # -----------------------------
    # Contracts & Rollover helpers
    # -----------------------------
    _MONTH_CODE = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}

    def _observed_date(d: dt.date) -> dt.date:
        # If holiday falls on Saturday -> observed Friday; Sunday -> observed Monday
        if d.weekday() == 5:
            return d - dt.timedelta(days=1)
        if d.weekday() == 6:
            return d + dt.timedelta(days=1)
        return d

    def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> dt.date:
        # weekday: Monday=0 ... Sunday=6
        first = dt.date(year, month, 1)
        shift = (weekday - first.weekday()) % 7
        day = 1 + shift + (n - 1) * 7
        return dt.date(year, month, day)

    def _last_weekday_of_month(year: int, month: int, weekday: int) -> dt.date:
        # weekday: Monday=0 ... Sunday=6
        if month == 12:
            next_month = dt.date(year + 1, 1, 1)
        else:
            next_month = dt.date(year, month + 1, 1)
        last_day = next_month - dt.timedelta(days=1)
        shift = (last_day.weekday() - weekday) % 7
        return last_day - dt.timedelta(days=shift)

    def _easter_sunday(year: int) -> dt.date:
        # Anonymous Gregorian algorithm
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return dt.date(year, month, day)

    def _cme_like_holidays(year: int) -> set[dt.date]:
        """Approximate holiday set for NG futures (good enough for rollover estimates).
        Includes major US holidays + Good Friday. (Early closes not modeled.)
        """
        holidays = set()

        # Fixed-date holidays (observed)
        holidays.add(_observed_date(dt.date(year, 1, 1)))   # New Year's Day
        holidays.add(_observed_date(dt.date(year, 6, 19)))  # Juneteenth
        holidays.add(_observed_date(dt.date(year, 7, 4)))   # Independence Day
        holidays.add(_observed_date(dt.date(year, 12, 25))) # Christmas

        # Monday holidays
        holidays.add(_nth_weekday_of_month(year, 1, 0, 3))  # MLK Day (3rd Mon Jan)
        holidays.add(_nth_weekday_of_month(year, 2, 0, 3))  # Presidents' Day (3rd Mon Feb)
        holidays.add(_last_weekday_of_month(year, 5, 0))    # Memorial Day (last Mon May)
        holidays.add(_nth_weekday_of_month(year, 9, 0, 1))  # Labor Day (1st Mon Sep)

        # Thanksgiving (4th Thu Nov)
        holidays.add(_nth_weekday_of_month(year, 11, 3, 4))

        # Good Friday
        holidays.add(_easter_sunday(year) - dt.timedelta(days=2))

        return holidays

    def _holiday_set_for_years(years: List[int]) -> set[dt.date]:
        s: set[dt.date] = set()
        for y in years:
            s |= _cme_like_holidays(y)
        return s

    def _is_business_day(d: "dt.date", holiday_set: set[dt.date]) -> bool:
        return (d.weekday() < 5) and (d not in holiday_set)

    def _add_business_days(d: "dt.date", n: int) -> "dt.date":
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        cur = d
        while remaining > 0:
            cur = cur + dt.timedelta(days=step)
            if _is_business_day(cur, holiday_set):
                remaining -= 1
        return cur

    def ng_contract_expiry(delivery_year: int, delivery_month: int) -> "dt.date":
        # Rule of thumb (NYMEX Henry Hub NG): trading terminates 3 business days prior to
        # the first calendar day of the delivery month (weekend-adjusted only here).
        first_day = dt.date(delivery_year, delivery_month, 1)
        # Go back 3 business days from first_day
        cur = first_day
        count = 0
        while count < 3:
            cur = cur - dt.timedelta(days=1)
            if _is_business_day(cur, holiday_set):
                count += 1
        return cur

    def ng_symbol(delivery_year: int, delivery_month: int) -> str:
        yy = str(delivery_year)[-2:]
        return f"NG{_MONTH_CODE[delivery_month]}{yy}"

    def front_delivery_month(today: "dt.date") -> tuple[int,int]:
        # Choose the nearest delivery month that has not expired yet.
        y, m = today.year, today.month
        exp = ng_contract_expiry(y, m)
        if today <= exp:
            return y, m
        # else next month
        if m == 12:
            return y+1, 1
        return y, m+1


    # ==============================
    # Extra Drivers Modules (Tab 1)
    # LNG exports (terminal), Freeze-off risk, Backwardation, Power burn
    # ==============================

    @st.cache_data(ttl=60*30, show_spinner=False)
    def _fetch_eia_dnav_table(url: str):
        """Fetches EIA dnav HTML table and returns a tidy monthly time series.
        Returns DataFrame with columns: period (Timestamp), value (float in MMCF)
        """
        import pandas as _pd
        import requests as _requests

        r = _requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        tables = _pd.read_html(r.text)
        if not tables:
            return _pd.DataFrame(columns=["period", "value"])
        df = tables[0].copy()
        # Expected: first column Year, then Jan..Dec
        if df.shape[1] < 2:
            return _pd.DataFrame(columns=["period", "value"])
        df.columns = [str(c).strip() for c in df.columns]
        year_col = df.columns[0]
        month_cols = df.columns[1:]
        out_rows = []
        for _, row in df.iterrows():
            year = row.get(year_col)
            try:
                year_int = int(str(year).strip())
            except Exception:
                continue
            for mname in month_cols:
                val = row.get(mname)
                if val in (None, "-", "--", "NA", "W"):
                    continue
                try:
                    val_f = float(str(val).replace(",", "").strip())
                except Exception:
                    continue
                # Parse month name (Jan, Feb, ...)
                try:
                    month_int = dt.datetime.strptime(mname[:3], "%b").month
                except Exception:
                    continue
                out_rows.append({"period": _pd.Timestamp(year_int, month_int, 1), "value": val_f})
        out = _pd.DataFrame(out_rows).sort_values("period")
        return out

    def _mmcfm_to_bcfd(month_start: "pd.Timestamp", mmcf: float) -> float:
        import pandas as _pd
        # mmcf per month -> bcf/d
        days = (_pd.Timestamp(month_start) + _pd.offsets.MonthEnd(1)).day
        return (mmcf / 1000.0) / float(days)

    @st.cache_data(ttl=60*10, show_spinner=False)
    def _open_meteo_daily_minmax(lat: float, lon: float, days: int = 7):
        import requests as _requests
        import pandas as _pd
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_min,temperature_2m_max"
            f"&forecast_days={days}"
            "&timezone=auto"
        )
        r = _requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        js = r.json()
        daily = js.get("daily", {}) if isinstance(js, dict) else {}
        dates = daily.get("time", [])
        tmin = daily.get("temperature_2m_min", [])
        tmax = daily.get("temperature_2m_max", [])
        df = _pd.DataFrame({"date": dates, "tmin_c": tmin, "tmax_c": tmax})
        if not df.empty:
            df["date"] = _pd.to_datetime(df["date"])
        return df

    def _safe_yf_last(ticker: str):
        try:
            import yfinance as _yf
            import pandas as _pd
            hist = _yf.download(ticker, period="7d", interval="1h", progress=False)
            if hist is None or hist.empty:
                hist = _yf.download(ticker, period="30d", interval="1d", progress=False)
            if hist is None or hist.empty:
                return None
            last = hist["Close"].dropna().iloc[-1]
            return float(last)
        except Exception:
            return None

    def _guess_next_ng_ticker() -> str:
        """Best-effort guess for next-month NYMEX NG ticker on Yahoo Finance.
        Yahoo commonly uses 'NG{MonthCode}{YY}.NYM' (not guaranteed)."""
        month_codes = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}
        now = dt.datetime.utcnow()
        m = now.month + 1
        y = now.year
        if m == 13:
            m = 1
            y += 1
        code = month_codes.get(m, "F")
        yy = str(y)[-2:]
        return f"NG{code}{yy}.NYM"

    def _render_lng_tracker():
        st.subheader("LNG exports / feedgas proxy (fast signal)")
        st.caption("EIA terminal-level live feedgas is limited. This module uses a resilient proxy: tries terminal tables first, then falls back to EIA series trend so you always get a signal.")

        api_key = str(st.session_state.get("EIA_API_KEY", "") or "").strip()
        if not api_key:
            st.info("Enter your EIA API key in the sidebar to enable LNG export proxy signals.")
            return

        terminals = {
            "Freeport": "https://www.eia.gov/dnav/ng/ng_move_exp_lng_a_EPG0_TXM_mmcf_a.htm",
            "Sabine Pass": "https://www.eia.gov/dnav/ng/ng_move_exp_lng_a_EPG0_LA2_mmcf_a.htm",
            "Corpus Christi": "https://www.eia.gov/dnav/ng/ng_move_exp_lng_a_EPG0_TX2_mmcf_a.htm",
        }

        rows = []
        any_ok = False
        for name, url in terminals.items():
            try:
                ts = _fetch_eia_dnav_table(url)
                if ts is None or ts.empty:
                    rows.append({"Terminal": name, "Latest month": "N/A", "MMCF": None, "Approx Bcf/d": None})
                    continue
                any_ok = True
                last = ts.dropna().iloc[-1]
                period = str(last["date"])[:7] if "date" in last else "Latest"
                val = float(last["value"])
                bcfd = val / 1000.0 / 30.0  # rough conversion for display
                rows.append({"Terminal": name, "Latest month": period, "MMCF": round(val, 0), "Approx Bcf/d": round(bcfd, 2)})
            except Exception:
                rows.append({"Terminal": name, "Latest month": "N/A", "MMCF": None, "Approx Bcf/d": None})

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Fallback series (trend-based) so signal isn't stuck at N/A
        # Default series IDs are best-effort; user can override if needed.
        with st.expander("Advanced: series IDs (optional)", expanded=False):
            lng_series = st.text_input("EIA series id (LNG exports / feedgas proxy)", value="NG.N9133US2.M")
            pipe_series = st.text_input("EIA series id (Pipeline exports proxy)", value="NG.N9132US2.M")

        def _trend_label(vals: list[float]) -> tuple[str, float]:
            if len(vals) < 6:
                return "Neutral", 0.0
            a = float(np.mean(vals[-3:]))
            b = float(np.mean(vals[-6:-3]))
            if b == 0:
                return "Neutral", 0.0
            pct = (a - b) / abs(b)
            if pct >= 0.02:
                return "Bullish", pct
            if pct <= -0.02:
                return "Bearish", pct
            return "Neutral", pct

        try:
            lng_vals = []
            pipe_vals = []
            try:
                df_lng = fetch_eia_series(str(lng_series).strip(), api_key=api_key)
                lng_vals = df_lng["value"].dropna().astype(float).tolist()
            except Exception:
                lng_vals = []

            try:
                df_pipe = fetch_eia_series(str(pipe_series).strip(), api_key=api_key)
                pipe_vals = df_pipe["value"].dropna().astype(float).tolist()
            except Exception:
                pipe_vals = []

            sig_lng, pct_lng = _trend_label(lng_vals) if lng_vals else ("Neutral", 0.0)
            sig_pipe, pct_pipe = _trend_label(pipe_vals) if pipe_vals else ("Neutral", 0.0)

            score = 0
            score += 1 if sig_lng == "Bullish" else (-1 if sig_lng == "Bearish" else 0)
            score += 1 if sig_pipe == "Bullish" else (-1 if sig_pipe == "Bearish" else 0)

            if score >= 1:
                st.success(f"LNG / exports signal: Bullish (LNG trend {pct_lng*100:+.1f}%, pipeline trend {pct_pipe*100:+.1f}%)")
            elif score <= -1:
                st.error(f"LNG / exports signal: Bearish (LNG trend {pct_lng*100:+.1f}%, pipeline trend {pct_pipe*100:+.1f}%)")
            else:
                st.info(f"LNG / exports signal: Neutral (LNG trend {pct_lng*100:+.1f}%, pipeline trend {pct_pipe*100:+.1f}%)")
        except Exception:
            st.info("LNG / exports signal: Neutral (data fetch issue).")

    def _render_freezeoff_alert():
        st.subheader("Production freeze‑off risk (weather + supply)")
        st.caption("Weather-based freeze risk (proxy). Higher risk can reduce production and is usually bullish for NG. Confirmed risk zones are highlighted below.")

        # Key producing areas (approx)
        points = [
            ("Permian (Midland, TX)", 31.9973, -102.0779),
            ("Oklahoma (OKC, OK)", 35.4676, -97.5164),
            ("Haynesville (Shreveport, LA)", 32.5252, -93.7502),
            ("Appalachia (Pittsburgh, PA)", 40.4406, -79.9959),
        ]

        import pandas as _pd
        risk_rows = []
        worst_score = 0
        for name, lat, lon in points:
            try:
                df = _open_meteo_daily_minmax(lat, lon, days=7)
                if df.empty:
                    risk_rows.append({"Region": name, "Coldest next 7d (°C)": None, "Risk": "N/A"})
                    continue
                coldest = float(df["tmin_c"].min())
                if coldest <= -10:
                    risk = "High"
                    score = 1
                elif coldest <= -5:
                    risk = "Medium"
                    score = 0.5
                else:
                    risk = "Low"
                    score = 0
                worst_score = max(worst_score, score)
                risk_rows.append({"Region": name, "Coldest next 7d (°C)": round(coldest, 1), "Risk": risk})
            except Exception:
                risk_rows.append({"Region": name, "Coldest next 7d (°C)": None, "Risk": "N/A"})

        st.dataframe(_pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

        if worst_score >= 1:
            st.success("Freeze‑off alert: Bullish (High risk in at least one key region)")
        elif worst_score >= 0.5:
            st.info("Freeze‑off alert: Neutral‑to‑Bullish (Medium risk in at least one key region)")
        else:
            st.error("Freeze‑off alert: Bearish/Neutral (Low freeze risk)")

    def _render_backwardation_detector():
        st.subheader("Backwardation detector (front vs next contract)")
        st.caption("Backwardation = near contract price > next contract price (often supportive). Contango is the opposite.")

        # Auto-pick explicit contracts to avoid Yahoo mapping both to the same continuous price.
        try:
            y1, m1 = front_delivery_month(dt.date.today())
            if m1 == 12:
                y2, m2 = y1 + 1, 1
            else:
                y2, m2 = y1, m1 + 1
            auto_front = f"{ng_symbol(y1, m1)}.NYM"
            auto_next = f"{ng_symbol(y2, m2)}.NYM"
        except Exception:
            auto_front, auto_next = "NG=F", _guess_next_ng_ticker()

        colA, colB = st.columns(2)
        with colA:
            front = st.text_input("Front-month ticker (Yahoo Finance)", value=auto_front)
        with colB:
            nxt = st.text_input("Next-month ticker (Yahoo Finance)", value=auto_next)

        p_front = _safe_yf_last(front.strip()) if front.strip() else None
        p_next = _safe_yf_last(nxt.strip()) if nxt.strip() else None

        if p_front is None or p_next is None:
            st.warning("Could not fetch one of the tickers from Yahoo Finance. If you see N/A, try a different ticker format.")
            st.write(f"Front: **{front}** → {p_front if p_front is not None else 'N/A'}")
            st.write(f"Next: **{nxt}** → {p_next if p_next is not None else 'N/A'}")
            st.info("Tip: Try explicit contracts like NGF26.NYM / NGG26.NYM (front/next).")
            return

        spread = float(p_front) - float(p_next)
        st.write(f"Front: **{front}** = {p_front:.3f}")
        st.write(f"Next: **{nxt}** = {p_next:.3f}")
        st.write(f"Spread (Front - Next): **{spread:+.4f}**")

        # If Yahoo returns the same price for both, it can mean one ticker isn't resolving correctly.
        if abs(spread) < 1e-6:
            st.info("Curve signal: Neutral (Flat)")
            st.caption("Note: If spread is always 0.0000, Yahoo may be mapping one ticker incorrectly. Try explicit contracts (e.g., NGF26.NYM vs NGG26.NYM).")
            return

        # Small threshold to avoid noise / rounding
        thr = 0.03
        if spread >= thr:
            st.success("Curve signal: Bullish (Backwardation)")
        elif spread <= -thr:
            st.error("Curve signal: Bearish (Contango)")
        else:
            st.info("Curve signal: Neutral (Flat-ish)")
    def _eia_v2_get(url: str, params: dict):
        import requests as _requests
        r = _requests.get(url, params=params, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.json()

    def _render_power_burn_score(eia_key: str | None):
        st.subheader("Power burn real-time signal (gas generation)")
        st.caption("Uses EIA API v2 electricity RTO fuel-type data (hourly). If it fails, adjust facets below.")

        if not eia_key:
            st.warning("EIA API key not found. Add your key in sidebar (EIA Storage Settings).")
            return

        # User-tunable facets (because EIA facets can differ by dataset version)
        col1, col2, col3 = st.columns(3)
        with col1:
            respondent = st.text_input("respondent facet", value="US48")
        with col2:
            fueltype = st.text_input("fueltype facet", value="NG")
        with col3:
            freq = st.selectbox("frequency", ["hourly", "local-hourly"], index=0)

        end = dt.datetime.utcnow()
        start = end - dt.timedelta(days=8)

        base_url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
        params = {
            "api_key": eia_key,
            "frequency": freq,
            "data[0]": "value",
            "start": start.strftime("%Y-%m-%dT%H"),
            "end": end.strftime("%Y-%m-%dT%H"),
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 5000,
        }
        # Add facets if provided
        if respondent.strip():
            params["facets[respondent][]"] = respondent.strip()
        if fueltype.strip():
            params["facets[fueltype][]"] = fueltype.strip()

        try:
            js = _eia_v2_get(base_url, params)
            data = js.get("response", {}).get("data", [])
            import pandas as _pd
            df = _pd.DataFrame(data)
            if df.empty or "value" not in df.columns or "period" not in df.columns:
                st.warning("No data returned (check facets).")
                return
            df["period"] = _pd.to_datetime(df["period"], errors="coerce")
            df["value"] = _pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["period", "value"])
            if df.empty:
                st.warning("No usable data after parsing.")
                return

            # Sum across BAs for each timestamp if multiple rows
            series = df.groupby("period")["value"].sum().sort_index()
            last_24h = series[series.index >= (series.index.max() - _pd.Timedelta(hours=24))].mean()
            prev_7d = series[series.index < (series.index.max() - _pd.Timedelta(hours=24))]
            avg_7d = prev_7d.mean() if not prev_7d.empty else None


            st.write(f"Last 24h avg gas generation: **{last_24h:,.0f}** (units as provided by EIA)")

            # Compare vs short baseline (7d) AND medium baseline (~28d) to reduce false signals.
            delta7 = None
            delta28 = None
            if avg_7d is not None and avg_7d > 0:
                delta7 = (last_24h - avg_7d) / avg_7d
                st.write(f"Vs prior 7d avg: **{delta7*100:+.1f}%**")

            avg_28d = None
            try:
                if len(vals) >= 24 * 28:
                    avg_28d = float(pd.Series(vals).tail(24 * 28).mean())
                elif len(vals) >= 24 * 14:
                    avg_28d = float(pd.Series(vals).tail(24 * 14).mean())
            except Exception:
                avg_28d = None

            if avg_28d is not None and avg_28d > 0:
                delta28 = (last_24h - avg_28d) / avg_28d
                st.write(f"Vs prior ~28d avg: **{delta28*100:+.1f}%**")

            # Optional: confirm with GWDD trend if available from the app
            gwdd_hint = None
            try:
                od = st.session_state.get("overall_disp")
                if isinstance(od, pd.DataFrame) and (not od.empty) and ("Signal" in od.columns):
                    gwdd_hint = str(od.iloc[-1]["Signal"])
            except Exception:
                gwdd_hint = None

            # Decision
            thr = 0.02  # 2% threshold
            if delta7 is None and delta28 is None:
                st.info("Power burn signal: Neutral (not enough history).")
            else:
                d7_ok = (delta7 is not None)
                d28_ok = (delta28 is not None)

                bullish = ((not d7_ok or delta7 >= thr) and (not d28_ok or delta28 >= thr))
                bearish = ((not d7_ok or delta7 <= -thr) and (not d28_ok or delta28 <= -thr))

                if bullish:
                    if gwdd_hint == "BEARISH":
                        st.info("Power burn signal: Neutral (power burn up, but GWDD trend bearish)")
                    else:
                        st.success("Power burn signal: Bullish (above baseline gas generation)")
                elif bearish:
                    if gwdd_hint == "BULLISH":
                        st.info("Power burn signal: Neutral (power burn down, but GWDD trend bullish)")
                    else:
                        st.error("Power burn signal: Bearish (below baseline gas generation)")
                else:
                    st.info("Power burn signal: Neutral (mixed / near baseline)")
        except Exception as e:
            st.warning(f"Power burn fetch failed: {e}")

    tabs = st.tabs(["NG Drivers + Auto Signal", "GWDD Dashboard", "EIA Storage Dashboard", "Signals & Summary", "Contracts & Rollover", "NG News"])

    # -----------------------------
    # Tab 1: GWDD
    # -----------------------------
    with tabs[1]:
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

            def _fetch_city_gwdd(_row) -> pd.DataFrame | None:
                try:
                    cdf = fetch_open_meteo_daily(float(_row["lat"]), float(_row["lon"]), days=days)
                    gw = compute_gwdd(cdf, base_f=base_f)
                    gw["city"] = str(_row["city"])
                    gw["weight"] = float(_row["weight"])
                    return gw
                except Exception:
                    return None  # silent fail

            with st.spinner("Fetching weather + calculating GWDD..."):
                rows = [r for _, r in cities_df.iterrows()]
                max_workers = min(8, max(1, len(rows)))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = [ex.submit(_fetch_city_gwdd, r) for r in rows]
                    for fut in as_completed(futs):
                        gw = fut.result()
                        if gw is not None and not gw.empty:
                            city_frames.append(gw)

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
                c3.metric("Base (°C)", f"{base_c:g}")

                st.markdown("### Overall Weighted GWDD (Daily)")

                # Add 3-day & 7-day averages + bullish/neutral/bearish signal
                overall_disp = overall.copy()
                overall_disp["date"] = pd.to_datetime(overall_disp["date"])
                overall_disp = overall_disp.sort_values("date")
                overall_disp["GWDD_overall_weighted"] = pd.to_numeric(overall_disp["GWDD_overall_weighted"], errors="coerce")

                overall_disp["GWDD_avg_3d"] = overall_disp["GWDD_overall_weighted"].rolling(window=3, min_periods=1).mean()
                overall_disp["GWDD_avg_7d"] = overall_disp["GWDD_overall_weighted"].rolling(window=7, min_periods=1).mean()
                overall_disp["Trend_3d_minus_7d"] = overall_disp["GWDD_avg_3d"] - overall_disp["GWDD_avg_7d"]

                def _gwdd_signal(x: float) -> str:
                    if pd.isna(x):
                        return "N/A"
                    if x >= 0.5:
                        return "BULLISH"
                    if x <= -0.5:
                        return "BEARISH"
                    return "NEUTRAL"

                overall_disp["Signal"] = overall_disp["Trend_3d_minus_7d"].apply(_gwdd_signal)

                # Save for NG Drivers tab
                st.session_state["overall_disp"] = overall_disp.copy()

                # Show latest signal box
                if len(overall_disp) > 0:
                    latest = overall_disp.iloc[-1]
                    sig = str(latest["Signal"])
                    trend = float(latest["Trend_3d_minus_7d"]) if pd.notna(latest["Trend_3d_minus_7d"]) else 0.0
                    a3 = float(latest["GWDD_avg_3d"]) if pd.notna(latest["GWDD_avg_3d"]) else 0.0
                    a7 = float(latest["GWDD_avg_7d"]) if pd.notna(latest["GWDD_avg_7d"]) else 0.0
                    if sig == "BULLISH":
                        st.success(f"GWDD Signal: **BULLISH** (3d–7d = {trend:+.2f})")
                    elif sig == "BEARISH":
                        st.error(f"GWDD Signal: **BEARISH** (3d–7d = {trend:+.2f})")
                    elif sig == "NEUTRAL":
                        st.warning(f"GWDD Signal: **NEUTRAL** (3d–7d = {trend:+.2f})")
                    c1, c2 = st.columns(2)
                    c1.metric("GWDD 3-day avg", f"{a3:.2f}")
                    c2.metric("GWDD 7-day avg", f"{a7:.2f}")

                st.dataframe(
                    overall_disp[["date","GWDD_overall_weighted","GWDD_avg_3d","GWDD_avg_7d","Signal"]],
                    use_container_width=True,
                    hide_index=True,
                )
                st.line_chart(overall_disp.set_index("date")["GWDD_overall_weighted"])

                # -----------------------------
                # -----------------------------
                # US Gas-Weighted HDD (GWHDD) — Forecast + Trend (forecast-only, stable)
                # -----------------------------
                st.divider()
                st.subheader("Global NG Countries — Daily GWDD (Country + Date)")

                global_days = st.selectbox("Outlook days (Global)", [7, 14], index=1, key="global_days")

                rows = []
                weighted_daily = {}  # date -> weighted gwdd sum
                for _, r in GLOBAL_NG_COUNTRIES.iterrows():
                    try:
                        df_t = fetch_open_meteo_daily(float(r["lat"]), float(r["lon"]), days=int(global_days))
                        gw = compute_gwdd(df_t, base_f=base_f)
                        for d, t_f, g in zip(gw["date"].tolist(), gw["temp_f"].tolist(), gw["gwdd"].tolist()):
                            rows.append({
                                "date": d,
                                "country": str(r["country"]),
                                "city": str(r["city"]),
                                "temp_c": f_to_c(float(t_f)),
                                "gwdd": float(g),
                                "weight": float(r["weight"]),
                                "weighted_gwdd": float(g) * float(r["weight"]),
                            })
                            weighted_daily[d] = weighted_daily.get(d, 0.0) + (float(g) * float(r["weight"]))
                    except Exception as e:
                        pass  # global fetch failed (silenced to keep UI fast)

                if rows:
                    df_global = pd.DataFrame(rows).sort_values(["date", "country"]).reset_index(drop=True)
                    st.dataframe(
                        df_global[["date", "country", "city", "temp_c", "gwdd"]],
                        use_container_width=True,
                        hide_index=True,
                        height=320,
                    )

                    df_weighted = pd.DataFrame(
                        [{"date": d, "global_weighted_gwdd": v} for d, v in sorted(weighted_daily.items())]
                    )
                    df_weighted["global_weighted_gwdd"] = pd.to_numeric(df_weighted["global_weighted_gwdd"], errors="coerce")
                    st.markdown("### Global Weighted GWDD (Daily)")
                    st.dataframe(df_weighted, use_container_width=True, hide_index=True, height=220)
                    st.line_chart(df_weighted.set_index("date")["global_weighted_gwdd"])

                    # --- Single Signal: Bullish / Bearish / Neutral ---
                    cur_avg = float(df_weighted["global_weighted_gwdd"].head(min(7, len(df_weighted))).mean())
                    prev_avg = st.session_state.get("_prev_global_gwdd_avg")
                    st.session_state["_prev_global_gwdd_avg"] = cur_avg

                    signal_g = "Neutral"
                    if prev_avg is not None:
                        delta = cur_avg - float(prev_avg)
                        if delta >= 1.0:
                            signal_g = "Bullish"
                        elif delta <= -1.0:
                            signal_g = "Bearish"

                    st.markdown(f"### Global GWDD Signal: **{signal_g}**")
                else:
                    st.info("Global GWDD: No data returned (check internet).")

    # -----------------------------
    with tabs[2]:
        st.subheader("EIA Weekly Natural Gas Storage (BCF) + Weekly Injection/Withdrawal")

        st.markdown("### EIA storage report (forecast + actual)")
        next_report_date = estimate_next_eia_report_date(dt.date.today())
        st.caption(f"Estimated next report date: **{next_report_date.strftime('%Y-%m-%d')}** (typically Thursday)")

        # Save the market forecast for the upcoming report week (so Surprise is computed correctly later)
        colsf1, colsf2 = st.columns([1, 2])
        with colsf1:
            if st.button("Save forecast for next EIA week"):
                _set_saved_forecast(next_report_date, float(eia_market_forecast_bcf))
                st.success(f"Saved forecast for {next_report_date.isoformat()}: {float(eia_market_forecast_bcf):+.0f} Bcf")
        with colsf2:
            st.caption("Tip: save forecast before EIA release. After the report prints, the app will use the saved forecast for that same week to compute Surprise.")


        if not api_key or not str(api_key).strip():
            st.info("Enter your EIA API key in the sidebar to auto-show the latest actual change when the report is released.")
        else:
            try:
                latest_pack = fetch_eia_latest_storage_and_change(str(api_key).strip())
                actual_chg = latest_pack.get("actual_weekly_change_bcf", None)

                a1, a2, a3 = st.columns(3)
                with a1:
                    st.metric("Forecast (market)", f"{float(eia_market_forecast_bcf):+.0f} Bcf")
                with a2:
                    if actual_chg is None:
                        st.metric("Actual (latest)", "N/A")
                    else:
                        st.metric("Actual (latest)", f"{float(actual_chg):+.0f} Bcf", help=str(latest_pack.get("latest_period_date")))
                with a3:
                    if actual_chg is None:
                        st.metric("Surprise (actual - forecast)", "N/A")
                    else:
                        ld = latest_pack.get("latest_period_date")
                        saved_fc = _get_saved_forecast(ld) if ld else None

                        # --- AUTO SURPRISE fallback (added) ---
                        # If there is no saved forecast for this report date, use the current market forecast input
                        # and save it automatically so Surprise always updates.
                        if saved_fc is None:
                            try:
                                saved_fc = float(eia_market_forecast_bcf)
                            except Exception:
                                saved_fc = None
                            if (ld is not None) and (saved_fc is not None):
                                _set_saved_forecast(ld, saved_fc)

                        if saved_fc is None:
                            st.metric("Surprise (actual - forecast)", "N/A")
                        else:
                            surprise = float(actual_chg) - float(saved_fc)

                            # --- Color-coded Surprise + Alert (added) ---
                            # Interpretation: more negative surprise = bigger withdrawal / tighter than expected => bullish NG
                            if surprise <= -5:
                                s_label = "Bullish"
                                s_color = "#16a34a"  # green
                            elif surprise >= 5:
                                s_label = "Bearish"
                                s_color = "#dc2626"  # red
                            else:
                                s_label = "Neutral"
                                s_color = "#6b7280"  # gray

                            st.metric("Surprise (actual - forecast)", f"{surprise:+.0f} Bcf")
                            st.markdown(
                                f"<div style='margin-top:-8px; font-weight:700; color:{s_color};'>Signal from Surprise: {s_label}</div>",
                                unsafe_allow_html=True,
                            )

                            if abs(surprise) >= 20:
                                st.warning(f"Big Surprise alert: {surprise:+.0f} Bcf ({s_label})")

                if st.button("Refresh EIA now", help="Clears cached EIA fetch and reloads latest numbers."):
                    # Clear both the 'latest pack' cache and the full series cache
                    fetch_eia_latest_storage_and_change.clear()
                    fetch_eia_series.clear()
                    # Force a rerun after clearing cache (Streamlit version safe)
                    if hasattr(st, 'rerun'):
                        st.rerun()
                    elif hasattr(st, 'experimental_rerun'):
                        st.experimental_rerun()
            except Exception as _e:
                st.warning(f"Could not load latest EIA actual automatically: {_e}")

        # -----------------------------
        # Contango / Backwardation check (Front vs Next contract)
        # -----------------------------



    # --- EIA report forecast (date + actual, auto-updates) ---

        try:
            stor = fetch_eia_series(EIA_SERIES_TOTAL_R48, api_key=api_key)
            stor = compute_weekly_change(stor)

            # --- Actual (last week) + Forecast (next week) table ---
            s = stor.dropna(subset=["value"]).sort_values("date").copy()

            latest_row = s.iloc[-1]
            prev_row = s.iloc[-2] if len(s) >= 2 else None

            latest_storage = float(latest_row["value"])
            latest_date = latest_row["date"].date()

            # Actual weekly change (last reported week)
            actual_change = None
            if prev_row is not None:
                actual_change = float(latest_row["value"] - float(prev_row["value"]))

            # Next week forecast (your manual input)
            forecast_next_change = float(eia_market_forecast_bcf)

            # Projected next week storage (latest storage + forecast change)
            projected_next_storage = latest_storage + forecast_next_change

            rows = []
            rows.append({
                "Week": str(latest_date),
                "Type": "Last week (Actual)",
                "Change (BCF)": "N/A" if actual_change is None else f"{actual_change:+.0f}",
                "Storage (BCF)": f"{latest_storage:.0f}",
            })
            rows.append({
                "Week": "Next week",
                "Type": "Forecast (Manual input)",
                "Change (BCF)": f"{forecast_next_change:+.0f}",
                "Storage (BCF)": f"{projected_next_storage:.0f}",
            })

            st.markdown("### EIA Report — Actual (Last Week) vs Forecast (Next Week)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if actual_change is not None:
                saved_fc2 = _get_saved_forecast(latest_date)
                if saved_fc2 is None:
                    st.caption("Surprise (Actual - Forecast): N/A (save forecast for the same report week)")
                else:
                    surprise2 = float(actual_change) - float(saved_fc2)
                    st.caption(f"Surprise (Actual - Forecast): {surprise2:+.0f} Bcf")


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

            # Market forecast (manual) for next EIA report

            st.markdown("### Storage (BCF)")
            st.dataframe(stor.tail(104), use_container_width=True, hide_index=True)  # last ~2 years
            st.line_chart(stor.set_index("date")["value"])

            # (Hidden) Weekly Injection/Withdrawal (BCF) chart removed for cleaner UI

            # --- Add 5-year average + signal + next-7-day outlook (added) ---
            try:
                ch_df = stor.dropna(subset=["weekly_change"]).copy()
                ch_df["date"] = pd.to_datetime(ch_df["date"])
                ch_df = ch_df.sort_values("date")
                ch_df["weekly_change"] = pd.to_numeric(ch_df["weekly_change"], errors="coerce")
                ch_df = ch_df.dropna(subset=["weekly_change"])

                # 5-year average by ISO week number (uses last ~5 years of history)
                today = dt.date.today()
                start_5y = pd.Timestamp(today) - pd.Timedelta(days=365*5 + 31)
                hist5 = ch_df[ch_df["date"] >= start_5y].copy()
                # Exclude current year from the average to avoid leakage
                hist5 = hist5[hist5["date"].dt.year < today.year]
                if hist5.empty:
                    hist5 = ch_df[ch_df["date"] >= start_5y].copy()

                hist5["iso_week"] = hist5["date"].dt.isocalendar().week.astype(int)
                avg5_by_week = hist5.groupby("iso_week")["weekly_change"].mean()

                ch_df["iso_week"] = ch_df["date"].dt.isocalendar().week.astype(int)
                ch_df["avg_5y_week"] = ch_df["iso_week"].map(avg5_by_week)

                # Show combined series: actual vs 5y-average
                comp = ch_df.set_index("date")[["weekly_change", "avg_5y_week"]].rename(
                    columns={"weekly_change": "actual_weekly_change", "avg_5y_week": "avg_5y_weekly_change"}
                )
                st.markdown("### Weekly Injection/Withdrawal vs 5-Year Average (BCF) - Clear View")
                import altair as alt

                actual_df = comp.reset_index()[["date", "actual_weekly_change"]].rename(
                    columns={"date": "Date", "actual_weekly_change": "BCF"}
                )
                avg_df = comp.reset_index()[["date", "avg_5y_weekly_change"]].rename(
                    columns={"date": "Date", "avg_5y_weekly_change": "BCF"}
                )

                chart_actual = alt.Chart(actual_df).mark_line(color="#1f77b4").encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("BCF:Q", title="Actual weekly change (BCF)")
                ).properties(height=220)

                chart_avg = alt.Chart(avg_df).mark_line(color="#d62728").encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("BCF:Q", title="5-year average (BCF)")
                ).properties(height=220)

                st.altair_chart(
                    alt.vconcat(chart_actual, chart_avg).resolve_scale(y="independent"),
                    use_container_width=True,
                )

                # Latest comparison + signal
                latest_row = ch_df.iloc[-1]
                actual = float(latest_row["weekly_change"])
                avg5 = float(latest_row["avg_5y_week"]) if pd.notna(latest_row["avg_5y_week"]) else None

                # 3-week vs 7-week rolling averages (storage change trend)
                ch_df["avg_3w"] = ch_df["weekly_change"].rolling(window=3, min_periods=1).mean()
                ch_df["avg_7w"] = ch_df["weekly_change"].rolling(window=7, min_periods=1).mean()
                avg3w = float(ch_df["avg_3w"].iloc[-1])
                avg7w = float(ch_df["avg_7w"].iloc[-1])
                trend_w = avg3w - avg7w

                # Signal rules (simple + robust):
                # More negative (bigger withdrawals) => bullish. More positive (bigger injections) => bearish.
                score = 0
                if avg5 is not None:
                    diff_vs_5y = actual - avg5  # + means looser than normal, - means tighter than normal
                    if diff_vs_5y <= -10:  # tighter than normal by 10+ Bcf
                        score += 1
                    elif diff_vs_5y >= 10:  # looser than normal by 10+ Bcf
                        score -= 1
                else:
                    diff_vs_5y = None

                if trend_w <= -5:
                    score += 1
                elif trend_w >= 5:
                    score -= 1

                # Surprise (actual - saved forecast) if available from earlier table
                # Negative surprise => more withdrawal / tighter => bullish
                try:
                    if "surprise" in locals() and saved_fc is not None:
                        if float(surprise) <= -5:
                            score += 1
                        elif float(surprise) >= 5:
                            score -= 1
                except Exception:
                    pass

                if score >= 2:
                    sig = "BULLISH"
                    st.success("NG Storage Signal: **BULLISH**")
                elif score <= -2:
                    sig = "BEARISH"
                    st.error("NG Storage Signal: **BEARISH**")
                else:
                    sig = "NEUTRAL"
                    st.warning("NG Storage Signal: **NEUTRAL**")

                st.markdown("""
    **How to read this (EIA storage):**
    - **Weekly change** = this week's working gas minus last week.
      - **Negative** = *withdrawal* (storage falling) → usually **more bullish** in winter.
      - **Positive** = *injection* (storage rising) → usually **more bearish** in winter.
    - **5-year average (same week)** = the typical seasonal change for this exact week of the year.
    - The app compares **Actual vs 5-year avg**:
      - If **Actual is more negative** than the 5-year avg (bigger withdrawal / smaller injection), that's **tighter** → **Bullish**.
      - If **Actual is less negative / more positive** than the 5-year avg, that's **looser** → **Bearish**.
    - **3-week / 7-week avg** smooths noise and shows the trend.
    """)

                cA, cB, cC, cD = st.columns(4)
                cA.metric("Latest weekly change", f"{actual:+.0f} Bcf")
                cB.metric("5y avg (same week)", "n/a" if avg5 is None else f"{avg5:+.0f} Bcf")
                cC.metric("3-week avg", f"{avg3w:+.0f} Bcf")
                cD.metric("7-week avg", f"{avg7w:+.0f} Bcf")

                if diff_vs_5y is not None:
                    st.caption(f"Diff vs 5-year avg (actual - 5y): {diff_vs_5y:+.0f} Bcf (negative = tighter/bullish)")
                st.caption(f"Trend (3w - 7w): {trend_w:+.0f} Bcf (more negative = tightening/bullish)")

                # Next 7 days outlook (driver-based, not a guaranteed price forecast)
                st.markdown("### Next 7 Days Outlook (Driver-Based)")
                outlook_lines = []
                if sig == "BULLISH":
                    outlook_lines.append("• Bias: **Up / Bullish** (storage tightening vs normal)")
                elif sig == "BEARISH":
                    outlook_lines.append("• Bias: **Down / Bearish** (storage looser vs normal)")
                else:
                    outlook_lines.append("• Bias: **Sideways / Neutral** (mixed drivers)")

                # Use upcoming market forecast input if available
                try:
                    fc = float(eia_market_forecast_bcf)
                    if fc < 0:
                        outlook_lines.append(f"• Next EIA expectation: withdrawal forecast {fc:+.0f} Bcf (tends bullish)")
                    else:
                        outlook_lines.append(f"• Next EIA expectation: injection forecast {fc:+.0f} Bcf (tends bearish)")
                except Exception:
                    pass

                st.write("\n".join(outlook_lines))
            except Exception as _e:
                st.info("5-year avg / signal not available (data issue).")


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


    def _compute_us_gwhdd_trend(cities_df: pd.DataFrame, base_f: float, days: int, past_days: int = 7) -> dict:
        """
        Compute US Gas-Weighted HDD (GWHDD) actual vs forecast trend.
        Returns dict with: actual_avg, forecast_avg, delta, label
        """
        out = {"actual_avg": None, "forecast_avg": None, "delta": None, "label": "Neutral"}

        frames: List[pd.DataFrame] = []
        for _, row in cities_df.iterrows():
            try:
                hdf = fetch_open_meteo_daily_history(
                    float(row["lat"]), float(row["lon"]),
                    past_days=int(past_days),
                    forecast_days=int(days),
                )
                if hdf is None or hdf.empty:
                    continue
                gw = compute_gwdd(hdf, base_f=base_f).rename(columns={"gwdd": "gwhdd"})
                gw = gw.merge(hdf[["date", "kind"]], on="date", how="left")
                gw["weight"] = float(row["weight"])
                frames.append(gw)
            except Exception:
                continue

        if not frames:
            return out

        df = pd.concat(frames, ignore_index=True)
        df["w_gwhdd"] = df["gwhdd"] * df["weight"]

        overall = (
            df.groupby(["date", "kind"], as_index=False)
            .agg(weighted_sum=("w_gwhdd", "sum"), weight_sum=("weight", "sum"))
        )
        overall["gwhdd_w"] = overall["weighted_sum"] / overall["weight_sum"]

        actual = overall[overall["kind"] == "Actual"].sort_values("date")["gwhdd_w"]
        fcst = overall[overall["kind"] == "Forecast"].sort_values("date")["gwhdd_w"]

        if len(actual):
            out["actual_avg"] = float(actual.tail(min(7, len(actual))).mean())
        if len(fcst):
            out["forecast_avg"] = float(fcst.head(min(7, len(fcst))).mean())

        if out["actual_avg"] is not None and out["forecast_avg"] is not None:
            out["delta"] = float(out["forecast_avg"] - out["actual_avg"])
            if out["delta"] >= 1.0:
                out["label"] = "Bullish (colder / higher demand)"
            elif out["delta"] <= -1.0:
                out["label"] = "Bearish (warmer / lower demand)"
            else:
                out["label"] = "Neutral"
        return out



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


    with tabs[3]:
        st.header("Signals & Summary")
        st.divider()
        st.subheader("🌡️ NOAA 8–14 Day Temperature Outlook (Weather → NG Demand)")

        colA, colB = st.columns([2, 1])

        with colA:
            st.image(
                NOAA_814_TEMP_MAP,
                caption="NOAA CPC 8–14 Day Temperature Outlook (Blue = colder, Red = warmer)",
                use_container_width=True
            )

        with colB:
            st.markdown("### How to read")
            st.markdown(
                """
    - **Blue areas** → Colder than normal → **Bullish NG**
    - **Red areas** → Warmer than normal → **Bearish NG**
    - **Neutral/white** → Mixed → Range-bound NG
                """
            )
            st.markdown(f"[📄 NOAA Official Discussion]({NOAA_814_DISCUSSION})")


        st.caption("Bullish/Bearish summary + GWDD vs EIA divergence alert + projected next-week storage level.")
        # -----------------------------
        # NG price projection (Next 5 Days) — model-based (not guaranteed)
        # -----------------------------
        with st.expander("📈 NG Price Forecast (Next 5 Days) — Model Projection", expanded=False):
            st.caption("⚠️ ਇਹ **ਗਾਰੰਟੀਡ ਪ੍ਰਾਈਸ ਫੋਰਕਾਸਟ ਨਹੀਂ**। ਇਹ ਸਿਰਫ਼ ਪਿਛਲੇ ਡਾਟਾ ਤੋਂ ਇੱਕ ਸਾਦੀ projection ਹੈ (trend + volatility band).")
            hist = fetch_ng_history_daily("NG=F", period="3mo")
            proj = project_ng_prices_next_5_days(hist)

            if hist.empty or proj.empty:
                if yf is None:
                    st.info("ਇਸ module ਲਈ **yfinance** ਚਾਹੀਦਾ ਹੈ (pip install yfinance). ਜੇ ਤੁਸੀਂ paid Investing.com ਵਰਤਦੇ ਹੋ, ਮੈਂ ਉਹ integration ਵੀ add ਕਰ ਸਕਦਾ ਹਾਂ।")
                else:
                    st.info("NG history load ਨਹੀਂ ਹੋਈ (internet / Yahoo delay).")
            else:
                last_close = float(hist["close"].iloc[-1])
                st.metric("Latest NG=F close (Yahoo)", f"{last_close:.3f}")
                show = proj.copy()
                show["date"] = show["date"].dt.strftime("%Y-%m-%d")
                show = show.rename(columns={"date": "Date", "projected": "Projected", "low": "Low (band)", "high": "High (band)"})
                st.dataframe(show, use_container_width=True, hide_index=True)

                # Profit / Loss helper (optional)
                st.divider()
                st.markdown("### 🧮 Profit / Loss (Estimate)")
                c1, c2, c3 = st.columns(3)
                with c1:
                    entry = st.number_input("Entry price (NG)", value=float(last_close), step=0.01)
                with c2:
                    units = st.number_input("Units (contracts/shares/CFD size)", value=1.0, step=1.0)
                with c3:
                    direction = st.selectbox("Direction", ["LONG (Buy)", "SHORT (Sell)"], index=0)

                sign = 1.0 if direction.startswith("LONG") else -1.0
                pl = proj.copy()
                pl["P/L (estimate)"] = (pl["projected"] - float(entry)) * float(units) * sign
                pl_show = pl[["date", "projected", "P/L (estimate)"]].copy()
                pl_show["date"] = pl_show["date"].dt.strftime("%Y-%m-%d")
                pl_show = pl_show.rename(columns={"date": "Date", "projected": "Projected"})
                st.dataframe(pl_show, use_container_width=True, hide_index=True)
                st.caption("ਇਹ P/L **approx** ਹੈ—broker fees, slippage, leverage/ETF decay (HNU), rollover ਆਦਿ include ਨਹੀਂ।")


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



        # US GWHDD Trend (Actual vs Forecast)
        us_tr = _compute_us_gwhdd_trend(cities_df, base_f=base_f, days=days, past_days=7)
        if us_tr.get("delta") is None:
            st.caption("US GWHDD Trend: N/A (weather data not available)")
        else:
            st.caption(f"US GWHDD Trend: {us_tr['label']}  •  Δ(next7-last7) = {float(us_tr['delta']):+.1f}")

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

    # -----------------------------
    # Tab 4: Contracts & Rollover
    # -----------------------------
    with tabs[4]:

        # Holiday set for more accurate rollover estimates (major US holidays + Good Friday)
        years = list(range(dt.date.today().year - 1, dt.date.today().year + 3))
        holiday_set = _holiday_set_for_years(years)
        st.subheader("Henry Hub NG — Contracts & Rollover (Estimate)")
        st.caption("Expiry rule used: 3 business days before the 1st of the delivery month (weekends only). Holidays can shift dates.")

        with st.expander("CME holidays used (observed dates) — for rollover math", expanded=False):
            # Show the holiday dates the app is using for business-day calculations
            hol_by_year = {}
            for d in sorted(holiday_set):
                hol_by_year.setdefault(d.year, []).append(d)
            for yy in sorted(hol_by_year):
                st.markdown(f"**{yy}**")
                st.write(", ".join([x.strftime("%Y-%m-%d") for x in hol_by_year[yy]]))
            st.caption("Note: This is an approximation (major US holidays + Good Friday). Early-close days are not included.")


        today = dt.date.today()
        y, m = front_delivery_month(today)
        exp = ng_contract_expiry(y, m)
        roll_start = _add_business_days(exp, -5)
        roll_reco = _add_business_days(exp, -1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Today", today.strftime("%Y-%m-%d"))
        c2.metric("Front (delivery) month", f"{y}-{m:02d}  ({ng_symbol(y,m)})")
        c3.metric("Estimated expiry", exp.strftime("%Y-%m-%d"))

        # Rollover month name (auto-updates)
        roll_month_name = dt.date(y, m, 1).strftime("%B %Y")
        st.metric("Rollover Month", roll_month_name)


        c4, c5 = st.columns(2)
        c4.metric("Rollover start date", roll_start.strftime("%Y-%m-%d"))
        c5.metric("Rollover end date (expiry)", exp.strftime("%Y-%m-%d"))


        # -----------------------------
        # EIA Storage Report — forecast vs actual (shown ONLY on this page)
        # -----------------------------
        st.markdown("### Curve check: Next contract contango or backwardation?")

        # Determine next delivery month
        if m == 12:
            ny, nm = y + 1, 1
        else:
            ny, nm = y, m + 1

        front_yahoo = f"{ng_symbol(y, m)}.NYM"
        next_yahoo = f"{ng_symbol(ny, nm)}.NYM"

        if yf is None:
            st.info("To auto-check contango/backwardation, install yfinance:  `pip install yfinance`  (then restart the app).")
            front_px = st.number_input("Front contract price (manual)", value=0.0, step=0.01, help="Enter front-month futures price (USD/MMBtu)")
            next_px = st.number_input("Next contract price (manual)", value=0.0, step=0.01, help="Enter next-month futures price (USD/MMBtu)")
        else:
            front_px = fetch_yahoo_last_price(front_yahoo)
            next_px = fetch_yahoo_last_price(next_yahoo)

            c4, c5 = st.columns(2)
            with c4:
                st.metric("Front (Yahoo)", front_yahoo, help="Yahoo monthly futures ticker (delayed)")
            with c5:
                st.metric("Next (Yahoo)", next_yahoo, help="Yahoo monthly futures ticker (delayed)")

            if front_px is None or next_px is None:
                st.warning("Could not load one or both monthly contract prices from Yahoo. You can still enter prices manually below.")
                front_px = st.number_input("Front contract price (manual)", value=float(front_px or 0.0), step=0.01)
                next_px = st.number_input("Next contract price (manual)", value=float(next_px or 0.0), step=0.01)

        # Compute structure
        structure = "N/A"
        spread = None
        spread_pct = None
        if front_px and next_px and float(front_px) != 0.0:
            spread = float(next_px) - float(front_px)
            spread_pct = (spread / float(front_px)) * 100.0

            if spread > 0:
                structure = "Contango (next > front)"
            elif spread < 0:
                structure = "Backwardation (next < front)"
            else:
                structure = "Flat (next ≈ front)"

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Front price", "N/A" if front_px is None else f"{float(front_px):.3f}")
        with s2:
            st.metric("Next price", "N/A" if next_px is None else f"{float(next_px):.3f}")
        with s3:
            if spread is None:
                st.metric("Spread (next-front)", "N/A")
            else:
                st.metric("Spread (next-front)", f"{spread:+.3f} ({spread_pct:+.2f}%)")

        if structure != "N/A":
            st.write(f"**Structure:** {structure}")
        st.caption("Note: This is a simple front-vs-next check using delayed Yahoo monthly futures (or manual input).")


        # -----------------------------
        # HNU rollover impact estimate (optional)
        # -----------------------------
        with st.expander("HNU rollover estimate (optional)", expanded=False):
            st.caption("Estimate only (not broker-exact). Assumes HNU behaves ~2× daily % move of NG and rollover impact approximates -2×(next-front)/front.")
            hnu_px = st.number_input("Current HNU price (CAD)", value=0.0, step=0.01, help="Enter current HNU market price to estimate rollover impact.")
            leverage = st.number_input("HNU leverage vs NG (approx)", value=2.0, step=0.1, help="HNU is approximately 2× daily NG move (estimate).")

            if front_px and next_px and float(front_px) != 0.0 and hnu_px and float(hnu_px) > 0.0:
                roll_pct = (float(next_px) - float(front_px)) / float(front_px)  # + = contango (headwind), - = backwardation (tailwind)
                hnu_roll_pct = -float(leverage) * roll_pct
                est_hnu = float(hnu_px) * (1.0 + hnu_roll_pct)

                if hnu_roll_pct > 0:
                    label = "tailwind (backwardation)"
                elif hnu_roll_pct < 0:
                    label = "headwind (contango)"
                else:
                    label = "flat"

                cA, cB = st.columns(2)
                with cA:
                    st.metric("Estimated rollover impact", f"{hnu_roll_pct*100:+.1f}%")
                    st.caption(label)
                with cB:
                    st.metric("Estimated HNU after rollover (NG flat)", f"{est_hnu:.2f}")
            else:
                st.info("Enter current HNU price above (and ensure front/next prices are available) to see the rollover estimate.")

        st.markdown(
            f"""
    **Rollover window (suggested):** {roll_start.strftime("%Y-%m-%d")} to {exp.strftime("%Y-%m-%d")}  
    **Suggested rollover day:** {roll_reco.strftime("%Y-%m-%d")} (1 business day before expiry)

    If you trade **NG=F (Yahoo front month)** or CFDs, the rollover behavior can differ by broker/platform.
    """
        )

        # Next 12 delivery months table
        rows = []
        yy, mm = y, m
        for i in range(12):
            sym = ng_symbol(yy, mm)
            expiry = ng_contract_expiry(yy, mm)
            rows.append({
                "Delivery Month": f"{yy}-{mm:02d}",
                "Symbol": sym,
                "Estimated Expiry": expiry.strftime("%Y-%m-%d"),
                "Suggested Rollover": _add_business_days(expiry, -1).strftime("%Y-%m-%d"),
            })
            # advance month
            if mm == 12:
                yy += 1
                mm = 1
            else:
                mm += 1

        df_roll = pd.DataFrame(rows)
        st.dataframe(df_roll, use_container_width=True)





    # -----------------------------
    # Tab 5: NG News
    # -----------------------------
    with tabs[5]:
        st.subheader("📰 Natural Gas News (Headlines)")
        st.caption("Headlines only (opens sources in a new tab).")

        if feedparser is None:
            st.info("To enable this tab, install feedparser:  pip install feedparser  (then restart the app).")
        else:
            # Keep feeds small to stay fast
            FEEDS = [
                "https://www.eia.gov/rss/todayinenergy.xml",
                "https://www.rigzone.com/news/rss/rigzone_latest.aspx",
                "https://www.rigzone.com/news/rss/rigzone_finance.aspx",
            ]

            col1, col2 = st.columns([1, 2])
            with col1:
                max_items = st.number_input("Max headlines", min_value=5, max_value=50, value=25, step=5)
            with col2:
                keyword = st.text_input("Filter keyword (optional)", value="", help="Example: storage, LNG, Freeport, Henry Hub")

            @st.cache_data(ttl=60 * 15, show_spinner=False)  # 15 min
            def _fetch_all_news():
                items = []
                for url in FEEDS:
                    d = feedparser.parse(url)
                    source = getattr(getattr(d, "feed", {}), "title", "") or url
                    for e in (getattr(d, "entries", []) or []):
                        items.append({
                            "title": (e.get("title") or "").strip(),
                            "link": (e.get("link") or "").strip(),
                            "published": (e.get("published") or e.get("updated") or "").strip(),
                            "source": source,
                        })
                # De-dup
                seen = set()
                uniq = []
                for it in items:
                    key = (it.get("link") or "") + "|" + (it.get("title") or "")
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(it)
                return uniq

            # Manual refresh button (clears cache)
            if st.button("Refresh headlines"):
                _fetch_all_news.clear()

            uniq = _fetch_all_news()

            if keyword and keyword.strip():
                k = keyword.strip().lower()
                uniq = [it for it in uniq if k in (it.get("title","").lower()) or k in (it.get("source","").lower())]

            if not uniq:
                st.info("No headlines found (check internet or try removing the keyword filter).")
            else:
                st.write(f"Showing **{min(int(max_items), len(uniq))}** headlines from {len(FEEDS)} feeds.")
                for it in uniq[: int(max_items)]:
                    t = it.get("title") or "(no title)"
                    link = it.get("link") or ""
                    src = it.get("source") or ""
                    pub = it.get("published") or ""
                    st.markdown(f"- [{t}]({link})  \n  {src}  •  {pub}")


    # ------------------------------
    # NG Drivers + Auto Signal (NEW)
    # ------------------------------
    def _ema(series, span: int):
        return series.ewm(span=span, adjust=False).mean()

    def compute_ng_price_signal():
        # Returns dict: {"signal": "Bullish"/"Neutral"/"Bearish", "details": "..."}
        try:
            import yfinance as yf
            dfp = yf.download("NG=F", period="90d", interval="1d", progress=False)
            if dfp is None or dfp.empty:
                return {"signal": "Neutral", "details": "Price: no data (yfinance/internet)."}
            close = dfp["Close"].dropna()
            if len(close) < 30:
                return {"signal": "Neutral", "details": "Price: not enough history."}

            ema9 = _ema(close, 9)
            ema21 = _ema(close, 21)

            last = float(close.iloc[-1])
            e9 = float(ema9.iloc[-1])
            e21 = float(ema21.iloc[-1])

            chg5 = float(close.iloc[-1] - close.iloc[-6]) if len(close) >= 6 else 0.0

            if e9 > e21 and chg5 > 0:
                sig = "Bullish"
            elif e9 < e21 and chg5 < 0:
                sig = "Bearish"
            else:
                sig = "Neutral"

            details = f"NG=F last={last:.3f} | EMA9={e9:.3f} | EMA21={e21:.3f} | 5D Δ={chg5:.3f}"
            return {"signal": sig, "details": details}
        except Exception as e:
            return {"signal": "Neutral", "details": f"Price: error ({type(e).__name__})."}

    def compute_storage_signal_simple(eia_api_key):
        # Bullish when actual weekly change is lower than 5y avg by >= 10 Bcf (tighter).
        if not eia_api_key:
            return {"signal": "Neutral", "details": "Storage: no EIA key."}
        try:
            series_id = globals().get("STORAGE_SERIES_ID", "NG.NW2_EPG0_SWO_R48_BCF.W")
            fetch_fn = globals().get("fetch_eia_series") or globals().get("fetch_eia_series_seriesid")
            if fetch_fn is None:
                return {"signal": "Neutral", "details": "Storage: fetch function missing."}

            # fetch_eia_series signature is (series_id, api_key).
            # Some older helper functions may use (api_key, series_id), so we try both safely.
            try:
                stor = fetch_fn(series_id, eia_api_key)
            except TypeError:
                stor = fetch_fn(eia_api_key, series_id)
            if stor is None or stor.empty:
                return {"signal": "Neutral", "details": "Storage: no data."}

            stor = stor.sort_values("date")
            stor["weekly_change"] = stor["value"].diff()

            last_row = stor.dropna(subset=["weekly_change"]).iloc[-1:]
            if last_row.empty:
                return {"signal": "Neutral", "details": "Storage: not enough data."}

            last_date = last_row["date"].iloc[0]
            last_change = float(last_row["weekly_change"].iloc[0])

            tmp = stor.copy()
            tmp["week"] = tmp["date"].dt.isocalendar().week.astype(int)
            tmp["year"] = tmp["date"].dt.year.astype(int)
            last_week = int(last_date.isocalendar().week)

            hist = tmp[(tmp["week"] == last_week) & (tmp["year"] < int(last_date.year))].tail(5)
            hist_changes = hist["weekly_change"].dropna()
            if hist_changes.empty:
                avg5 = float(tmp[tmp["week"] == last_week]["weekly_change"].dropna().tail(10).mean())
            else:
                avg5 = float(hist_changes.mean())

            diff = last_change - avg5  # negative = tighter
            if diff <= -10:
                sig = "Bullish"
            elif diff >= 10:
                sig = "Bearish"
            else:
                sig = "Neutral"

            details = f"Latest weekly change={last_change:.0f} Bcf | 5y avg (same week)={avg5:.0f} Bcf | diff={diff:.0f} (neg=tighter)"
            return {"signal": sig, "details": details}
        except Exception as e:
            return {"signal": "Neutral", "details": f"Storage: error ({type(e).__name__})."}

    def compute_weather_signal_simple(overall_df):
        # Weather signal from Overall Weighted GWDD (Daily)
        try:
            if overall_df is None or getattr(overall_df, "empty", True) or "GWDD_overall_weighted" not in overall_df.columns:
                return {"signal": "Neutral", "details": "Weather: no GWDD overall data."}
            d = overall_df.copy().sort_values("date")
            s = d["GWDD_overall_weighted"].astype(float)

            if len(s) < 14:
                avg7 = float(s.tail(7).mean())
                return {"signal": "Neutral", "details": f"Weather: 7D avg={avg7:.2f} (need more history)."}

            avg7_now = float(s.tail(7).mean())
            avg7_prev = float(s.tail(14).head(7).mean())
            delta = avg7_now - avg7_prev

            if delta > 1.0 and avg7_now >= 18:
                sig = "Bullish"
            elif delta < -1.0 and avg7_now <= 16:
                sig = "Bearish"
            else:
                sig = "Neutral"

            details = f"GWDD 7D avg={avg7_now:.2f} | prev7={avg7_prev:.2f} | Δ={delta:+.2f}"
            return {"signal": sig, "details": details}
        except Exception as e:
            return {"signal": "Neutral", "details": f"Weather: error ({type(e).__name__})."}

    def combine_signals(s_weather, s_storage, s_price):
        weight = {"Weather": 0.4, "Storage": 0.4, "Price": 0.2}
        score_map = {"Bearish": -1, "Neutral": 0, "Bullish": 1}
        score = (
            score_map.get(s_weather.get("signal"), 0) * weight["Weather"]
            + score_map.get(s_storage.get("signal"), 0) * weight["Storage"]
            + score_map.get(s_price.get("signal"), 0) * weight["Price"]
        )
        if score >= 0.35:
            sig = "Bullish"
        elif score <= -0.35:
            sig = "Bearish"
        else:
            sig = "Neutral"
        return sig, score

    with tabs[0]:
        st.header("NG Drivers + Auto Signal")

        auto_refresh = st.checkbox("Auto-refresh every 60 seconds", value=False)
        refresh_now = st.button("Refresh now")

        # Try to get EIA key (sidebar input likely sets st.session_state)
        eia_key = st.session_state.get("EIA_API_KEY") if "EIA_API_KEY" in st.session_state else None
        # st.secrets throws StreamlitSecretNotFoundError if no secrets.toml exists, so guard with try/except
        if not eia_key:
            try:
                secrets_obj = st.secrets
                if "EIA_API_KEY" in secrets_obj:
                    eia_key = secrets_obj.get("EIA_API_KEY")
            except Exception:
                pass

        overall_df = st.session_state.get("overall_disp") if "overall_disp" in st.session_state else None

        weather_sig = compute_weather_signal_simple(overall_df)
        storage_sig = compute_storage_signal_simple(eia_key)
        price_sig = compute_ng_price_signal()

        final_sig, final_score = combine_signals(weather_sig, storage_sig, price_sig)

        if final_sig == "Bullish":
            st.success(f"Overall NG Signal: {final_sig} (score={final_score:+.2f})")
        elif final_sig == "Bearish":
            st.error(f"Overall NG Signal: {final_sig} (score={final_score:+.2f})")
        else:
            st.info(f"Overall NG Signal: {final_sig} (score={final_score:+.2f})")

        st.subheader("Signal breakdown")
        st.write(f"**Weather:** {weather_sig['signal']} — {weather_sig['details']}")
        st.write(f"**Storage:** {storage_sig['signal']} — {storage_sig['details']}")
        st.write(f"**Price/Trend:** {price_sig['signal']} — {price_sig['details']}")


        st.subheader("Live Drivers (Auto)")
        cols = st.columns(2)
        with cols[0]:
            _render_lng_tracker()
            _render_freezeoff_alert()
        with cols[1]:
            _render_backwardation_detector()
            _render_power_burn_score(eia_key)


        # (removed empty subheader placeholder)

        if auto_refresh and not refresh_now:
            import time
            time.sleep(60)
            st.experimental_rerun()
        elif refresh_now:
            st.experimental_rerun()



    # Mark app as loaded so the full-screen overlay only appears on the FIRST open
    if not st.session_state.get("app_loaded", False):
        st.session_state["app_loaded"] = True

    # Remove the full-screen loader once the UI has been rendered
    _loader.empty()

main()
