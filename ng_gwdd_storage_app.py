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

# Optional market data (front-month NG)
try:
    import yfinance as yf
except Exception:
    yf = None


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



@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)  # 6 hours
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
            r = requests.get(url, params=params, timeout=30)
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

tabs = st.tabs(["GWDD Dashboard", "EIA Storage Dashboard", "Signals & Summary", "Contracts & Rollover", "NG Price Indexes & Curves"])

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

            st.divider()
            st.subheader("Global demand regions — Weather forecast & GWDD (export / burn markets)")
            st.caption("Add-on view: selected world regions where natural gas demand matters (US regions + key LNG/Mexico markets). Uses the same GWDD base you set in the sidebar.")

            # Editable region list (inside the tab — does not affect your sidebar cities)
            regions_df = st.data_editor(
                GLOBAL_DEMAND_REGIONS,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "region": st.column_config.TextColumn("Region"),
                    "lat": st.column_config.NumberColumn("Lat"),
                    "lon": st.column_config.NumberColumn("Lon"),
                    "weight": st.column_config.NumberColumn("Weight", help="Higher weight = more demand impact"),
                },
                key="global_regions_editor",
            )

            selected = st.multiselect(
                "Select regions to include",
                options=regions_df["region"].astype(str).tolist(),
                default=regions_df["region"].astype(str).tolist()[:6],
                key="global_regions_select",
            )

            if not selected:
                st.info("Select at least one region to show global weather + GWDD.")
            else:
                g_frames = []
                with st.spinner("Fetching global weather + calculating GWDD..."):
                    for _, rrow in regions_df.iterrows():
                        name = str(rrow.get("region", "")).strip()
                        if name not in selected:
                            continue
                        try:
                            rdf = fetch_open_meteo_daily(float(rrow["lat"]), float(rrow["lon"]), days=days)
                            rgw = compute_gwdd(rdf, base_f=base_f)
                            rgw["region"] = name
                            rgw["weight"] = float(rrow.get("weight", 1.0))
                            g_frames.append(rgw)
                        except Exception as e:
                            st.warning(f"Region failed: {name} — {e}")

                if g_frames:
                    g_all = pd.concat(g_frames, ignore_index=True)
                    g_pivot = g_all.pivot_table(index="date", columns="region", values="gwdd", aggfunc="mean").sort_index()

                    # Weighted overall
                    g_all["w_gwdd"] = g_all["gwdd"] * g_all["weight"]
                    g_overall = (
                        g_all.groupby("date", as_index=False)
                        .agg(weighted_sum=("w_gwdd", "sum"), weight_sum=("weight", "sum"))
                    )
                    g_overall["GWDD_global_weighted"] = g_overall["weighted_sum"] / g_overall["weight_sum"]
                    g_overall = g_overall[["date", "GWDD_global_weighted"]].sort_values("date")

                    cga, cgb = st.columns([1, 1])
                    with cga:
                        st.markdown("#### Global weighted GWDD (daily)")
                        st.dataframe(g_overall, use_container_width=True, hide_index=True)
                        st.line_chart(g_overall.set_index("date")["GWDD_global_weighted"])
                    with cgb:
                        st.markdown("#### Region GWDD (daily)")
                        st.dataframe(g_pivot, use_container_width=True)
                        st.line_chart(g_pivot)

                    # Quick temperature snapshot (next day)
                    st.markdown("#### Next-day temperature snapshot (°F)")
                    next_day = g_all["date"].min()
                    snap = (
                        g_all[g_all["date"] == next_day][["region", "temp_f", "gwdd", "weight"]]
                        .sort_values("weight", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.dataframe(snap, use_container_width=True, hide_index=True)
                else:
                    st.error("No global region data fetched. Check internet or region coordinates.")

# -----------------------------
# Tab 2: EIA Storage
# -----------------------------
with tabs[1]:
    st.subheader("EIA Weekly Natural Gas Storage (BCF) + Weekly Injection/Withdrawal")

    st.markdown("### EIA storage report (forecast + actual)")
    next_report_date = estimate_next_eia_report_date(dt.date.today())
    st.caption(f"Estimated next report date: **{next_report_date.strftime('%Y-%m-%d')}** (typically Thursday)")

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
                    surprise = float(actual_chg) - float(eia_market_forecast_bcf)
                    st.metric("Surprise (actual - forecast)", f"{surprise:+.0f} Bcf")

            if st.button("Refresh EIA now", help="Clears cached EIA fetch and reloads latest numbers."):
                fetch_eia_latest_storage_and_change.clear()
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
            surprise2 = actual_change - forecast_next_change
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

# -----------------------------
# Tab 4: Contracts & Rollover
# -----------------------------
with tabs[3]:

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
            structure = "Contango (next > front) → rollover drag for longs (HNU), helps shorts (HND)"
        elif spread < 0:
            structure = "Backwardation (next < front) → rollover benefit for longs (HNU), hurts shorts (HND)"
        else:
            structure = "Flat (next ≈ front) → minimal rollover impact"

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
# Tab 5: NG Price Indexes & Curves (ADD ONLY)
# -----------------------------
with tabs[4]:
    st.header("Natural Gas Price Indexes & Forward Curves")
    st.caption("Bidweek, Spot, Weekly, Forward Curves, LNG, Mexico, Shale & Historical Data")
    st.info("This tab is ADD-ONLY. No existing logic or calculations are modified.")

    st.subheader("Bidweek Price Indexes")
    st.caption("First-of-Month natural gas price indexes (150+ locations in North America)")
    bidweek_file = st.file_uploader("Upload Bidweek CSV (optional)", type=["csv"], key="bidweek")
    if bidweek_file:
        df = pd.read_csv(bidweek_file)
        st.dataframe(df, use_container_width=True)

    st.subheader("Daily Price Indexes (Spot)")
    st.caption("Daily spot natural gas price indexes (170+ locations in North America)")
    daily_file = st.file_uploader("Upload Daily Spot CSV (optional)", type=["csv"], key="daily")
    if daily_file:
        df = pd.read_csv(daily_file)
        st.dataframe(df, use_container_width=True)

    st.subheader("Weekly Price Indexes")
    st.caption("Weekly averages of spot natural gas price indexes")
    weekly_file = st.file_uploader("Upload Weekly CSV (optional)", type=["csv"], key="weekly")
    if weekly_file:
        df = pd.read_csv(weekly_file)
        st.dataframe(df, use_container_width=True)

    st.subheader("Forward Curve Data")
    st.caption("Monthly basis & fixed forward curves out 10 years (updated daily) at key natgas hubs")
    curve_file = st.file_uploader("Upload Forward Curve CSV (optional)", type=["csv"], key="curve")
    if curve_file:
        df = pd.read_csv(curve_file)
        st.dataframe(df, use_container_width=True)
        # If user has a date/period column + a price column, they can rename to these
        if "price" in df.columns:
            try:
                st.line_chart(df.set_index(df.columns[0])["price"])
            except Exception:
                pass

    st.subheader("LNG Data Suite")
    st.caption("Key data to support North American natural gas and LNG business decisions")
    lng_file = st.file_uploader("Upload LNG CSV (optional)", type=["csv"], key="lng")
    if lng_file:
        df = pd.read_csv(lng_file)
        st.dataframe(df, use_container_width=True)

    st.subheader("Mexico Price & Flow Data")
    st.caption("Mexico natural gas pricing and pipeline flow data")
    mexico_file = st.file_uploader("Upload Mexico CSV (optional)", type=["csv"], key="mexico")
    if mexico_file:
        df = pd.read_csv(mexico_file)
        st.dataframe(df, use_container_width=True)

    st.subheader("Daily Preliminary Prices")
    st.caption("Indicative daily natural gas price data based on actual trade data")
    prelim_daily = st.file_uploader("Upload Daily Preliminary CSV (optional)", type=["csv"], key="prelim_daily")
    if prelim_daily:
        df = pd.read_csv(prelim_daily)
        st.dataframe(df, use_container_width=True)

    st.subheader("Bidweek Preliminary Prices")
    st.caption("Indicative bidweek pricing (last three days of the month) based on actual trade data")
    prelim_bidweek = st.file_uploader("Upload Bidweek Preliminary CSV (optional)", type=["csv"], key="prelim_bidweek")
    if prelim_bidweek:
        df = pd.read_csv(prelim_bidweek)
        st.dataframe(df, use_container_width=True)

    st.subheader("Shale Prices")
    st.caption("Transparent pricing data for major shale and unconventional plays in North America")
    shale_file = st.file_uploader("Upload Shale Prices CSV (optional)", type=["csv"], key="shale")
    if shale_file:
        df = pd.read_csv(shale_file)
        st.dataframe(df, use_container_width=True)

    st.subheader("Historical Data")
    st.caption("Historical datasets (some series back to 1988)")
    hist_file = st.file_uploader("Upload Historical CSV (optional)", type=["csv"], key="historical")
    if hist_file:
        df = pd.read_csv(hist_file)
        st.dataframe(df, use_container_width=True)

    st.success("Loaded. When you have an OPIS / Argus / S&P export, upload CSVs here. API integration can be added later (still safe).")

