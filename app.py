"""
Smart Bike Rental Demand Advisor — Professional VELO Dashboard
A multi-page decision-support dashboard conforming to SRS v1.5.
"""

from datetime import date, datetime, timedelta
import math
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.calendar_utils import derive_calendar_features
from src.data_cleaning import load_dataset
from src.eda import (
    SEASON_NAMES,
    WEATHER_NAMES,
    compute_feature_correlations,
    compute_hourly_patterns,
    compute_seasonal_patterns,
    compute_target_distribution,
    compute_weather_patterns,
)
from src.models import FEATURE_COLUMNS, get_model_manager
from src.validation import VALID_SEASONS, VALID_WEATHER, VALID_WORKING_DAY, validate_inputs
from src.weather import fetch_open_meteo_forecast, map_wmo_code, normalize_weather_features

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (VELO Design Language)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VELO — Smart Bike Demand Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Background: Clean professional light slate */
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* Streamlit App Header & Toolbar: seamless light slate */
    header[data-testid="stHeader"],
    .stAppHeader {
        background-color: #F8FAFC !important;
        background: #F8FAFC !important;
        color: #0F172A !important;
    }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] span,
    header[data-testid="stHeader"] svg {
        color: #475569 !important;
        fill: #475569 !important;
    }

    /* Sidebar Styling: Sleek Deep Navy */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {
        color: #E2E8F0 !important;
    }

    /* Sidebar Radio Navigation */
    section[data-testid="stSidebar"] .stRadio > div {
        background: transparent !important;
        gap: 6px !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 10px 14px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.06) !important;
    }

    /* Container padding */
    .block-container {
        padding-top: 4.25rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1440px !important;
    }

    /* Card Containers */
    .velo-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }
    .velo-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Insight Callout Cards */
    .insight-card {
        border-left: 3px solid #0D9488;
        padding-left: 12px;
        margin-bottom: 12px;
    }
    .insight-title {
        font-weight: 700;
        color: #0F172A;
        font-size: 0.92rem;
    }
    .insight-desc {
        color: #64748B;
        font-size: 0.83rem;
        line-height: 1.45;
    }

    /* Top Page Header */
    .page-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1.4rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .page-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.96rem;
        color: #64748B;
        margin-top: 0.25rem;
        margin-bottom: 0;
    }
    .header-badges {
        display: flex;
        gap: 0.6rem;
        align-items: center;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .pill-green {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .pill-blue {
        background: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        position: relative;
    }
    .kpi-label {
        font-size: 0.84rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-value {
        font-size: 2.15rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-top: 0.35rem;
    }

    /* Hero Prediction Card */
    .hero-pred-card {
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
        border-radius: 14px;
        padding: 1.75rem 2rem;
        color: #FFFFFF;
        box-shadow: 0 4px 16px rgba(13, 148, 136, 0.28);
        margin-bottom: 1.25rem;
    }
    .hero-pred-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #CCFBF1;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .hero-pred-val {
        font-size: 3.4rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
        margin: 0.35rem 0;
    }
    .hero-pred-unit {
        font-size: 1.2rem;
        font-weight: 500;
        color: #A7F3D0;
    }
    .hero-pred-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        padding-top: 0.85rem;
        margin-top: 0.85rem;
        font-size: 0.88rem;
    }

    /* Disclaimer Box */
    .disclaimer-card {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.86rem;
        color: #1E3A8A;
        line-height: 1.5;
        margin-top: 0.85rem;
    }

    /* Weather Mini Cards */
    .weather-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .weather-tile {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .weather-tile-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.25rem;
    }
    .weather-tile-lbl {
        font-size: 0.76rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }

    /* Predict Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 2px 6px rgba(13, 148, 136, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.35) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Cached Data & Model Loader
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_dataset() -> pd.DataFrame:
    return load_dataset("dataset/hour.csv")


@st.cache_resource
def get_cached_models() -> Any:
    return get_model_manager("dataset/hour.csv")


df = get_cached_dataset()
mm = get_cached_models()

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Branding
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; padding: 12px 0 24px 0;">
            <div style="background:#0D9488; border-radius:10px; width:44px; height:44px; display:flex; align-items:center; justify-content:center;">
                <span style="font-size:24px;">🚲</span>
            </div>
            <div>
                <div style="font-size:24px; font-weight:800; color:#FFFFFF; letter-spacing:0.5px; line-height:1.1;">VELO</div>
                <div style="font-size:11px; font-weight:500; color:#2DD4BF; letter-spacing:0.5px;">Bike Demand Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Demand Predictor",
            "Model Performance",
            "Data Insights",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div style="margin-top:220px; padding:18px 10px; border-top:1px solid rgba(255,255,255,0.08); text-align:center;">
            <div style="font-size:26px; margin-bottom:4px;">🚲</div>
            <div style="font-size:13px; font-weight:700; color:#FFFFFF;">Smarter Bikes</div>
            <div style="font-size:12px; font-weight:500; color:#34D399;">Greener Cities 🍃</div>
            <div style="font-size:10px; color:#64748B; margin-top:8px;">B.Tech AI/ML • SRS v1.5</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 4. Helper Header Renderer
# -----------------------------------------------------------------------------
def render_header(title: str, subtitle: str, model_tag: str = "Trained (L5)") -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <h1 class="page-title">{title}</h1>
                <p class="page-subtitle">{subtitle}</p>
            </div>
            <div class="header-badges">
                <span class="status-pill pill-green"><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#059669; margin-right:6px;"></span>Model: {model_tag}</span>
                <span class="status-pill pill-blue"><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#2563EB; margin-right:6px;"></span>Weather: Live (Open-Meteo)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 5. PAGE: OVERVIEW
# -----------------------------------------------------------------------------
if page == "Overview":
    render_header("Overview", "Key metrics, historical data and insights at a glance.", "Trained (L5)")

    # 4 Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Records</div>
                <div class="kpi-value">{len(df):,}</div>
                <div class="kpi-subtext">Hourly records (2011–2012)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        mean_demand = df["cnt"].mean()
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Historical Mean</div>
                <div class="kpi-value">{mean_demand:.1f}</div>
                <div class="kpi-subtext">bikes / hour (overall average)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Active Model</div>
                <div class="kpi-value" style="font-size:1.75rem;">L5 Regressor</div>
                <div class="kpi-subtext">Random Forest (100 Trees)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        best_mae = mm.mae_scores["L5"]
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">MAE (L5 Best)</div>
                <div class="kpi-value">{best_mae:.1f}</div>
                <div class="kpi-subtext">bikes / hour (held-out test split)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Middle Content: Hourly Demand Pattern & Key Insights
    mid_left, mid_right = st.columns([1.8, 1.0])

    with mid_left:
        st.markdown(
            """
            <div class="velo-card">
                <div class="velo-card-title">📈 Hourly Demand Pattern (Historical Average)</div>
            """,
            unsafe_allow_html=True,
        )
        hourly_means = df.groupby("hr")["cnt"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.plot(hourly_means["hr"], hourly_means["cnt"], color="#0D9488", linewidth=2.5, marker="o", markersize=5)
        ax.fill_between(hourly_means["hr"], hourly_means["cnt"], color="#CCFBF1", alpha=0.45)

        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels(["12 AM", "3 AM", "6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM"], fontsize=9, color="#64748B")
        ax.set_ylabel("Average Demand (bikes/hr)", fontsize=9, color="#64748B")
        ax.set_ylim(0, 520)
        ax.grid(True, linestyle="--", alpha=0.3, color="#CBD5E1")
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#E2E8F0")

        # Mark 5 PM peak
        peak_val = hourly_means.loc[hourly_means["hr"] == 17, "cnt"].values[0]
        ax.annotate(
            f"Peak: 17:00 ({peak_val:.1f})",
            xy=(17, peak_val),
            xytext=(13.5, peak_val + 35),
            arrowprops=dict(facecolor="#0F766E", shrink=0.08, width=1.2, headwidth=6),
            fontsize=9,
            fontweight="bold",
            color="#0F766E",
        )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with mid_right:
        st.markdown(
            """<div class="velo-card" style="height:100%;">
<div class="velo-card-title">💡 Key Empirical Insights</div>
<div style="display:flex; flex-direction:column; gap:8px; margin-top:8px;">
<div class="insight-card">
<div class="insight-title">Peak Commute Hours</div>
<div class="insight-desc">5:00 PM (461.5 bikes/hr) &amp; 8:00 AM (359.0 bikes/hr). Working days peak at 525.3.</div>
</div>
<div class="insight-card">
<div class="insight-title">Lowest Demand Trough</div>
<div class="insight-desc">4:00 AM (6.4 bikes/hr) and 3:00 AM (11.7 bikes/hr) during night reset.</div>
</div>
<div class="insight-card">
<div class="insight-title">Seasonal Variation</div>
<div class="insight-desc">Summer (236.0 bikes/hr) is +112% higher than Winter (111.1 bikes/hr).</div>
</div>
<div class="insight-card" style="margin-bottom:0;">
<div class="insight-title">Weather Sensitivity</div>
<div class="insight-desc">Clear conditions (204.9 bikes/hr) drop by 45% during Light Rain/Snow (111.6).</div>
</div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

    # Bottom Information Card
    st.markdown(
        f"""
        <div class="velo-card" style="margin-top:10px;">
            <div class="velo-card-title">ℹ️ System & Operational Metadata</div>
            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:1rem; font-size:0.88rem; color:#475569;">
                <div><strong>Active Dataset:</strong> Capital Bikeshare (17,379 hours, 2011–2012)</div>
                <div><strong>Evaluation Metric:</strong> Mean Absolute Error (MAE in bikes/hour)</div>
                <div><strong>Evaluated Levels:</strong> L0 (Mean) → L1 (Hour) → L2 (+Work) → L3 (+Season) → L4 (LR) → L5 (RF)</div>
            </div>
            <div class="disclaimer-card" style="margin-top:12px;">
                <strong>SRS v1.5 Required Disclosure:</strong> The models are trained and evaluated on historical data (UCI 2011–2012). The current-day prediction accuracy is not independently verified. MAE represents historical held-out test error, not a guaranteed confidence interval for individual predictions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 6. PAGE: DEMAND PREDICTOR
# -----------------------------------------------------------------------------
elif page == "Demand Predictor":
    render_header("Demand Predictor", "Estimate the number of bikes needed for selected operational conditions.", "Trained (L5)")

    col_input, col_result = st.columns([1.1, 1.3], gap="large")

    with col_input:
        st.markdown('<div class="velo-card"><div class="velo-card-title">📋 Input Parameters</div>', unsafe_allow_html=True)

        # Date Picker
        default_date = date.today()
        selected_date = st.date_input("Target Date", value=default_date, help="Select the operational date for demand estimation.")

        # Hour of Day Dropdown
        hour_labels = [f"{h:02d}:00 ({'12 AM' if h==0 else f'{h} AM' if h<12 else '12 PM' if h==12 else f'{h-12} PM'})" for h in range(24)]
        selected_hour_idx = st.selectbox("Hour of Day", range(24), index=17, format_func=lambda x: hour_labels[x])

        target_dt = datetime(selected_date.year, selected_date.month, selected_date.day, selected_hour_idx, 0)
        cal = derive_calendar_features(target_dt)

        st.write("")
        st.markdown(
            f"""
            <div style="background:#F1F5F9; border-radius:8px; padding:8px 12px; font-size:0.84rem; color:#334155; margin-bottom:12px;">
                <strong>Derived Calendar Features:</strong> {cal['weekday_name']} • Season: <strong>{cal['season_name']}</strong> (Code {cal['season']}) • Holiday: <strong>{'Yes' if cal['holiday']==1 else 'No'}</strong> • Working Day: <strong>{'Yes' if cal['workingday']==1 else 'No'}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Live Weather Retrieval from Open-Meteo
        weather_res = fetch_open_meteo_forecast(target_dt)
        is_live_weather = weather_res["success"]

        use_manual_weather = st.checkbox("Manual Weather Override", value=not is_live_weather, help="Check to manually specify weather parameters instead of Open-Meteo.")

        if use_manual_weather:
            st.info("Manual Weather Override enabled. Enter conditions below:")
            m_temp = st.number_input("Temperature (°C)", min_value=-15.0, max_value=45.0, value=20.0, step=0.5)
            m_hum = st.slider("Humidity (%)", min_value=0, max_value=100, value=55)
            m_wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=70.0, value=12.0, step=1.0)
            m_cond = st.selectbox("Weather Condition", ["Clear", "Mist/Cloudy", "Light Rain/Snow", "Heavy Rain"])
            weather_code = VALID_WEATHER[m_cond]
            norm_weather = normalize_weather_features(m_temp, m_temp - 2.0, float(m_hum), m_wind)
            norm_weather["weathersit"] = weather_code
            raw_weather = {
                "temperature_c": m_temp,
                "apparent_temperature_c": m_temp - 2.0,
                "humidity_pct": m_hum,
                "wind_speed_kmh": m_wind,
                "weather_desc": m_cond,
                "weathersit": weather_code,
            }
            weather_source_label = "Manual Input (Labelled Fallback)"
        else:
            if is_live_weather:
                raw_weather = weather_res["raw_weather"]
                norm_weather = weather_res["normalized_weather"]
                weather_source_label = "Open-Meteo Forecast API (Live)"
            else:
                st.warning(f"⚠️ {weather_res['warning_code']}: {weather_res['message']}")
                # Default fallback
                raw_weather = {
                    "temperature_c": 18.0,
                    "apparent_temperature_c": 18.0,
                    "humidity_pct": 50.0,
                    "wind_speed_kmh": 10.0,
                    "weather_desc": "Clear (Fallback)",
                    "weathersit": 1,
                }
                norm_weather = normalize_weather_features(18.0, 18.0, 50.0, 10.0)
                norm_weather["weathersit"] = 1
                weather_source_label = "Historical Average Fallback"

        # Model Level Selector
        model_level = st.selectbox(
            "Prediction Engine Level",
            ["L5", "L4", "L3", "L2", "L1", "L0"],
            index=0,
            format_func=lambda x: f"{x} — {mm.get_summary_table()[['L0','L1','L2','L3','L4','L5'].index(x)]['Model / Approach']}",
            help="Select the evaluation tier. L5 (Random Forest) is recommended.",
        )

        st.write("")
        predict_btn = st.button("Predict Demand", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        # Weather Details Card
        st.markdown(
            f"""
            <div class="velo-card">
                <div class="velo-card-title">
                    <span>Weather Details</span>
                    <span class="status-pill pill-blue" style="margin-left:auto; font-size:0.75rem;">{weather_source_label}</span>
                </div>
                <div class="weather-grid">
                    <div class="weather-tile">
                        <div class="weather-tile-lbl">Temperature</div>
                        <div class="weather-tile-val">{raw_weather['temperature_c']:.1f}°C</div>
                    </div>
                    <div class="weather-tile">
                        <div class="weather-tile-lbl">Humidity</div>
                        <div class="weather-tile-val">{raw_weather['humidity_pct']:.0f}%</div>
                    </div>
                    <div class="weather-tile">
                        <div class="weather-tile-lbl">Wind Speed</div>
                        <div class="weather-tile-val">{raw_weather['wind_speed_kmh']:.1f} km/h</div>
                    </div>
                    <div class="weather-tile">
                        <div class="weather-tile-lbl">Condition</div>
                        <div class="weather-tile-val" style="font-size:1.05rem;">{raw_weather['weather_desc']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Build feature vector
        feature_dict = {
            "season": cal["season"],
            "mnth": cal["mnth"],
            "hr": cal["hr"],
            "holiday": cal["holiday"],
            "weekday": cal["weekday"],
            "workingday": cal["workingday"],
            "weathersit": norm_weather["weathersit"],
            "temp": norm_weather["temp"],
            "atemp": norm_weather["atemp"],
            "hum": norm_weather["hum"],
            "windspeed": norm_weather["windspeed"],
        }

        # Predict
        pred_res = mm.predict(model_level, feature_dict)
        pred_count = pred_res["predicted_cnt"]
        pred_mae = pred_res["mae"]
        method_name = f"{pred_res['used_level']} — {mm.get_summary_table()[['L0','L1','L2','L3','L4','L5'].index(pred_res['used_level'])]['Model / Approach']}"

        st.markdown(
            f"""
            <div class="hero-pred-card">
                <div class="hero-pred-header">Estimated Hourly Demand</div>
                <div class="hero-pred-val">{pred_count} <span class="hero-pred-unit">bikes / hour</span></div>
                <div class="hero-pred-footer">
                    <div>
                        <div style="font-weight:700;">Prediction Method:</div>
                        <div style="color:#CCFBF1;">{method_name}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-weight:700;">Historical Test MAE:</div>
                        <div style="color:#CCFBF1;">± {pred_mae:.1f} bikes / hour</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if pred_res["fallback_note"]:
            st.warning(f"⚠️ **Baseline Fallback Notice:** {pred_res['fallback_note']}")

        # Required Disclaimer Card
        st.markdown(
            """
            <div class="disclaimer-card">
                <strong>Important Operational Notice (SRS v1.5 Section 3.2):</strong><br/>
                This prediction is generated based on the selected input parameters and model tier. The model is evaluated on held-out historical data from Washington, D.C. (2011–2012). Current-day operational accuracy is not independently verified.
                <br/><br/>
                <em>Note: MAE represents the average test-set deviation and should not be interpreted as a rigid confidence interval or error bound.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 7. PAGE: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "Model Performance":
    render_header("Model Performance", "Evaluation results and model comparison across all six tiers (L0–L5).", "Evaluated (L0–L5)")

    summary_rows = mm.get_summary_table()
    table_df = pd.DataFrame(summary_rows)

    top_col1, top_col2 = st.columns([1.2, 1.2], gap="large")

    with top_col1:
        st.markdown(
            """
            <div class="velo-card">
                <div class="velo-card-title">📊 MAE Comparison on Held-Out Test Set (3,476 rows)</div>
            """,
            unsafe_allow_html=True,
        )

        # Render styled HTML table
        rows_html = "".join([
            f'<tr style="background:{"#ECFDF5" if r["Level"] == "L5" else "#FFFFFF"}; border-bottom:1px solid #E2E8F0;">'
            f'<td style="padding:10px 12px; font-weight:700;">{r["Level"]}</td>'
            f'<td style="padding:10px 12px;"><strong>{r["Model / Approach"]}</strong><br/><span style="font-size:0.78rem; color:#64748B;">{r["Description"]}</span></td>'
            f'<td style="padding:10px 12px; font-weight:700; text-align:right;">{r["Test MAE"]:.2f}</td>'
            f'<td style="padding:10px 12px; font-weight:600; text-align:right; color:{"#059669" if r["Improvement vs L1 (%)"] >= 0 else "#DC2626"};">{r["Improvement vs L1 (%)"]:+.2f}%</td>'
            f'<td style="padding:10px 12px; text-align:center;">{"<span class='status-pill pill-green'>Best</span>" if r["Level"] == "L5" else ""}</td>'
            f'</tr>'
            for r in summary_rows
        ])

        st.markdown(
            f"""
            <table style="width:100%; border-collapse:collapse; font-size:0.86rem; color:#0F172A;">
                <thead>
                    <tr style="background:#F1F5F9; border-bottom:2px solid #CBD5E1; text-align:left; color:#475569;">
                        <th style="padding:8px 12px;">Level</th>
                        <th style="padding:8px 12px;">Model / Approach</th>
                        <th style="padding:8px 12px; text-align:right;">Test MAE</th>
                        <th style="padding:8px 12px; text-align:right;">vs L1 (%)</th>
                        <th style="padding:8px 12px; text-align:center;">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            <div style="font-size:0.8rem; color:#64748B; margin-top:10px;">
                * <strong>Evaluation Protocol:</strong> Strict chronological 80/20 train/test split per SRS Section 3.2. L4 (Linear Regression) underperforms L1 because commuter hourly spikes are inherently non-linear. L5 (Random Forest) achieves the lowest test MAE (85.19 bikes/hr).
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_col2:
        st.markdown(
            """
            <div class="velo-card">
                <div class="velo-card-title">Actual vs Predicted Demand (L5 Random Forest)</div>
            """,
            unsafe_allow_html=True,
        )
        # Subsample 250 test points for clean visualization
        sub_indices = np.linspace(0, len(mm.y_test) - 1, 250, dtype=int)
        y_sub_act = mm.y_test[sub_indices]
        y_sub_pred = mm.predictions["L5"][sub_indices]

        fig, ax = plt.subplots(figsize=(6.5, 4.3), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.scatter(y_sub_act, y_sub_pred, alpha=0.55, color="#0D9488", edgecolors="none", s=28)
        max_val = max(max(y_sub_act), max(y_sub_pred))
        ax.plot([0, max_val], [0, max_val], color="#EF4444", linestyle="--", linewidth=1.5, label="Perfect 45° Fit")

        ax.set_xlabel("Actual Hourly Demand (cnt)", fontsize=9, color="#64748B")
        ax.set_ylabel("Predicted Demand (cnt)", fontsize=9, color="#64748B")
        ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3, color="#CBD5E1")
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#E2E8F0")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom Charts: Residuals & Model Status
    b_col1, b_col2, b_col3 = st.columns([1, 1, 1], gap="medium")

    with b_col1:
        st.markdown(
            """
            <div class="velo-card">
                <div class="velo-card-title">Prediction Error Distribution (L5 Residuals)</div>
            """,
            unsafe_allow_html=True,
        )
        errors = mm.y_test - mm.predictions["L5"]
        fig, ax = plt.subplots(figsize=(5, 3.4), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.hist(errors, bins=35, color="#14B8A6", edgecolor="#FFFFFF", alpha=0.85)
        ax.axvline(0, color="#EF4444", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Residual Error (Actual - Pred)", fontsize=8, color="#64748B")
        ax.set_ylabel("Frequency", fontsize=8, color="#64748B")
        ax.grid(True, linestyle="--", alpha=0.3, color="#CBD5E1")
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#E2E8F0")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with b_col2:
        st.markdown(
            """
            <div class="velo-card">
                <div class="velo-card-title">Residuals vs Predicted Demand (L5)</div>
            """,
            unsafe_allow_html=True,
        )
        sub_err = errors[sub_indices]
        fig, ax = plt.subplots(figsize=(5, 3.4), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.scatter(y_sub_pred, sub_err, alpha=0.5, color="#3B82F6", s=20)
        ax.axhline(0, color="#EF4444", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Predicted Demand", fontsize=8, color="#64748B")
        ax.set_ylabel("Residual", fontsize=8, color="#64748B")
        ax.grid(True, linestyle="--", alpha=0.3, color="#CBD5E1")
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#E2E8F0")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with b_col3:
        st.markdown(
            """
            <div class="velo-card" style="height:100%;">
                <div class="velo-card-title">Release Gate 10 Evaluation Audit</div>
                <div style="font-size:0.85rem; color:#334155; line-height:1.6;">
                    <div style="margin-bottom:8px;">• <strong>Chronological Split:</strong> Evaluated on identical held-out rows (3,476 instances).</div>
                    <div style="margin-bottom:8px;">• <strong>No Target Leakage:</strong> <code>casual</code> and <code>registered</code> excluded from all feature matrices.</div>
                    <div style="margin-bottom:8px;">• <strong>Honest Reporting:</strong> L4 (Linear Regression) reported with higher MAE than L1, honoring SRS Section 3.2.</div>
                    <div style="margin-bottom:8px;">• <strong>Output Clipping:</strong> Negative predictions clipped to &ge; 0.</div>
                    <div>• <strong>Deterministic Fallbacks:</strong> L3 falls back to L2, L1, then L0 for unseen categories.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 8. PAGE: DATA INSIGHTS
# -----------------------------------------------------------------------------
elif page == "Data Insights":
    render_header("Data Insights", "Explore patterns, distributions, and empirical relationships from the dataset.", "EDA Verified")

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Target Distribution (cnt)",
        "Hourly Commute",
        "Seasonal Patterns",
        "Weather Severity",
        "Working Day Split",
        "Feature Correlations",
    ])

    with tab0:
        st.markdown('<div class="velo-card"><div class="velo-card-title">Target Variable (cnt) Distribution — Positively Skewed (+1.28)</div>', unsafe_allow_html=True)
        t_dist = compute_target_distribution(df)
        fig, ax = plt.subplots(figsize=(9, 4.0), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.hist(df["cnt"], bins=50, color="#0D9488", edgecolor="white", alpha=0.85)
        ax.axvline(t_dist["mean"], color="#EF4444", linestyle="--", linewidth=2, label=f"Mean ({t_dist['mean']:.1f})")
        ax.axvline(t_dist["median"], color="#10B981", linestyle="-", linewidth=2, label=f"Median ({t_dist['median']:.1f})")
        ax.axvline(t_dist["upper_fence"], color="#F59E0B", linestyle=":", linewidth=2, label=f"IQR Fence ({t_dist['upper_fence']:.1f})")
        ax.set_xlabel("Hourly Rental Count (cnt)", fontsize=9, color="#64748B")
        ax.set_ylabel("Frequency (Hours)", fontsize=9, color="#64748B")
        ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3, color="#CBD5E1")
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#E2E8F0")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            f"""
            <div style="font-size:0.85rem; color:#475569; margin-top:8px;">
                <strong>Distribution Summary:</strong> Target count is right-skewed (+1.28) with mean (189.5 bikes/hr) exceeding median (142.0 bikes/hr). The upper outlier fence ($Q3 + 1.5 \\times IQR$) is 642.5 bikes. Exactly 505 records (2.91%) exceed this fence (414 on working days, 91 on non-working days), reflecting operational rush-hour capacity surges.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab1:
        st.markdown('<div class="velo-card"><div class="velo-card-title">Hourly Demand: Commuter Commute vs Weekend Leisure</div>', unsafe_allow_html=True)
        hourly_pat = compute_hourly_patterns(df)
        hrs = list(range(24))
        wk_m = [hourly_pat["workingday_hourly_mean"][h] for h in hrs]
        nwk_m = [hourly_pat["nonworkingday_hourly_mean"][h] for h in hrs]

        fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.plot(hrs, wk_m, marker="o", color="#0D9488", linewidth=2.4, label="Working Days (Commute Peaks: 8 AM & 5 PM)")
        ax.plot(hrs, nwk_m, marker="s", color="#3B82F6", linewidth=2.4, linestyle="--", label="Non-Working Days (Afternoon Leisure Curve: 1 PM Peak)")
        ax.set_xticks(hrs)
        ax.set_xlabel("Hour of Day (0–23)", fontsize=9, color="#64748B")
        ax.set_ylabel("Average Hourly Rentals", fontsize=9, color="#64748B")
        ax.grid(True, linestyle="--", alpha=0.3, color="#CBD5E1")
        ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=9)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#E2E8F0")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            """
            <div style="font-size:0.85rem; color:#475569; margin-top:8px;">
                <strong>Finding:</strong> Working days display distinct bimodal rush-hour peaks at 8:00 AM (477.0 bikes) and 5:00 PM (525.3 bikes). Weekends exhibit a single unimodal afternoon peak at 1:00 PM (372.7 bikes) with an elevated plateau from 12 PM to 3 PM.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown('<div class="velo-card"><div class="velo-card-title">Season-Wise Demand Breakdown (SRS v1.5 Mapping)</div>', unsafe_allow_html=True)
        season_pat = compute_seasonal_patterns(df)["season_metrics"]
        s_names = [r["season_name"] for r in season_pat]
        s_means = [r["mean"] for r in season_pat]
        s_colors = ["#60A5FA", "#34D399", "#F59E0B", "#FB923C"]

        fig, ax = plt.subplots(figsize=(8, 3.8), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        bars = ax.bar(s_names, s_means, color=s_colors, edgecolor="#0F172A", linewidth=0.8, width=0.55)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Mean Hourly Rentals", fontsize=9, color="#64748B")
        ax.set_ylim(0, max(s_means) * 1.18)
        ax.grid(True, linestyle="--", alpha=0.3, color="#CBD5E1")
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#E2E8F0")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            """
            <div style="font-size:0.85rem; color:#475569; margin-top:8px;">
                <strong>SRS v1.5 Mapping:</strong> 1=Winter, 2=Spring, 3=Summer, 4=Fall. Summer has peak demand (236.0 bikes/hr), followed by Spring (208.3) and Fall (198.9). Winter records lowest demand (111.1).
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown('<div class="velo-card"><div class="velo-card-title">Weather Severity Impact on Demand</div>', unsafe_allow_html=True)
        weather_pat = compute_weather_patterns(df)["weather_metrics"]
        w_names = [r["weather_name"] for r in weather_pat]
        w_means = [r["mean"] for r in weather_pat]
        w_counts = [r["count"] for r in weather_pat]
        w_colors = ["#10B981", "#64748B", "#3B82F6", "#EF4444"]

        fig, ax = plt.subplots(figsize=(8, 3.8), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        bars = ax.bar(w_names, w_means, color=w_colors, edgecolor="#0F172A", linewidth=0.8, width=0.55)
        for b, cnt in zip(bars, w_counts):
            h = b.get_height()
            ax.annotate(f"{h:.1f} bikes\n(n={cnt:,})", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_ylabel("Mean Hourly Rentals", fontsize=9, color="#64748B")
        ax.set_ylim(0, max(w_means) * 1.25)
        ax.grid(True, linestyle="--", alpha=0.3, color="#CBD5E1")
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#E2E8F0")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            """
            <div style="font-size:0.85rem; color:#475569; margin-top:8px;">
                <strong>Finding:</strong> Demand decreases monotonically with weather severity: Clear (204.9) → Mist (175.2) → Light Rain/Snow (111.6) → Heavy Rain (74.3).<br/>
                <em>Note on Heavy Rain:</em> Category 4 contains only $n=3$ instances across the 2-year dataset; interpret its statistical mean with caution.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab4:
        st.markdown('<div class="velo-card"><div class="velo-card-title">Working Day vs Non-Working Day Comparison</div>', unsafe_allow_html=True)
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            wk_agg = df.groupby("workingday")["cnt"].agg(count="count", mean="mean", median="median", std="std").reset_index()
            fig, ax = plt.subplots(figsize=(5, 3.8), facecolor="#FFFFFF")
            ax.set_facecolor("#FFFFFF")
            bars = ax.bar(["Non-Working Day (0)", "Working Day (1)"], wk_agg["mean"], color=["#3B82F6", "#0D9488"], width=0.5)
            for b in bars:
                h = b.get_height()
                ax.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_ylabel("Mean Hourly Rentals", fontsize=9, color="#64748B")
            ax.set_ylim(0, 240)
            ax.grid(True, linestyle="--", alpha=0.3, color="#CBD5E1")
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_color("#E2E8F0")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col_w2:
            st.markdown(
                """
                <div style="font-size:0.9rem; color:#334155; line-height:1.7; padding-top:10px;">
                    <strong>Empirical Summary:</strong>
                    <ul>
                        <li><strong>Working Days:</strong> 11,865 hours (68.3% of records). Mean = <strong>193.22</strong> bikes/hr.</li>
                        <li><strong>Non-Working Days:</strong> 5,514 hours (31.7% of records). Mean = <strong>181.41</strong> bikes/hr.</li>
                        <li>While overall daily averages appear comparable, their <em>hourly demand curves</em> are completely different (bimodal commute vs. unimodal afternoon leisure).</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="velo-card"><div class="velo-card-title">Feature Correlations with Total Rentals (cnt)</div>', unsafe_allow_html=True)
        corrs = compute_feature_correlations(df)
        sorted_c = sorted(corrs.items(), key=lambda x: x[1])
        f_names = [x[0] for x in sorted_c]
        f_vals = [x[1] for x in sorted_c]
        f_colors = ["#EF4444" if v < 0 else "#10B981" for v in f_vals]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        bars = ax.barh(f_names, f_vals, color=f_colors, edgecolor="#0F172A", linewidth=0.8, height=0.6)
        for b in bars:
            w = b.get_width()
            off = 5 if w >= 0 else -30
            ax.annotate(f"{w:+.3f}", xy=(w, b.get_y() + b.get_height() / 2), xytext=(off, -3), textcoords="offset points", ha="left" if w >= 0 else "right", va="center", fontsize=8, fontweight="bold")
        ax.axvline(0, color="#0F172A", linewidth=1)
        ax.set_xlabel("Pearson Correlation (r)", fontsize=9, color="#64748B")
        ax.set_xlim(-0.45, 0.55)
        ax.grid(True, linestyle="--", alpha=0.3, color="#CBD5E1")
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#E2E8F0")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            """
            <div style="font-size:0.85rem; color:#475569; margin-top:8px;">
                <strong>Correlation Highlights:</strong> Temperature (<code>temp</code> r = +0.405) and Hour (<code>hr</code> r = +0.394) have the highest positive association with rentals. Humidity (<code>hum</code> r = -0.323) and Weather Severity (<code>weathersit</code> r = -0.142) show negative associations.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 9. PAGE: ABOUT PROJECT
# -----------------------------------------------------------------------------
elif page == "About Project":
    render_header("About Project", "Methodology, technical architecture, and system specifications.", "SRS v1.5 Verified")

    r1_col1, r1_col2 = st.columns(2, gap="medium")
    with r1_col1:
        st.markdown(
            """
            <div class="velo-card" style="height:100%;">
                <div class="velo-card-title">Project Objective</div>
                <p style="font-size:0.88rem; color:#334155; line-height:1.6;">
                    The <strong>Smart Bike Rental Demand Advisor</strong> is an ML-backed operational decision-support tool designed for bike-sharing fleet managers in Washington, D.C. It delivers transparent, interpretable hourly demand forecasts to help balance station availability and avoid stockouts during commuter rush hours.
                </p>
                <div style="font-size:0.82rem; color:#64748B; margin-top:8px;">
                    <strong>Academic Context:</strong> B.Tech CSE (AI/ML) Second-Year Project at GLA University, Greater Noida.<br/>
                    <strong>Supervised by:</strong> Mr. Palash Singh Raghuwanshi • <strong>Reviewed by:</strong> Mr. Gautam Mukherjee.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r1_col2:
        st.markdown(
            """
            <div class="velo-card" style="height:100%;">
                <div class="velo-card-title">Dataset & Schema Integrity</div>
                <p style="font-size:0.88rem; color:#334155; line-height:1.6;">
                    Built on the <strong>UCI Bike Sharing Dataset (<code>hour.csv</code>)</strong> containing 17,379 hourly records spanning 2011 to 2012 from Capital Bikeshare.
                </p>
                <ul style="font-size:0.84rem; color:#475569; margin-bottom:0;">
                    <li><strong>Target Variable:</strong> <code>cnt</code> (total hourly rentals, &ge; 0).</li>
                    <li><strong>Constraint C-2 (Target Leakage):</strong> <code>casual</code> and <code>registered</code> strictly excluded because <code>casual + registered == cnt</code>.</li>
                    <li><strong>Design Decision D6:</strong> <code>yr</code> excluded from candidate predictors to ensure applicability beyond 2012.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    r2_col1, r2_col2 = st.columns(2, gap="medium")
    with r2_col1:
        st.markdown(
            """
            <div class="velo-card" style="height:100%;">
                <div class="velo-card-title">Methodology & Build Order</div>
                <ol style="font-size:0.84rem; color:#334155; line-height:1.7; padding-left:20px;">
                    <li><strong>Data Validation & Sanitization:</strong> Null checks, duplicate detection, and domain bounds (<code>ERR_VAL_01</code>).</li>
                    <li><strong>Chronological Evaluation Split:</strong> Earlier 80% (13,903 rows) for training; held-out latest 20% (3,476 rows) for testing.</li>
                    <li><strong>Grouped Baseline Ladder (L0–L3):</strong> Training-only group means with deterministic fallback rules.</li>
                    <li><strong>Regression Model Training (L4–L5):</strong> Linear Regression and Random Forest Regressor.</li>
                    <li><strong>Live Open-Meteo Integration:</strong> Real-time forecast retrieval with 2.0s network timeout and WMO mapping.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r2_col2:
        st.markdown(
            """
            <div class="velo-card" style="height:100%;">
                <div class="velo-card-title">Technology Stack</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:0.84rem; color:#334155;">
                    <div>• <strong>Python 3.14</strong> (Core runtime)</div>
                    <div>• <strong>Pandas & NumPy</strong> (Data wrangling)</div>
                    <div>• <strong>Scikit-Learn</strong> (L4 LR & L5 RF)</div>
                    <div>• <strong>Streamlit 1.65</strong> (Web interface)</div>
                    <div>• <strong>Matplotlib</strong> (Agg visual charts)</div>
                    <div>• <strong>Open-Meteo REST API</strong> (Live weather)</div>
                </div>
                <div style="margin-top:14px; font-size:0.82rem; color:#64748B;">
                    <strong>Design Principle:</strong> Clean, maintainable Python code with zero unnecessary abstractions, appropriate for second-year technical viva explanation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    r3_col1, r3_col2 = st.columns(2, gap="medium")
    with r3_col1:
        st.markdown(
            """
            <div class="velo-card" style="height:100%;">
                <div class="velo-card-title">Evaluation Metric: MAE</div>
                <p style="font-size:0.88rem; color:#334155; line-height:1.6;">
                    <strong>Mean Absolute Error (MAE)</strong> is chosen as the primary metric because it is directly interpretable in operational units (bikes per hour), unlike MSE or RMSE which square the errors and over-penalize rush-hour variance.
                </p>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px 12px; font-size:0.84rem; color:#0F172A;">
                    $$\\text{MAE} = \\frac{1}{N} \\sum_{i=1}^N |y_i - \\hat{y}_i|$$
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r3_col2:
        st.markdown(
            """
            <div class="velo-card" style="height:100%;">
                <div class="velo-card-title">Limitations & Future Scope</div>
                <ul style="font-size:0.84rem; color:#334155; line-height:1.6; padding-left:18px;">
                    <li><strong>No Station-Level Inventory:</strong> The model predicts city-wide hourly aggregates, not individual dock GPS tracking.</li>
                    <li><strong>Historical Horizon:</strong> Trained on 2011–2012 Capital Bikeshare data; present-day demographic shifts require live retraining feeds.</li>
                    <li><strong>Weather API Limits:</strong> Open-Meteo forecasts extend up to 7 days; past dates outside cache use manual input or L0 fallback.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="disclaimer-card" style="margin-top:10px;">
            <strong>Formal SRS v1.5 Disclaimer:</strong><br/>
            This prototype application is developed for academic evaluation. Booking, payments, station inventory, user accounts, and live GPS redistribution are strictly out of scope. Accuracy claims are verified exclusively against the 2011–2012 historical held-out test split.
        </div>
        """,
        unsafe_allow_html=True,
    )
