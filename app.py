"""
Smart Bike Rental Demand Advisor
Streamlit application for hourly bike rental demand estimation.
"""

import streamlit as st
from src.baseline import get_baseline_instance
from src.validation import validate_inputs, sanitize_inputs

st.set_page_config(
    page_title="Smart Bike Rental Demand Advisor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom dashboard styling
st.markdown(
    """
    <style>
    /* Completely hide sidebar and collapsed toggle */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Full-width container with responsive breathing room */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1400px !important;
    }

    /* Hero Banner Card */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }

    .hero-title {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 !important;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .hero-subtitle {
        font-size: 1.15rem !important;
        color: #94a3b8 !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.8rem !important;
    }

    .badge-container {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }

    .badge {
        display: inline-block;
        font-size: 0.88rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
    }

    .badge-primary {
        background: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .badge-info {
        background: rgba(59, 130, 246, 0.18);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }

    .badge-success {
        background: rgba(34, 197, 94, 0.18);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }

    /* Section Card */
    .section-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .section-heading {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
        margin-bottom: 1.2rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Form Widget Labels */
    [data-testid="stWidgetLabel"] p {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        margin-bottom: 0.3rem !important;
    }

    /* Form Inputs */
    .stSelectbox div[data-baseweb="select"] div,
    .stNumberInput input {
        font-size: 1.15rem !important;
        border-radius: 8px !important;
    }

    /* Predict Button */
    button[kind="primary"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        padding: 0.8rem 2rem !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.5) !important;
    }

    /* Metric Card Styling */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.7rem !important;
        font-weight: 800 !important;
        color: #f87171 !important;
    }

    [data-testid="stMetricLabel"] p {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }

    /* Alerts */
    [data-testid="stAlert"] p {
        font-size: 1.12rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hero Header Banner across the full page width
st.markdown(
    """
    <div class="hero-card">
        <h1 class="hero-title">🚲 Smart Bike Rental Demand Advisor</h1>
        <p class="hero-subtitle">
            ML-backed operational decision-support system for hourly bike-sharing fleet demand estimation.
        </p>
        <div class="badge-container">
            <span class="badge badge-primary">Sprint I MVP: Input & Baseline Foundation</span>
            <span class="badge badge-info">Active Dataset: UCI hour.csv (17,379 records)</span>
            <span class="badge badge-success">Target Variable: cnt (Hourly Total)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Input controls
st.markdown('<div class="section-heading">⚙️ 1. Operational & Environmental Conditions</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    hour_input = st.number_input(
        "Hour of Day (0 to 23)",
        min_value=-5,
        max_value=30,
        value=17,
        step=1,
        help="Valid operational hours are integers from 0 to 23 (e.g., 17 for 5:00 PM)."
    )

with col2:
    season_input = st.selectbox(
        "Season",
        options=["Winter", "Spring", "Summer", "Fall"],
        index=0,
        help="Select the operational season (1: Winter, 2: Spring, 3: Summer, 4: Fall per SRS v1.5)."
    )

with col3:
    weather_input = st.selectbox(
        "Weather Severity",
        options=["Clear", "Mist/Cloudy", "Light Rain/Snow", "Heavy Rain"],
        index=0,
        help="Select prevailing weather conditions."
    )

with col4:
    working_day_input = st.selectbox(
        "Working Day",
        options=["Yes", "No"],
        index=0,
        help="Operational working day (non-holiday weekday)?"
    )

st.write("")
predict_clicked = st.button("🚴 Predict Hourly Demand", type="primary", use_container_width=True)

# Validation and estimation output
if predict_clicked:
    st.write("")
    st.markdown('<div class="section-heading">📊 2. Validation & Demand Estimation Results</div>', unsafe_allow_html=True)

    is_valid, validation_msg = validate_inputs(
        hour=hour_input,
        season=season_input,
        weather=weather_input,
        working_day=working_day_input
    )

    if not is_valid:
        st.error(f"❌ **Input Validation Error:** `{validation_msg}`")
        st.warning("Execution halted before baseline/model invocation. Please adjust inputs to valid domain ranges.")
    else:
        st.success("✅ **Input Sanity Check Passed:** All operational parameters strictly within domain bounds.")

        clean_inputs = sanitize_inputs(
            hour=hour_input,
            season=season_input,
            weather=weather_input,
            working_day=working_day_input
        )

        st.warning("⚠️ **WARN_SYS_01:** Model artifact unavailable; showing Level 0 fallback.")

        baseline = get_baseline_instance()
        prediction_result = baseline.predict(clean_inputs)

        m1, m2, m3, m4 = st.columns(4, gap="medium")
        with m1:
            st.metric(
                label="Estimated Hourly Demand",
                value=f"{prediction_result['predicted_cnt']} bikes"
            )
        with m2:
            st.metric(
                label="Baseline MAE Margin",
                value=f"± {prediction_result['mae']} bikes"
            )
        with m3:
            st.metric(
                label="Estimation Method",
                value="Level 0 Mean"
            )
        with m4:
            st.metric(
                label="Historical Baseline Mean",
                value=f"{prediction_result['predicted_cnt_raw']:.1f} bikes"
            )

        st.write("")
        with st.expander("📋 Sprint-I Technical Architecture & Level 0 Details", expanded=True):
            det_col1, det_col2 = st.columns(2, gap="large")
            with det_col1:
                st.markdown(
                    f"""
                    **Validated Request Vector:**
                    - **Hour:** `{clean_inputs['hour']}:00`
                    - **Season:** `{clean_inputs['season']}` (Code: `{clean_inputs['season_code']}`)
                    - **Weather:** `{clean_inputs['weather']}` (Code: `{clean_inputs['weather_code']}`)
                    - **Working Day:** `{clean_inputs['working_day']}` (Flag: `{clean_inputs['working_day_flag']}`)
                    """
                )
            with det_col2:
                st.markdown(
                    fr"""
                    **Level 0 Baseline Principles:**
                    - **Benchmark Nature:** Predicts overall training dataset mean ($\mu = 189.46$).
                    - **Input Invariance:** Intentionally ignores condition parameters to establish the lowest performance bar.
                    - **Sprint II Goal:** Train Linear Regression (L4) and Random Forest (L5) models to evaluate on chronological split.
                    """
                )
