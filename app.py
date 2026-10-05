"""
Smart Bike Rental Demand Advisor
Streamlit Application — Sprint I MVP Prototype (11–27 September 2026)

Integrates:
- Operational input interface (FR-1.1)
- Server-side input validation and boundary enforcement (FR-1.2, NFR-2)
- Level 0 Baseline prediction and fallback system (FR-2.2, FR-2.4)
"""

import streamlit as st
from src.baseline import get_baseline_instance
from src.validation import validate_inputs, sanitize_inputs

# Configure page layout and metadata (sidebar collapsed)
st.set_page_config(
    page_title="Smart Bike Rental Demand Advisor",
    page_icon="🚲",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS to completely remove sidebar and increase typography / font sizes in the main view
st.markdown(
    """
    <style>
    /* Hide sidebar and sidebar toggle button completely */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Main container spacing and max width */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 860px !important;
    }

    /* Main Title Styling */
    h1 {
        font-size: 2.7rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 0.4rem !important;
    }

    /* Subheadings */
    h2, h3 {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.8rem !important;
    }

    /* General Paragraph & Descriptive Text */
    p, .stMarkdown p {
        font-size: 1.15rem !important;
        line-height: 1.65 !important;
    }

    /* Widget Labels (Hour, Season, Weather, Working Day) */
    [data-testid="stWidgetLabel"] p {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.35rem !important;
    }

    /* Input Controls (selectboxes, number inputs) text size */
    .stSelectbox div[data-baseweb="select"] div, 
    .stNumberInput input {
        font-size: 1.15rem !important;
    }

    /* Primary Action Button */
    button[kind="primary"] {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.5rem !important;
        border-radius: 8px !important;
        margin-top: 0.5rem !important;
    }

    /* Metrics Values & Labels */
    [data-testid="stMetricValue"] {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        color: #ff4b4b !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #d1d5db !important;
    }

    /* Alert / Banner text */
    [data-testid="stAlert"] p {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
    }

    /* Tag / Badge for Sprint indicator */
    .sprint-badge {
        display: inline-block;
        background-color: rgba(255, 75, 75, 0.15);
        color: #ff6b6b;
        border: 1px solid rgba(255, 75, 75, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Header
st.markdown('<span class="sprint-badge">Sprint I Deliverable • Input & Level 0 Baseline</span>', unsafe_allow_html=True)
st.title("🚲 Smart Bike Rental Demand Advisor")
st.caption("Decision-support system for bike-sharing fleet demand estimation (UCI Dataset: 17,379 records)")

st.markdown(
    """
    Enter the operational and environmental conditions below to validate parameters and obtain 
    the demand estimation. During **Sprint I**, the system operates on the **Level 0 Baseline** 
    (historical training mean benchmark), with full regression models arriving in Sprint II.
    """
)

st.divider()

# Section: Input Selection
st.subheader("1. Operational & Environmental Conditions")

col1, col2 = st.columns(2, gap="large")

with col1:
    # Allow input range with bounds to demonstrate TC-02 validation in viva / audit
    hour_input = st.number_input(
        "Hour of Day (0 to 23)",
        min_value=-5,
        max_value=30,
        value=17,
        step=1,
        help="Valid operational hours are integers from 0 to 23 (e.g., 17 for 5:00 PM)."
    )

    season_input = st.selectbox(
        "Season",
        options=["Spring", "Summer", "Fall", "Winter"],
        index=1,
        help="Select the current season."
    )

with col2:
    weather_input = st.selectbox(
        "Weather Severity",
        options=["Clear", "Mist/Cloudy", "Light Rain/Snow", "Heavy Rain"],
        index=0,
        help="Select the prevailing weather conditions."
    )

    working_day_input = st.selectbox(
        "Working Day",
        options=["Yes", "No"],
        index=0,
        help="Is the date an operational working day (non-holiday weekday)?"
    )

# Prediction Trigger Button
predict_clicked = st.button("🚴 Predict Demand", type="primary", use_container_width=True)

if predict_clicked:
    st.divider()
    st.subheader("2. Validation & Demand Estimation")

    # Step 1: Input Validation (FR-1.2, NFR-2)
    is_valid, validation_msg = validate_inputs(
        hour=hour_input,
        season=season_input,
        weather=weather_input,
        working_day=working_day_input
    )

    if not is_valid:
        # Halt execution and show validation error message as required by FR-1.2 & Appendix B
        st.error(f"❌ Input Validation Error:\n\n`{validation_msg}`")
        st.warning("Execution halted. Please provide parameters strictly within operational bounds.")
    else:
        st.success("✅ Inputs validated successfully.")

        # Sanitize parameters
        clean_inputs = sanitize_inputs(
            hour=hour_input,
            season=season_input,
            weather=weather_input,
            working_day=working_day_input
        )

        # Step 2: Fallback Notification (FR-2.4 & Appendix B)
        st.warning(
            "⚠️ **WARN_SYS_01:** Pre-trained model artifact missing. Loading **Level 0 baseline**."
        )

        # Step 3: Compute Level 0 Estimate (FR-2.2, FR-2.4)
        baseline = get_baseline_instance()
        prediction_result = baseline.predict(clean_inputs)

        # Step 4: Display Output Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(
                label="Estimated Bike Demand",
                value=f"{prediction_result['predicted_cnt']} bikes"
            )
        with metric_col2:
            st.metric(
                label="Baseline MAE Margin",
                value=f"± {prediction_result['mae']} bikes"
            )
        with metric_col3:
            st.metric(
                label="Prediction Method",
                value="Level 0 Mean"
            )

        # Technical Details & Academic Explanation
        with st.expander("📋 Technical Request & Baseline Details", expanded=True):
            st.markdown(
                f"""
                - **Validated Parameters:** Hour `{clean_inputs['hour']}`, Season `{clean_inputs['season']}`, Weather `{clean_inputs['weather']}`, Working Day `{clean_inputs['working_day']}`
                - **Baseline Method:** `{prediction_result['baseline_type']}`
                - **Exact Historical Mean:** `{prediction_result['predicted_cnt_raw']:.2f}` bikes
                - **Baseline MAE:** `{prediction_result['mae']}` bikes
                - **Note:** {prediction_result['note']}
                """
            )
