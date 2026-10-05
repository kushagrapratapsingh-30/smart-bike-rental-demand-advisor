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

# Configure page layout and metadata
st.set_page_config(
    page_title="Smart Bike Rental Demand Advisor",
    page_icon="🚲",
    layout="centered"
)

# App Header
st.title("🚲 Smart Bike Rental Demand Advisor")
st.caption("Sprint I Prototype — Input & Baseline Foundation (11–27 September 2026)")

st.markdown(
    """
    Welcome to the **Smart Bike Rental Demand Advisor** decision-support interface.
    This prototype verifies the end-to-end data pipeline, UI input capture, server-side validation,
    and **Level 0 Baseline / Fallback** estimation mechanism.
    """
)

st.divider()

# Sidebar: System Status & Architecture Overview
with st.sidebar:
    st.header("System Status")
    st.info("**Sprint Phase:** Sprint I (Foundation)")
    st.write("**Model Engine:** Level 0 Baseline")
    st.write("**Active Dataset:** `dataset/hour.csv` (17,379 records)")
    st.divider()
    st.subheader("About Level 0 Baseline")
    st.markdown(
        """
        **Level 0** predicts the overall historical mean demand:
        - **Global Mean:** ~189 bikes / hour
        - **Baseline MAE:** ~142 bikes
        - **Behavior:** By design, Level 0 produces a stable global mean estimate regardless of condition combinations.
        - **Next Phase:** Sprint II introduces feature engineering and regression models (Decision Tree, Random Forest) to beat this baseline.
        """
    )

# Section: Input Selection
st.subheader("1. Enter Operational & Environmental Conditions")

col1, col2 = st.columns(2)

with col1:
    # Allow input range with boundaries to demonstrate TC-02 validation in viva / audit
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

# Prediction Trigger
predict_clicked = st.button("🚴 Predict Demand", type="primary", use_container_width=True)

if predict_clicked:
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
        st.warning("Prediction halted. Please adjust the input values to fall within operational boundaries.")
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
                - **Request Parameters:** Hour `{clean_inputs['hour']}`, Season `{clean_inputs['season']}`, Weather `{clean_inputs['weather']}`, Working Day `{clean_inputs['working_day']}`
                - **Baseline Method:** `{prediction_result['baseline_type']}`
                - **Exact Historical Mean:** `{prediction_result['predicted_cnt_raw']:.2f}` bikes
                - **Note:** {prediction_result['note']}
                """
            )
