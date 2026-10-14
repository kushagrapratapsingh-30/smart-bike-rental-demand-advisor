# Smart Bike Rental Demand Advisor

A machine learning-backed decision-support application for estimating hourly bike rental demand in Washington, D.C., built using the UCI Bike Sharing Dataset and Streamlit.

Based on Software Requirement Specification (**SRS v1.5**).

---

## Sprint 1A, 1B & 1C Verified Milestones

### 1. Dataset Integrity & Empirical Verification (`hour.csv`)
* **Source:** UCI Bike Sharing Dataset (`dataset/hour.csv`, Capital Bikeshare, Washington, D.C., 2011–2012).
* **Record Count & Schema:** 17,379 hourly records across 17 columns (the UCI repository listing header states 17,389; empirical verification of the supplied `hour.csv` confirms exactly 17,379 records).
* **Unit of Observation:** One row represents **one hour of bike-sharing activity**, not an individual bike trip or user.
* **Missing Values & Duplicates:** 0 missing values, 0 duplicate records.
* **Target Variable (`cnt`) Statistics:**
  * Mean: 189.46 bikes/hour
  * Median: 142.00 bikes/hour
  * Standard Deviation: 181.39 bikes/hour
  * Skewness: +1.28 (positively skewed distribution)
  * Range: Min 1, Max 977 (no negative rental counts)
* **Target Leakage Audit (Constraint C-2):**
  * `casual + registered == cnt` verified for 100% of rows (0 mismatches).
  * `casual` and `registered` strictly excluded as model predictors.

### 2. Season Mapping & Astronomical Transition Verification
To maintain full traceability across requirements, data sources, and the application:
* **Original UCI Documentation (`Readme.txt`):** The repository README documents the `season` column with labels `1:springer, 2:summer, 3:fall, 4:winter`.
* **Submitted SRS v1.5 Specification:** Section 4.4, Appendix A, and Release Gate 13 explicitly designate `1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall`. Section 4.4 directs that season boundaries be verified from the actual `dteday` records rather than English documentation labels alone. The calendar dates in `hour.csv` transition around astronomical boundaries:
  * Code `1`: Jan 1 – Mar 20, and Dec 21 – Dec 31 (Winter solstice boundary: Dec 21).
  * Code `2`: Mar 21 – Jun 20 (Spring equinox boundary: Mar 21).
  * Code `3`: Jun 21 – Sep 22 (Summer solstice boundary: Jun 21).
  * Code `4`: Sep 23 – Dec 20 (Autumnal equinox boundary: Sep 23).
* **Application Mapping (`src/validation.py` & `app.py`):** The application follows the SRS v1.5 contract:
  `{"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}`.
  Selecting a season in the UI encodes the corresponding integer (e.g., "Winter" $\rightarrow 1$, "Spring" $\rightarrow 2$), matching the SRS training schema while preserving the raw dataset values immutable.

### 3. Data Validation & Boundary Rules (Sprint 1B)
* **Operational Hour Bounds (`ERR_VAL_01`):** Hours strictly bounded within integers `0` to `23`.
* **Dataset Schema Validation:** Programmatic DataFrame validation for column presence, data types, nulls, negative counts, domain categories, and target sum consistency.
* **Record-Level Validation:** Single-row sanity checks for individual query and observation vectors.
* **System Fallback (`WARN_SYS_01`):** Displays `"Model artifact unavailable; showing Level 0 fallback."` when pre-trained models are unavailable.
* **Level 0 Baseline:** Predicts historical global mean ($\mu = 189.46 \approx 189$ bikes) with an unconditioned baseline MAE of 142.4 bikes.

### 4. Exploratory Data Analysis — Statistical Deliverables (Sprint 1C)
Statistical demand profiling and data-quality checks implemented in `src/eda.py` (visual charts postponed to later visualization deliverable):
* **Target Distribution & Outlier Audit:**
  * Target `cnt` is right-skewed (+1.28). Mean (189.46) exceeds median (142.00) due to rush-hour commute surges.
  * Q1 = 40.0, Q3 = 281.0, IQR = 241.0, Upper fence = 642.5.
  * 505 records (2.91%) exceed 642.5 (max 977). Verified empirically: 414 records (82.0%) occur on working days (concentrated around 8 AM, 5 PM, 6 PM) and 91 records (18.0%) on non-working days (concentrated around 12 PM-5 PM). These represent likely demand surges rather than proven individual records and are retained pending further validation to avoid prematurely discarding high-capacity operational signals.

* **Hourly Demand Profiling:**
  * Working days exhibit distinct bimodal commute spikes at 8:00 AM (mean 477.0 bikes) and 5:00 PM (mean 525.3 bikes).
  * Non-working days (weekends/holidays) show an afternoon demand peak at 1:00 PM (mean 372.7 bikes/hour), with a broad high-demand plateau from 12:00 PM to 3:00 PM (approximately 359–373 bikes/hour).
* **Season-Wise Demand Breakdown (SRS v1.5 Mapping):**
  * Summer exhibits peak demand (mean 236.0 bikes), followed by Spring (208.3) and Fall (198.9).
  * Winter records the lowest demand (mean 111.1 bikes).
* **Weather Severity Impact:**
  * Demand declines monotonically with weather severity: Clear (204.9 bikes) -> Mist/Cloudy (175.2 bikes) -> Light Rain/Snow (111.6 bikes) -> Heavy Rain (74.3 bikes, n=3).
* **Feature Correlation Analysis:**
  * Temperature (`temp` r = +0.405) and hour of day (`hr` r = +0.394) have the strongest positive correlations with rental demand.
  * Humidity (`hum` r = -0.323) and weather severity (`weathersit` r = -0.142) show negative correlations.

### 5. Automated Test Suite
* 33 unit tests covering Sprint 1A, 1B, and 1C passing (`tests/test_sprint1.py`):
  * Dataset loading, shape, nulls, duplicates, and target non-negativity.
  * Descriptive statistics (mean, median, std, skewness).
  * Season coding transitions and target leakage constraints.
  * Scalar input boundary validation (`validate_hour`, `validate_inputs`).
  * DataFrame schema validation (`validate_dataset_schema`).
  * Synthetic corrupted data tests (missing columns, nulls, negative values, target sum violations).
  * Single-record validation (`validate_dataset_record`).
  * Level 0 baseline estimation invariance and fallback warning formatting.
  * EDA distribution metrics, outlier calculation, seasonal patterns, hourly commute peaks, and feature correlations.

> **Status Note (SRS v1.5 Section 13):** Subsequent release gates (Open-Meteo live API integration, chronological train/test 80/20 split, L1–L5 baseline/model ladder, and weather-code mapping) remain pending for later sprints.
