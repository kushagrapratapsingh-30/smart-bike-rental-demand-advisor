# Smart Bike Rental Demand Advisor

A machine learning-backed decision-support application for estimating hourly bike rental demand in Washington, D.C., built using the UCI Bike Sharing Dataset and Streamlit.

Based on Software Requirement Specification (**SRS v1.5**).

---

## Sprint 1A & 1B Verified Milestones

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

### 4. Automated Test Suite
* 28 unit tests covering Sprint 1A and Sprint 1B passing (`tests/test_sprint1.py`):
  * Dataset loading, shape, nulls, duplicates, and target non-negativity.
  * Descriptive statistics (mean, median, std, skewness).
  * Season coding transitions and target leakage constraints.
  * Scalar input boundary validation (`validate_hour`, `validate_inputs`).
  * DataFrame schema validation (`validate_dataset_schema`).
  * Synthetic corrupted data tests (missing columns, nulls, negative values, target sum violations).
  * Single-record validation (`validate_dataset_record`).
  * Level 0 baseline estimation invariance and fallback warning formatting.

> **Status Note (SRS v1.5 Section 13):** Subsequent release gates (Open-Meteo live API integration, chronological train/test 80/20 split, L1–L5 baseline/model ladder, and weather-code mapping) remain pending for later sprints.
