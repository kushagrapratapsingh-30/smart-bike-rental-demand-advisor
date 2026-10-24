# Sprint 1 Closure Report: Smart Bike Rental Demand Advisor (VELO)

- **Project:** Smart Bike Rental Demand Advisor (VELO Dashboard)
- **Institution:** GLA University, Greater Noida — B.Tech CSE (AI/ML) Second-Year Project
- **Academic Supervisor:** Mr. Palash Singh Raghuwanshi | **Reviewer:** Mr. Gautam Mukherjee
- **Specification Source of Truth:** SRS v1.5
- **Audited Commit Hash:** `0964cf5c9a21f2db9188e53c6b265d14607009cd` (`0964cf5`)
- **Remote Branch:** `origin/main` on `https://github.com/kushagrapratapsingh-30/smart-bike-rental-demand-advisor`
- **Closure Recommendation:** **READY TO CLOSE SPRINT 1** (Audit Verified)

---

## 1. Executive Summary

Sprint 1 of the **Smart Bike Rental Demand Advisor** project has successfully achieved all core objectives outlined in **SRS v1.5**. The system delivers an end-to-end operational decision-support tool for bike-sharing fleet managers in Washington, D.C. It combines data sanitization, comprehensive exploratory and statistical data analysis, a 6-tier baseline and machine learning evaluation ladder (L0 to L5), real-time Open-Meteo weather integration, and a five-page professional Streamlit dashboard inspired by the VELO design language.

The full test suite (`python -m unittest discover -s tests -v`) runs **52 discovered tests** with **52 passed, 0 skipped, 0 failures, 0 errors** (execution time: 2.882s). All 33 core tests in `test_sprint1.py` and all 19 tests in `test_sprint2.py` (including calendar, weather normalization, split sizes, data leakage prevention, deterministic baseline fallbacks, L4/L5 predictions, and bundle loading integrity) pass cleanly. Windows Smart App Control has validated `_ball_tree.cp314-win_amd64.pyd` as trusted on the host, permitting native Random Forest training and inference. Additionally, robust model-bundle integrity validation ensures that if the application is deployed in an environment where Random Forest is blocked, it gracefully and transparently falls back to L4 (Linear Regression) without fabricating L5 metrics. Bytecode compilation and Git diff whitespace checks pass with zero errors, and the repository is cleanly synchronized with remote `origin/main`.

---

## 2. Completed Deliverables

| Deliverable | Location | Description & Role |
| :--- | :--- | :--- |
| **Data Ingestion & Cleaning** | `src/data_cleaning.py` | Robust dataset loading, column mapping, date parsing, and schema validation. |
| **Input Validation** | `src/validation.py` | Strict domain bound checks, type coercions, and standardized validation error reporting (`ERR_VAL_01`). |
| **Exploratory Data Analysis** | `src/eda.py` | Hourly commute patterns, seasonal distributions, weather impact metrics, working day splits, Pearson correlations, and target distribution analytics. |
| **Calendar Feature Derivation** | `src/calendar_utils.py` | Automatic derivation of month, hour, day of week, season code, holiday indicator, and working day flag from datetime timestamps. |
| **Weather API Integration** | `src/weather.py` | Open-Meteo REST API client with WMO weather condition code mapping, min-max normalization, 2.0s network timeout, and deterministic fallback handling. |
| **Model Management & Ladder** | `src/models.py` | Chronological 80/20 train/test split, L0–L3 grouped mean baselines with cascading fallbacks, L4 Linear Regression, L5 Random Forest Regressor, bundle integrity validation with auto-retraining, and transparent fallbacks. |
| **Visualization Helpers** | `src/visualization.py` | Reusable Matplotlib plotting routines conforming to the VELO aesthetic. |
| **VELO Streamlit Dashboard** | `app.py` | 5-page interactive UI (Overview, Demand Predictor, Model Performance, Data Insights, About Project) with dynamic model tags, custom CSS, and zero-indent callouts. |
| **Full Unit Test Suite** | `tests/test_sprint1.py`<br/>`tests/test_sprint2.py` | 52 discovered tests: **52 passed, 0 skipped, 0 failures, 0 errors**. Covers cleaning, schema validation, EDA, calendar, weather, baselines, L4/L5 models, and bundle loading integrity. |

---

## 3. Dataset & Schema Integrity

- **Active Dataset:** Capital Bikeshare hourly records (`dataset/hour.csv`).
- **Record Volume:** 17,379 hourly instances spanning January 1, 2011 to December 31, 2012.
- **Column Count:** 17 raw features.
- **Missing Values:** 0 null values across all columns.
- **Duplicate Records:** 0 duplicate rows (excluding index column `instant`).
- **Target Variable ($y$):** `cnt` (total hourly bike rentals, integer $\ge 0$).
  - Mean: $189.46$ bikes/hour
  - Standard Deviation: $181.39$ bikes/hour
  - Range: $1$ to $977$ bikes/hour
- **Constraint C-2 (Target Leakage Prevention):** `casual` and `registered` strictly excluded from candidate predictors, honoring $\text{casual} + \text{registered} \equiv \text{cnt}$.
- **Design Decision D6 (Temporal Leakage Prevention):** `yr` (year) excluded to ensure model applicability beyond 2012.
- **Active Model Feature Set (11 features):** `['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']`.

---

## 4. Statistical Analysis & Empirical Insights

### 4.1 Target Variable Distribution (`cnt`)
- **Central Tendency:** Mean ($189.46$) exceeds Median ($142.00$).
- **Distribution Shape:** Positively skewed (+1.28 skewness), characteristic of urban trip demand count distributions.
- **Dispersion & Interquartile Range:** $Q1 = 40.00$, $Q3 = 281.00$, $\text{IQR} = 241.00$.
- **Tukey's Outlier Fence:** Upper fence ($Q3 + 1.5 \times \text{IQR}$) = $642.50$ bikes/hour.
- **Outlier Quantification:** Exactly 505 records (2.91%) exceed the fence (414 on working days during peak commuter rushes, 91 on non-working days during major leisure spikes).

### 4.2 Key Empirical Patterns
- **Hourly Commute Spikes:** Working days exhibit a bimodal distribution with distinct morning rush-hour peak at 8:00 AM (477.0 bikes/hr; overall 359.0) and evening peak at 5:00 PM (525.3 bikes/hr; overall 461.5).
- **Lowest Demand Troughs:** Night reset trough occurs at 4:00 AM (6.4 bikes/hr) and 3:00 AM (11.7 bikes/hr).
- **Seasonal Variation:** Summer experiences peak demand ($236.0$ bikes/hr, $+112\%$ over Winter), followed by Spring ($208.3$), Fall ($198.9$), and Winter ($111.1$).
- **Weather Sensitivity:** Monotonic decrease with weather severity: Clear ($204.9$) $\to$ Mist/Cloudy ($175.2$) $\to$ Light Rain/Snow ($111.6$) $\to$ Heavy Rain ($74.3$, $n=3$).
- **Feature Correlations:** Temperature ($r = +0.405$) and Hour ($r = +0.394$) have the strongest positive linear association with total rentals; Humidity ($r = -0.323$) and Weather severity ($r = -0.142$) show negative associations.

---

## 5. Machine Learning & Evaluation Ladder (L0–L5)

### 5.1 Chronological Evaluation Protocol
- **Split Strategy:** Strict chronological 80/20 train/test split without random shuffling.
- **Training Set:** Earlier 80.0% of records ($13,903$ rows, 2011-01-01 00:00 to 2012-08-05 05:00).
- **Held-Out Test Set:** Latest 20.0% of records ($3,476$ rows, 2012-08-05 06:00 to 2012-12-31 23:00).
- **Primary Metric:** Mean Absolute Error (MAE in bikes/hour), providing direct operational interpretability.

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

### 5.2 Results Ladder on Held-Out Test Set

| Tier | Model / Approach | Test MAE (bikes/hr) | vs L1 Benchmark | Technical Behavior |
| :---: | :--- | :---: | :---: | :--- |
| **L0** | Overall Mean Baseline | 174.98 | -53.86% | Fixed historical train mean ($174.64$); ignores temporal/weather factors. |
| **L1** | Hour Mean Baseline | 113.73 | 0.00% | Hourly train group means; captures diurnal commute cycles. |
| **L2** | Hour + Working Day Baseline | 101.55 | +10.71% | 48 group means distinguishing workday vs weekend diurnal curves. |
| **L3** | Hour + Working Day + Season | 95.48 | +16.05% | 192 group means capturing seasonal workday cycles. |
| **L4** | Multiple Linear Regression | 134.70 | -18.44% | **Honest reporting per SRS 3.2**: Underperforms L1 due to inability of hyperplanes to fit non-linear bimodal surges. |
| **L5** | Random Forest Regressor (100 Trees) | **85.19** | **+25.09%** | **Best ML model**: Non-linear ensemble fitting weather, calendar, and diurnal interactions. (*Gracefully falls back to L4 with explicit notice if native libraries are blocked). |

### 5.3 Operational Safeguards
1. **Deterministic Baselines Fallback:** L3 safely cascades to L2 $\to$ L1 $\to$ L0 if an input combination was unobserved in the training window.
2. **Model Bundle Integrity Validation:** If Random Forest is available at runtime but a loaded bundle lacks a valid L5 model, the system automatically triggers retraining. If Random Forest is unavailable, the bundle enforces `rf_model=None` and suppresses fabricated L5 scores.
3. **Graceful ML Fallback:** If Random Forest is blocked by host policies, requests deterministically fall back to L4 (Linear Regression) with a prominent UI disclaimer and honest reporting of the active model and MAE.
4. **Output Clipping:** Predictions are bounded by $\hat{y} \ge 0$ to guarantee physical plausibility.

---

## 6. Dashboard Implementation (Five Pages)

The application provides a 5-page dashboard adhering to the VELO design system:
1. **Overview Page:**
   - 4 Top KPI Cards (Total Records: 17,379, Historical Mean: 189.5, Active Model: L5/L4, Test MAE: 85.2 / 134.7).
   - Dynamic 24-hour demand pattern chart with peak annotation (5:00 PM at 461.5).
   - Key Empirical Insights panel with polished `.insight-card` callouts.
   - System & Operational Metadata card with formal SRS v1.5 disclosures.
2. **Demand Predictor Page:**
   - Interactive date picker and hour slider with automatic calendar feature derivation.
   - Real-time weather sync via Open-Meteo REST API with manual fallback toggle.
   - Tier selector (L0–L5) and single-click prediction engine.
   - Hero prediction display with historical test MAE and SRS Section 3.2 operational notice.
   - Dynamic fallback notice banner when baseline or environment fallbacks are invoked.
3. **Model Performance Page:**
   - Benchmark table comparing all 6 tiers with percentage improvement relative to L1.
   - Visual MAE comparison and residual error distribution histogram.
   - Residuals vs Predicted Demand scatter plot.
   - Release Gate 10 evaluation audit verification card.
4. **Data Insights Page:**
   - 6 Interactive Tabs:
     - Tab 0: Target Variable (`cnt`) Distribution with Mean, Median, and IQR Fence overlays.
     - Tab 1: Commuter Commute vs Weekend Leisure comparative curves.
     - Tab 2: Season-Wise Demand Breakdown (SRS v1.5 Winter/Spring/Summer/Fall mapping).
     - Tab 3: Weather Severity Impact breakdown with small-sample notice for Heavy Rain ($n=3$).
     - Tab 4: Working Day vs Non-Working Day statistical comparison.
     - Tab 5: Feature Pearson Correlation horizontal bar chart.
5. **About Project Page:**
   - Academic context, institutional details, and supervision credits.
   - Methodology, build order, technology stack, and MAE metric rationale.
   - Limitations and formal academic prototype disclaimer.

---

## 7. Verification Matrix (Automated vs Manual)

To preserve analytical honesty and strictly honor Requirement 4, deliverables and findings are categorized by their exact verification method and state:

| Area | Verified Fact | Verification Method | Verification State |
| :--- | :--- | :--- | :---: |
| **Dataset & Cleaning** | 17,379 rows, 0 nulls, 0 duplicates, valid domain bounds | Executed Python inspection & test suite | **VERIFIED** |
| **Feature Leakage** | `casual`, `registered`, and `yr` excluded from feature matrix | Code inspection of `src/models.py` | **VERIFIED** |
| **Chronological Split** | Exactly 13,903 train rows and 3,476 test rows | Array shape inspection in test suite | **VERIFIED** |
| **Evaluation MAEs** | L0: 174.98, L1: 113.73, L2: 101.55, L3: 95.48, L4: 134.70, L5: 85.19 | Independent reproduction on held-out test split | **VERIFIED** |
| **Output Bounds** | Non-negative predictions ($\ge 0$) across all models | Verification script & unit tests | **VERIFIED** |
| **Full Unit Test Suite** | 52 tests across `test_sprint1.py` & `test_sprint2.py` | `python -m unittest discover -s tests -v`: **Ran 52 tests in 2.882s: 52 passed, 0 skipped, 0 failures, 0 errors**. | **VERIFIED (52/52 PASSED)** |
| **Sprint 1 Unit Tests** | 33 passing unit tests in `test_sprint1.py` | `python -m unittest tests/test_sprint1.py` (`Ran 33 tests in 0.170s — OK`) | **RE-VERIFIED (33/33 PASSED)** |
| **Sprint 2 Tests** | 19 passing unit tests in `test_sprint2.py` | `python -m unittest tests/test_sprint2.py` (Calendar, Weather, Chronological Split, Leakage, L4/L5 predictions, Fallbacks, Bundle Integrity) | **RE-VERIFIED (19/19 PASSED)** |
| **Bytecode Compilation** | Zero syntax or indentation errors | `python -m py_compile app.py src/*.py` | **RE-VERIFIED** |
| **Git Whitespace** | Zero trailing whitespace or formatting conflicts | `git diff --check` | **RE-VERIFIED** |
| **Git Tracking** | Synchronized with `origin/main` at commit `0964cf5` | `git status -sb` | **RE-VERIFIED** |
| **Browser Top Header** | Black strip eliminated; header matches `#F8FAFC`; badges unclipped | User manual browser inspection | **VERIFIED (BROWSER)** |
| **Browser Insights** | Clean `.insight-card` styling with no raw `<div>` tags | User manual browser inspection | **VERIFIED (BROWSER)** |
| **Live Open-Meteo API** | HTTP 200 forecast retrieval with fallback on timeout | Unit test mock + live network tests | **VERIFIED** |

---

## 8. Known Limitations & Technical Scope

1. **Spatial Aggregation:** Predictions represent city-wide aggregate hourly demand across Washington, D.C.; station-level dock balancing and individual GPS tracking are out of scope for Sprint 1 per SRS v1.5.
2. **Temporal Horizon:** Models are calibrated on historical Capital Bikeshare data (2011–2012); present-day deployment requires continuous retraining pipelines.
3. **Weather Forecast Range:** Open-Meteo API provides hourly forecasts up to 7 days; dates outside this range utilize manual parameter overrides or historical fallbacks.
4. **Model Bundle Tracking:** Large serialized artifact `models/model_bundle.joblib` is ignored in Git per Option A; cold clones automatically regenerate the bundle on first run via deterministic training logic.
5. **Runtime Host Policy Resilience:** Windows Smart App Control has validated `_ball_tree.cp314-win_amd64.pyd`, enabling native execution of `RandomForestRegressor` on the current host. To maintain portability across any restricted environment:
   - `src/models.py` maintains the `try...except ImportError` guard and exposes `RF_AVAILABLE` and `RF_IMPORT_ERROR`.
   - `_load_bundle()` validates that loaded bundles in RF-capable environments contain valid L5 models, triggering auto-retraining if stale, while suppressing unmeasured L5 scores in restricted environments.
   - Transparent L4 fallbacks and clear UI banners ensure zero runtime crashes regardless of host OS security policies.

---

## 9. Conclusion & Recommendation

All functional requirements, architectural constraints, statistical EDA objectives, and UI styling standards defined in **SRS v1.5** are fully implemented, verified, and committed to GitHub `origin/main` at commit `0964cf5`.

**Final Recommendation:** **READY TO CLOSE SPRINT 1.**
