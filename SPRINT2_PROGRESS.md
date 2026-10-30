# Sprint 2 Progress & Verification Report: Smart Bike Rental Demand Advisor (VELO)

- **Project:** Smart Bike Rental Demand Advisor (VELO Dashboard)
- **Institution:** GLA University, Greater Noida — B.Tech CSE (AI/ML) Second-Year Project
- **Academic Supervisor:** Mr. Palash Singh Raghuwanshi | **Reviewer:** Mr. Gautam Mukherjee
- **Specification Source of Truth:** SRS v1.5
- **Working Branch:** `main` tracking `origin/main` (Base Commit: `6c174c9b2320de4a78082fb9179abc2b1d55fe11` (`6c174c9`) / `e6bab936b47d8c666fa15130062fdfceeffdcb1f` (`e6bab93`))
- **Status:** **SPRINT 2 CORE REQUIREMENTS & TEST MATRIX VERIFIED**

---

## 1. Executive Summary

Sprint 2 of the **Smart Bike Rental Demand Advisor** project implements and validates the end-to-end model training, prediction pipeline, deployment artifact contract, and standardized failure handling in strict adherence to **SRS v1.5**. 

Key accomplishments include:
1. **Deployment Artifact Contract (SRS Section 5.7):** Model bundles dynamically encapsulate verified training/testing date ranges, chronological split metadata, mapping schemas, and runtime dependency versions.
2. **Standardized Warning & Error Taxonomy (SRS Section 5.5 & Appendix B):** Implemented `WARN_SYS_01` (for L0 fallback during missing model artifacts or baseline cascades) and `ERR_SYS_01` (when prediction is impossible due to missing model and L0 configurations), guaranteeing no fabricated rental counts.
3. **Appendix C Test Matrix Coverage (TC-06 to TC-10):** Implemented automated, non-mocked tests for fallback handling, error handling, cascading baseline chains, repeated input numerical consistency, and chronological evaluation integrity across identical test subsets.
4. **Complete Unit Test Verification:** All **78 tests** pass cleanly with 0 failures, 0 errors, and 0 skipped (`python -m unittest discover -s tests -v`).
5. **Testing Plan Step 4 (Responsive Layout Audit):** Validated responsive presentation across Desktop (1280px), Tablet (768px), and Mobile (390px) viewports with touch-scroll table wrapping, stacked header layouts, 2-column mobile weather grids, and explicit Manual Weather Override parameter units.
6. **Testing Plan Step 5 (MAE Formula Mathematical Rendering):** Replaced raw delimiter LaTeX nested inside an HTML `<div>` on the About Project page with native `st.latex()`, ensuring standard formula rendering $\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$ accompanied by clear plain-English explanation.

---

## 2. Requirement Traceability Matrix (SRS v1.5)

| SRS ID | Requirement Description | Implementation Location | Verification Status |
| :--- | :--- | :--- | :--- |
| **FR-3.1** | **Preprocessing Parity:** Exact alignment of real-time input normalization with training data preconditioning (min-max scaling of temp, atemp, hum, windspeed; WMO weather code mapping). | `src/weather.py`<br/>`src/calendar_utils.py` | **Verified** (`test_wmo_code_mapping`, `test_temp_normalization`, `test_atemp_normalization`, `test_windspeed_and_hum_normalization`) |
| **FR-3.2** | **Non-Negative Predictions:** Physical demand constraint $\hat{y} \ge 0$ enforced via post-prediction non-negative clipping on all models. | `src/models.py` | **Verified** (`test_l4_non_negative_predictions`, `test_l5_non_negative_predictions`) |
| **FR-3.3** | **Evaluation Protocol Disclosures:** Explicit reporting of chronological 80/20 train/test split, sample sizes, and honest disclosure of Linear Regression (L4) underperformance vs L1. | `src/models.py`<br/>`app.py` | **Verified** (`test_chronological_split_sizes`, UI Model Performance disclosure note) |
| **FR-3.4** | **Model Comparison Table:** Evaluation across all six tiers (L0–L5) reporting MAE and percentage improvement relative to Level 1 benchmark. | `src/models.py`<br/>`app.py` | **Verified** (`test_mae_scores_calculated`, `test_tc_10_chronological_evaluation_integrity`) |
| **Section 5.5** | **Failure Handling & Standardized Codes:** Deterministic fallbacks with `WARN_SYS_01` and execution halt with `ERR_SYS_01` when prediction is impossible. | `src/models.py` | **Verified** (`test_tc_06_model_missing_uses_saved_l0_with_warn_sys_01`, `test_tc_07_model_and_l0_missing_returns_err_sys_01`) |
| **Section 5.7** | **Deployment Artifact Contract:** Persisted model bundle metadata containing versions, exact date ranges, split parameters, and dependency versions. | `src/models.py` | **Verified** (`test_deployment_artifact_contract_metadata`) |
| **NFR-4** | **Model Invariance & Numerical Consistency:** Repeated identical inputs yield identical outputs within floating-point tolerance ($\Delta < 10^{-6}$). | `src/models.py` | **Verified** (`test_tc_09_repeat_identical_input_five_times`) |

---

## 3. Deployment Artifact Contract (SRS Section 5.7)

The model bundle (`models/bike_demand_bundle.joblib`) records the following verified operational metadata generated directly from runtime state and dataset index boundaries:

```python
{
    "artifact_version": "1.5.0",
    "srs_version": "v1.5",
    "training_date_range": ("2011-01-01 00:00", "2012-08-07 11:00"),
    "test_date_range": ("2012-08-07 12:00", "2012-12-31 23:00"),
    "split_definition": {
        "train_ratio": 0.80,
        "test_ratio": 0.20,
        "split_index": 13903,
        "shuffle": False,
        "strategy": "chronological",
    },
    "mapping_versions": {
        "weather_code": "1.5",
        "season_mapping": "1.5",
    },
    "dependency_versions": {
        "python": "3.14.2",
        "scikit-learn": "1.9.1",
        "pandas": "3.0.6",
        "numpy": "2.5.3",
    },
}
```

---

## 4. Evaluation Ladder & Verified MAE Scores

Evaluation is conducted on the held-out chronological test split ($N = 3,476$ hours, spanning `2012-08-07 12:00` to `2012-12-31 23:00`):

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

$$\text{Improvement vs L1 (\%)} = \frac{\text{MAE}_{\text{L1}} - \text{MAE}_{\text{Tier}}}{\text{MAE}_{\text{L1}}} \times 100$$

| Tier | Model Architecture | Test MAE (bikes/hr) | vs L1 Benchmark | Technical Behavior & SRS Compliance |
| :---: | :--- | :---: | :---: | :--- |
| **L0** | Overall Mean Baseline | **174.98** | -53.86% | Fixed historical training mean ($174.64$). Ignores temporal/weather factors. Standardized fallback target. |
| **L1** | Hour Mean Baseline | **113.73** | 0.00% | 24 hourly training means. Benchmark model capturing diurnal commute rhythms. |
| **L2** | Hour + Working Day Baseline | **101.55** | +10.71% | 48 group means distinguishing weekday vs weekend diurnal curves. |
| **L3** | Hour + Working Day + Season | **95.48** | +16.05% | 192 group means capturing seasonal workday cycles. Lowest MAE among non-ML baselines. |
| **L4** | Multiple Linear Regression | **134.70** | -18.44% | **Honest disclosure per FR-3.3**: Underperforms L1 because planar hyperplanes cannot capture bimodal non-linear rush-hour spikes. |
| **L5** | Random Forest Regressor (100 Trees) | **85.19** | **+25.09%** | **Best Performing Model**: Successfully models complex feature interactions across calendar, commute, and weather dimensions. |

---

## 5. Appendix C Test Matrix Implementation & Status

| Test Case | Test Description | Condition Tested | Expected Behavior | Automated Test Method | Result |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **TC-01** | Core Prediction Baseline | Valid inputs | Level 0 baseline prediction with `WARN_SYS_01` (when model unbuilt) | `test_tc_01_core_prediction` | **PASSED** |
| **TC-02** | Boundary Violation | Out-of-bounds input (`hr=25`) | Execution halts immediately with `ERR_VAL_01` | `test_tc_02_invalid_hour_out_of_bounds` | **PASSED** |
| **TC-03** | Invariance Check | Multiple varied inputs | Level 0 returns identical mean for all queries | `test_tc_03_invariance_check` | **PASSED** |
| **TC-06** | Missing Model Handling | Model artifact missing, L0 present | Uses saved L0 configuration with `WARN_SYS_01` and status `model_fallback_l0` | `test_tc_06_model_missing_uses_saved_l0_with_warn_sys_01` | **PASSED** |
| **TC-07** | Total Model & Config Missing | Model and L0 config both missing | Returns `ERR_SYS_01`, `predicted_cnt = None`, no fabricated count | `test_tc_07_model_and_l0_missing_returns_err_sys_01` | **PASSED** |
| **TC-08** | Cascading Fallback Chain | Unseen groups in L3 | Deterministic cascading chain: $\text{L3} \to \text{L2} \to \text{L1} \to \text{L0}$ | `test_tc_08_cascading_baseline_fallback_chain` | **PASSED** |
| **TC-09** | Repeated Input Invariance | Identical inputs submitted 5 times | Identical outputs within float tolerance ($\Delta < 10^{-6}$), NFR-4 | `test_tc_09_repeat_identical_input_five_times` | **PASSED** |
| **TC-10** | Chronological Evaluation Integrity | L0 through L5 predictions on test set | Exactly 3,476 predictions evaluated against identical ground truth instances | `test_tc_10_chronological_evaluation_integrity` | **PASSED** |
| **TC-W01** | Weather API Success | Valid Open-Meteo response | Parsed into normalized features ($[0, 1]$) and raw units | `test_tc_w01_successful_weather_response` | **PASSED** |
| **TC-W02** | Connection Failure | Network resolution failure | Status `weather_api_error` with `WARN_API_01` | `test_tc_w02_connection_failure` | **PASSED** |
| **TC-W03** | Request Timeout | Timeout $> 2.0$s | Status `weather_api_error` with `WARN_API_01` (NFR-1) | `test_tc_w03_request_timeout` | **PASSED** |
| **TC-W04** | HTTP Error (503/500) | Non-200 HTTP response | Status `weather_api_error` with `WARN_API_01` | `test_tc_w04_http_error` | **PASSED** |
| **TC-W05** | Invalid Response & Codes | Malformed JSON, missing/null fields, unknown WMO, outside coverage | Status `weather_validation_error` (`WARN_API_01` / `WARN_MAP_01`) or `unsupported_time` (`WARN_API_02`) | `test_tc_w05_*` | **PASSED** |
| **TC-W06** | Weather Fallback Paths | Simulated weather failure | Manual override and L0 baseline paths produce non-negative predictions | `test_tc_w06_fallback_behavior_and_prediction` | **PASSED** |
| **TC-W07** | Weather Recovery | Failure followed by valid response | Immediate subsequent recovery with live predictions and zero error residue | `test_tc_w07_recovery_after_failure` | **PASSED** |
| **TC-C01** | Corrupt Bundle File Fallback | Corrupt joblib payload on disk | No auto-retraining; falls back to saved L0 mean with `WARN_SYS_01` | `test_bundle_load_corrupted_file_falls_back_to_l0_with_warn_sys_01` | **PASSED** |
| **TC-C02** | Corrupt Bundle & Missing L0 | Corrupt file and uninitialized L0 | Halts execution with `ERR_SYS_01`; no fabricated predictions | `test_bundle_load_corrupted_file_and_missing_l0_returns_err_sys_01` | **PASSED** |
| **TC-R01** | Media Query Breakpoints | CSS breakpoint presence in app | `@media (max-width: 768px)` and `@media (max-width: 480px)` defined | `test_responsive_media_queries_defined` | **PASSED** |
| **TC-R02** | Weather Grid Responsive Layout | Viewports $\le 768$px | Grid switches from 4 columns to 2 columns to prevent min-card squeeze | `test_responsive_weather_grid_breakpoint` | **PASSED** |
| **TC-R03** | Page Header Stacking | Narrow viewports $\le 768$px | Title and badges stack vertically; badges wrap to avoid clipping | `test_page_header_responsive_stacking` | **PASSED** |
| **TC-R04** | Weather Override Labels & Units | Manual override form controls | Explicit unit labels for temp (°C), hum (%), wind (km/h), condition | `test_manual_weather_override_controls_and_units` | **PASSED** |
| **TC-R05** | Table Touch-Scroll Wrapper | 5-tier evaluation summary table | Enclosed in `overflow-x: auto` wrapper with min-width preventing page blowout | `test_model_performance_table_responsive_scroll_wrapper` | **PASSED** |
| **TC-M01** | MAE Formula Native st.latex | About Project page metric display | Uses native `st.latex()` API instead of raw HTML delimiters | `test_mae_formula_uses_native_st_latex` | **PASSED** |
| **TC-M02** | MAE Formula Standard Form | Mathematical equation content | Formatted as $\text{MAE} = \frac{1}{n} \sum_{i=1}^n \|y_i - \hat{y}_i\|$ | `test_mae_formula_content_standard_form` | **PASSED** |
| **TC-M03** | MAE Explanatory Text Definition | Accompanying metric description | States MAE is the average absolute difference between actual and predicted counts | `test_mae_explanatory_text_definition` | **PASSED** |
| **TC-M04** | No Unrendered LaTeX in HTML | HTML container blocks in app.py | Verifies `$$` delimiters are not nested within raw HTML elements | `test_no_unrendered_latex_in_raw_html_containers` | **PASSED** |

---

## 6. Runtime Environment & Host Verification

- **Operating System:** Microsoft Windows 11 Home Single Language (Build 26200.7462)
- **Host Security Status:** Windows Smart App Control has validated `_ball_tree.cp314-win_amd64.pyd` as trusted.
- **Random Forest Runtime Capability:** Native `RandomForestRegressor` with 100 trees runs without restriction (`RF_AVAILABLE = True`, `RF_IMPORT_ERROR = None`).
- **Defensive Bundle Integrity:** Model bundle loader validates runtime capabilities; if loaded in a restricted environment, it cleanly enforces `rf_model = None` and transparently falls back to L4 with explicit user notification, completely preventing runtime crashes and fabricated L5 metrics.

---

## 7. Testing Plan Step 4: Responsive Layout Audit (1280px, 768px, 390px)

### 7.1 Viewport Audit Across All 5 Pages

| Page | 1280px (Desktop) | 768px (Tablet) | 390px (Mobile) | Confirmed Fix / Safeguard |
| :--- | :--- | :--- | :--- | :--- |
| **1. Overview** | 4-column KPI cards, side-by-side hourly chart and empirical insights. Max width 1440px. | KPI cards adapt to 2x2 grid. Header badges wrap cleanly without overlapping page title. | KPI cards stack into single column with tightened padding (1rem) and value font-size (1.65rem). Container padding reduced to 0.75rem. | Added responsive container padding and KPI card font clamping in CSS. |
| **2. Demand Predictor** | 2-column layout (inputs left, weather/prediction right). 4-card horizontal weather grid. | Columns stack vertically (inputs top, output card below). Weather grid adjusts to 2 columns. | Weather grid displays as clean 2x2 grid. Hero prediction card value clamps to 2.5rem; footer stacks vertically to prevent text collision. | Clamped `.hero-pred-val` to 2.5rem; stacked `.hero-pred-footer` vertically with `text-align: left`. |
| **3. Model Performance** | Side-by-side 5-tier table and scatter plot. Bottom 3-column residuals & audit checklist. | Top columns stack. Bottom 3-column cards stack vertically with responsive spacing. | 5-tier evaluation table enclosed in touch-scroll wrapper (`overflow-x: auto; min-width: 380px`), preventing mobile horizontal blowout. | Added `-webkit-overflow-scrolling: touch; overflow-x: auto;` wrapper around the HTML comparison table. |
| **4. Data Insights** | Full-width tabs with side-by-side charts on Tab 4. All Matplotlib figures scaled. | Tab strip scrolls horizontally. 2-column layout in Tab 4 stacks vertically. | Charts scale seamlessly via `use_container_width=True`. Tab labels scroll smoothly. Text annotations fit comfortably. | Standardized `use_container_width=True` on all 9 pyplot figures; container padding reduced. |
| **5. About Project** | 2-column grid pairs for cards; 2-column tech stack list. | Card pairs stack vertically in single column. | Tech stack switches to single column (`1fr`) to prevent multi-line bullet awkward wrapping. | Added `.tech-stack-grid` media query rule switching `grid-template-columns` to `1fr` at $\le 480$px. |

### 7.2 Manual Weather Override Controls Audit
All four manual override input controls strictly conform to SRS v1.5 with explicit labels, units, ranges, and types:
- **Temperature:** `st.number_input("Temperature (°C)", min_value=-15.0, max_value=45.0, value=20.0, step=0.5)`
- **Humidity:** `st.slider("Humidity (%)", min_value=0, max_value=100, value=55)`
- **Wind Speed:** `st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=70.0, value=12.0, step=1.0)`
- **Weather Condition:** `st.selectbox("Weather Condition", ["Clear", "Mist/Cloudy", "Light Rain/Snow", "Heavy Rain"])`

### 7.3 Visual Verification Limitation Notice
> [!NOTE]
> Automated tests in `TestSprint2ResponsiveLayout` confirm the presence, correctness, and structure of CSS media queries, responsive classes, overflow wrappers, and input control labels. However, automated Python tests **cannot verify pixel-level browser rasterization**. Final visual aesthetic confirmation at 1280px, 768px, and 390px viewports requires interactive inspection via a physical browser (e.g. Chrome DevTools Device Mode) connected to `http://localhost:8501`.

---

## 8. Testing Plan Step 5: MAE Formula Rendering & Mathematical Presentation

### 8.1 Issue Identification & Root Cause
On the **About Project** page, the mathematical definition of Mean Absolute Error (MAE) was previously placed inside an HTML `<div>` wrapper within `st.markdown(..., unsafe_allow_html=True)`. Because Streamlit's CommonMark / markdown-it parser treats raw HTML blocks as literal markup, inner LaTeX math delimiters (`$$...$$`) were never dispatched to KaTeX, displaying unformatted string literals (e.g., `$$\text{MAE} = ...$$`) directly on screen.

### 8.2 Compliant Minimal Fix
1. **Native Mathematical API:** Migrated the equation to Streamlit's first-class `st.latex(r"\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|")`, ensuring clean, centered KaTeX vector rendering.
2. **Definitional Clarity:** Updated the accompanying text to explicitly state that MAE measures the average absolute difference between actual and predicted rental counts across all evaluation instances.
3. **Container Consistency:** Preserved the surrounding `.velo-card` container styling with `height: 100%`, maintaining visual alignment with the adjacent "Limitations & Future Scope" card.
4. **Zero Model or Metric Drift:** Kept all evaluation metrics, baseline calculations, and prediction logic completely untouched.

### 8.3 Browser Verification Requirement
> [!NOTE]
> Unit tests in `TestSprint2MAEFormulaRendering` verify the syntax, presence, and AST structure of the `st.latex()` invocation. Physical visual rendering of the KaTeX formula should be confirmed in an interactive browser session at `http://localhost:8501` under the "About Project" tab.

---

## 9. Sprint III Pending Scope

To maintain clean project boundaries, the following tasks remain documented for **Sprint III** and were deliberately not backported into Sprint 2:
- **UUID Structured Request Logging (NFR-3 / SRS Table 9.4):** Generation of a 36-character RFC 4122 v4 UUID for every prediction request, logged to persistent structured storage with timestamp, input vector, output count, active tier, and latency in milliseconds.
- **Operational Fleet Management Advisory:** Recommendations for bike rebalancing thresholds and surplus/deficit alerts based on predicted demand tiers.

---

## 10. Verification Command & Output

```
$ python -m unittest discover -s tests -v
Ran 78 tests in 1.612s

OK (78 passed, 0 failures, 0 errors, 0 skipped)

$ python -m py_compile app.py src/*.py tests/*.py
[Exit Code: 0 - Clean Bytecode Compilation]

$ git diff --check
[Exit Code: 0 - Clean Whitespace & Line Feeds]
```
