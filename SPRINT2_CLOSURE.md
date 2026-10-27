# Sprint 2 Closure Report: Smart Bike Rental Demand Advisor (VELO)

- **Project:** Smart Bike Rental Demand Advisor (VELO Dashboard)
- **Institution:** GLA University, Greater Noida — B.Tech CSE (AI/ML) Second-Year Project
- **Academic Supervisor:** Mr. Palash Singh Raghuwanshi | **Reviewer:** Mr. Gautam Mukherjee
- **Specification Source of Truth:** SRS v1.5
- **Working Branch:** `main` tracking `origin/main` (Base Commit: `9fee18d`)
- **Closure Recommendation:** **READY TO CLOSE SPRINT 2** (Audit Verified)

---

## 1. Executive Summary

Sprint 2 of the **Smart Bike Rental Demand Advisor** project implements and formally verifies all core functional and architectural objectives defined in **SRS v1.5**. The sprint transitions the platform from baseline prototypes to a hardened, production-grade operational decision-support tool.

Key accomplishments achieved during Sprint 2 include:
1. **Deployment Artifact Contract (SRS Section 5.7):** Persisted model bundle dynamically encapsulates validated training/testing temporal boundaries, strict chronological split definitions, mapping schema versions, and runtime dependency versions.
2. **Standardized Warning & Error Taxonomy (SRS Section 5.5 & Appendix B):** Standardized codes implemented across all pipelines: `WARN_SYS_01` (missing artifact or baseline fallback to L0), `ERR_SYS_01` (unsupported execution when model and L0 configurations are unavailable), `WARN_API_01` (weather connection/timeout/validation fallback), `WARN_API_02` (timestamps outside forecast coverage), and `WARN_MAP_01` (unknown WMO weather code). No fabricated rental predictions under failure states.
3. **Appendix C Test Matrix Full Coverage:** Automated, non-mocked tests implemented for TC-06 (missing model artifact), TC-07 (unsupported execution without fabricated count), TC-08 (complete L3 $\to$ L2 $\to$ L1 $\to$ L0 cascading fallback ladder), TC-09 (numerical consistency across repeated identical inputs, NFR-4), and TC-10 (chronological evaluation integrity across identical 3,476 test instances).
4. **Complete Weather API Failure & Recovery Matrix (TC-W01 to TC-W07):** Deterministic mock tests validating connection failures, timeouts ($> 2.0$s per NFR-1), HTTP 500/503 errors, malformed payloads, null attributes, unknown WMO weather codes, and seamless live recovery.
5. **Corrupt Artifact Degradation (TC-C01 to TC-C02):** Unreadable or corrupt joblib artifacts on disk trigger zero automatic retraining or overwrite side-effects, safely degrading to saved L0 baselines with `WARN_SYS_01`.
6. **Responsive Layout Hardening (Testing Plan Step 4):** Standardized tablet (768px) and mobile (390px) CSS media queries, touch-scrolling table overflow wrappers (`overflow-x: auto`), card font clamping, 2-column mobile weather grids, and explicit unit labels for Manual Weather Override controls.
7. **Native Mathematical Presentation (Testing Plan Step 5):** Migrated from raw delimiter LaTeX inside HTML `<div>` tags to Streamlit's first-class `st.latex()` API for the Mean Absolute Error definition ($\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$).
8. **100% Test Suite Verification:** The full test suite runs **78 discovered tests** with **78 passed, 0 failures, 0 errors, and 0 skipped** (`python -m unittest discover -s tests -v`).

---

## 2. Completed Sprint 2 Deliverables

| Deliverable | Location | Description & Role |
| :--- | :--- | :--- |
| **Model Management & Evaluation Engine** | `src/models.py` | Chronological 80/20 train/test split, L0–L3 baseline ladder with fallback cascades, L4 Multiple Linear Regression, L5 Random Forest Regressor, Section 5.7 artifact contract metadata, and Section 5.5 standardized error/warning taxonomy. |
| **Weather API Client & Validation** | `src/weather.py` | Open-Meteo REST API client with 2.0s network timeout, WMO code mapping, null value checks, ISO timestamp lookups, and standardized warning codes (`WARN_API_01`, `WARN_API_02`, `WARN_MAP_01`). |
| **VELO Streamlit Dashboard** | `app.py` | 5-page interactive dashboard with responsive CSS breakpoints (768px tablet, 480px/390px mobile), touch-scrolling comparison table wrapper, explicit override input units, resilient active model tags, and native `st.latex()` mathematical rendering. |
| **Sprint 2 Unit & Regression Suite** | `tests/test_sprint2.py` | 45 comprehensive tests covering calendar utilities, weather normalization, weather API failure/recovery (TC-W01–TC-W07), model manager & baseline cascades (TC-06–TC-10), bundle loading & corruption safety (TC-C01–TC-C02), responsive layout logic (TC-R01–TC-R05), and MAE formula rendering (TC-M01–TC-M04). |
| **Documentation & Traceability Reports** | `SPRINT2_PROGRESS.md`<br/>`SPRINT2_CLOSURE.md` | Comprehensive progress and formal closure reports documenting requirements traceability, verification results, runtime status, and Sprint III boundaries. |

---

## 3. Deployment Artifact Contract (SRS Section 5.7)

Model bundles (`models/model_bundle.joblib`) strictly comply with Section 5.7 by recording verified operational metadata generated directly from runtime state and dataset index boundaries:

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

Evaluation is executed on identical held-out chronological test instances ($N = 3,476$ hours, spanning `2012-08-07 12:00` to `2012-12-31 23:00`). Candidate features strictly exclude `casual` and `registered` (Constraint C-2) and `yr` (Design Decision D6):

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

$$\text{Improvement vs L1 (\%)} = \frac{\text{MAE}_{\text{L1}} - \text{MAE}_{\text{Tier}}}{\text{MAE}_{\text{L1}}} \times 100$$

| Tier | Model Architecture | Test MAE (bikes/hr) | vs L1 Benchmark | Technical Behavior & Compliance |
| :---: | :--- | :---: | :---: | :--- |
| **L0** | Overall Mean Baseline | **174.98** | -53.86% | Fixed historical training mean ($174.64$). Standardized fallback target. |
| **L1** | Hour Mean Baseline | **113.73** | 0.00% | 24 hourly training group means. Benchmark model capturing diurnal commuter rhythms. |
| **L2** | Hour + Working Day Baseline | **101.55** | +10.71% | 48 group means distinguishing workday vs weekend diurnal curves. |
| **L3** | Hour + Working Day + Season | **95.48** | +16.05% | 192 group means capturing seasonal workday cycles. Lowest MAE among non-ML baselines. |
| **L4** | Multiple Linear Regression | **134.70** | -18.44% | **Honest disclosure per FR-3.3**: Underperforms L1 because planar hyperplanes cannot capture bimodal non-linear commuter rush-hour peaks. |
| **L5** | Random Forest Regressor (100 Trees) | **85.19** | **+25.09%** | **Best Performing Model**: Successfully models complex non-linear interactions across calendar, commute, and weather dimensions. |

---

## 5. Standardized Warning & Error Taxonomy (Section 5.5 & Appendix B)

| Code | Type | Trigger Condition | System Behavior |
| :---: | :---: | :--- | :--- |
| `WARN_SYS_01` | Warning | Model artifact missing or corrupted; baseline group unobserved in training split. | Degrades to saved Level 0 baseline mean ($174.64$) with fallback notice. |
| `ERR_SYS_01` | Error | Prediction requested when both model artifact and L0 baseline configuration are missing. | Execution halted; returns `predicted_cnt = None`. **Zero fabricated predictions.** |
| `WARN_API_01` | Warning | Open-Meteo connection failure, timeout ($> 2.0$s), HTTP error, or null/malformed payload. | Prompts manual weather override or routes to historical seasonal baseline. |
| `WARN_API_02` | Warning | Forecast requested for timestamp outside available API forecast horizon. | Alerts user that timestamp is unsupported; prompts manual override. |
| `WARN_MAP_01` | Warning | Open-Meteo returns unmapped WMO weather condition code. | Rejects silent coercion to default weather class; prompts manual specification. |
| `ERR_VAL_01` | Error | Out-of-bounds input parameter (e.g. `hr=25`, `season=5`). | Validation halts execution immediately before model execution. |

---

## 6. Full Unit Test Matrix Coverage (78 Tests)

```
$ python -m unittest discover -s tests -v
Ran 78 tests in 2.799s

OK (78 passed, 0 failures, 0 errors, 0 skipped)
```

### 6.1 Appendix C & Core Model Tests (TC-01 to TC-10)
- **TC-01 (Core Prediction Baseline):** Valid inputs predict Level 0 mean with `WARN_SYS_01` when models are unbuilt (`PASSED`).
- **TC-02 (Boundary Violation):** Out-of-bounds input (`hr=25`) halts execution immediately with `ERR_VAL_01` (`PASSED`).
- **TC-03 (Invariance Check):** Level 0 returns identical mean for all varied queries (`PASSED`).
- **TC-06 (Missing Model Handling):** Missing model artifact uses saved L0 configuration with `WARN_SYS_01` and status `model_fallback_l0` (`PASSED`).
- **TC-07 (Total Model & Config Missing):** Model and L0 config both missing returns `ERR_SYS_01` with `predicted_cnt = None` (`PASSED`).
- **TC-08 (Cascading Fallback Chain):** Unseen groups in L3 cascade deterministically: $\text{L3} \to \text{L2} \to \text{L1} \to \text{L0}$ (`PASSED`).
- **TC-09 (Repeated Input Consistency):** Identical inputs submitted 5 times yield identical outputs within floating-point tolerance ($\Delta < 10^{-6}$), satisfying NFR-4 (`PASSED`).
- **TC-10 (Chronological Evaluation Integrity):** L0 through L5 predictions evaluate on identical 3,476 held-out test instances (`PASSED`).

### 6.2 Weather API Failure & Recovery Matrix (TC-W01 to TC-W07)
- **TC-W01:** Valid Open-Meteo response parsed into normalized and raw schema (`PASSED`).
- **TC-W02:** Connection drop triggers `weather_api_error` with `WARN_API_01` (`PASSED`).
- **TC-W03:** Network timeout ($> 2.0$s) triggers `weather_api_error` with `WARN_API_01` (`PASSED`).
- **TC-W04:** HTTP 503/500 error triggers `weather_api_error` with `WARN_API_01` (`PASSED`).
- **TC-W05:** Malformed JSON, missing/null fields, unknown WMO codes (`WARN_MAP_01`), and unsupported forecast times (`WARN_API_02`) handled per specification (`PASSED`).
- **TC-W06:** Weather failure routes to manual override or L0 baseline with non-negative predictions (`PASSED`).
- **TC-W07:** Immediate recovery on subsequent successful request with zero error state leakage (`PASSED`).

### 6.3 Corrupt Artifact Safety (TC-C01 to TC-C02)
- **TC-C01:** Corrupt joblib payload on disk does not trigger auto-retraining; falls back to saved L0 mean with `WARN_SYS_01` (`PASSED`).
- **TC-C02:** Corrupt file and uninitialized L0 config returns `ERR_SYS_01` without fabricating predictions (`PASSED`).

### 6.4 Responsive Layout Logic (TC-R01 to TC-R05)
- **TC-R01:** CSS media queries defined for tablet (`<= 768px`) and mobile (`<= 480px` / `390px`) (`PASSED`).
- **TC-R02:** Weather grid transitions from 4 columns to 2 columns on narrow screens (`PASSED`).
- **TC-R03:** Page title and badges stack vertically and wrap without clipping (`PASSED`).
- **TC-R04:** Manual Weather Override controls specify clear labels with units: Temperature (°C), Humidity (%), Wind Speed (km/h), Weather Condition (`PASSED`).
- **TC-R05:** 5-tier evaluation comparison table wrapped in `overflow-x: auto` container (`PASSED`).

### 6.5 Mathematical Presentation (TC-M01 to TC-M04)
- **TC-M01:** About Project page utilizes native Streamlit `st.latex()` API instead of raw HTML delimiters (`PASSED`).
- **TC-M02:** Formula formatted strictly per standard: $\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$ (`PASSED`).
- **TC-M03:** Descriptive text states MAE is the average absolute difference between actual and predicted rental counts (`PASSED`).
- **TC-M04:** Verified zero raw `$$` delimiters trapped inside raw HTML elements in `app.py` (`PASSED`).

---

## 7. Verification Matrix (Automated vs Manual)

| Area | Verified Property | Verification Method | Verification State |
| :--- | :--- | :--- | :---: |
| **Dataset Sanitization** | 17,379 rows, 0 nulls, 0 duplicates, valid domains | Automated Unit Suite (`test_sprint1.py`) | **VERIFIED** |
| **Target Leakage** | `casual`, `registered`, and `yr` excluded from features | Code audit & test suite (`test_sprint2.py`) | **VERIFIED** |
| **Chronological Split** | Exactly 13,903 train rows and 3,476 test rows | Array shape inspection in test suite | **VERIFIED** |
| **Evaluation MAE Ladder** | L0: 174.98, L1: 113.73, L2: 101.55, L3: 95.48, L4: 134.70, L5: 85.19 | Deterministic execution on held-out test split | **VERIFIED** |
| **Artifact Contract** | Persisted versions, date ranges, dependency versions | `test_deployment_artifact_contract_metadata` | **VERIFIED** |
| **Failure Taxonomy** | `WARN_SYS_01`, `ERR_SYS_01`, `WARN_API_01`, `WARN_API_02`, `WARN_MAP_01` | Automated tests (TC-06–TC-08, TC-W01–TC-W07) | **VERIFIED** |
| **Bytecode Compilation** | Clean bytecode compilation across `app.py`, `src/*.py`, `tests/*.py` | `python -m py_compile` | **VERIFIED** |
| **Git Diff Whitespace** | Clean line endings, zero trailing whitespace or merge conflict markers | `git diff --check` | **VERIFIED** |
| **Desktop Layout (1280px)** | 4-column KPI cards, side-by-side charts, full-width tabs | User browser inspection (`http://localhost:8501`) | **VERIFIED (USER)** |
| **Tablet Layout (768px)** | 2x2 KPI cards, stacked headers, 2-column weather grid | User browser inspection (Device Mode) | **VERIFIED (USER)** |
| **Mobile Layout (390px)** | Single-column KPI cards, touch-scrolling table, clamped fonts | User browser inspection (Device Mode) | **VERIFIED (USER)** |
| **MAE Mathematical Display** | Native KaTeX centered formula without raw delimiter text | User browser inspection (`http://localhost:8501`) | **VERIFIED (USER)** |

---

## 8. Sprint III Pending Scope

To maintain clean sprint boundaries, the following items remain documented for **Sprint III** and were deliberately not backported:
- **UUID Structured Request Logging (NFR-3 / SRS Table 9.4):** Generation of a 36-character RFC 4122 v4 UUID for every prediction request, logged to persistent structured storage with timestamp, input vector, output count, active tier, and latency in milliseconds.
- **Operational Fleet Management Advisory:** Recommendations for bike rebalancing thresholds and surplus/deficit alerts based on predicted demand tiers.

---

## 9. Conclusion & Recommendation

All functional requirements, architectural constraints, statistical EDA objectives, failure taxonomies, and UI standards defined in **SRS v1.5** for Sprint 2 are fully implemented, verified, and ready for commit to GitHub.

**Final Recommendation:** **READY TO CLOSE SPRINT 2.**
