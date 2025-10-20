# Salary Optimization Framework for WNBA Players

## Performance index modeling (phase 4)

this repository holds all scripts and outputs for the nba–wnba performance index project.  
the phase 4 portion includes data cleaning, model training, feature importance analysis, and performance index derivation.

### structure

- `data_merge/` → joins salary data and preps inputs
- `modeling/` → contains all modeling scripts, data outputs, and cleaned files
  - `derive_feature_importance_v1.py` – baseline feature importance
  - `derive_feature_importance_v2.py` – season-weighted version
  - `derive_performance_index_v1.py` – initial performance index
  - `derive_performance_index_v2.py` – refined version
  - `prepare_model.py`, `train_models.py`, `data_cleaning.py`
  - outputs are saved under `model_outputs/` and `performance_index_outputs/`

### notes

- `v1` scripts treat all 25 seasons as a single dataset.
- `v2` scripts calculate feature importances per season, normalize, and average across years.
- this improves correlation between modeled performance index and salary (from ~0.36 → ~0.47).

to rerun the pipeline:
1. run `data_cleaning.py`
2. run `prepare_model.py`
3. run `derive_feature_importance_v2.py`
4. run `derive_performance_index_v2.py`

