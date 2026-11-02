# Monitoring and retraining design
- **Data drift** (inputs): PSI with a sample-size noise floor and rare-category pooling, plus KS tests, against the training reference profile (`scripts/monitor.py`).
- **Performance degradation** needs ground truth (verified sale prices). Without it the report says "not assessable" - accuracy is never invented. With it: relative MAE/RMSLE ratios and log-bias against test-time values.
- **Decision logic:** drift alone -> investigate; degradation (confirmed) -> retrain; both -> retrain + review. Suspicious-prediction flags catch out-of-support inputs.
- **Retraining** (`scripts/retrain.py`): champion/challenger on a fresh grouped holdout; promote only if the challenger wins by the same selection rule; registry pointer is switched atomically; old bundles stay.
- Mechanics were smoke-tested on synthetic input only - those results are not meaningful performance claims. No scheduler is deployed.
