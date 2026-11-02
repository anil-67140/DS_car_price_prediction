# Model card - used-car asking-price estimator

**Version:** `1.0.0+20261101013441.189781cd`   **Model:** `svr` (tuned)   **Feature set:** `base_plus_parsed`

## Intended use
Estimate the **asking price in INR** of a used passenger car in the Indian classified-listing market, with a
90% prediction interval, as decision support for sellers/buyers/analysts.
It is **not** an appraisal, transaction-price or ex-showroom-price model, and has no knowledge of vehicle condition,
accident history, city, or listing date.

## Data
quikr - 581 training / 145 held-out listings after cleaning (892 raw rows).
Snapshot year: 2019 (prices are circa that year; **not inflation-adjusted to today**). Licence: Not stated. The CSV is a scrape of public Quikr classified listings redistributed through the author's own repository. Treat as educational/portfolio use only; do not redistribute commercially. Verify terms before any commercial use.

## Held-out performance (test split, evaluated once)
| metric | model | median baseline |
|---|---:|---:|
| MAE (INR) | 102,387 (95% CI 74,227-137,105) | 209,637 |
| RMSE (INR) | 218,297 | 365,290 |
| R2 | 0.622 (95% CI 0.19-0.79) | -0.057 |
| median abs error (INR) | 42,096 | 140,001 |
| RMSLE | 0.378 | 0.804 |
| MAPE % | 29.3 | 86.2 |

## Prediction intervals
Mondrian conformal on log-error; target coverage 90%, **empirical test coverage 86.2%
(+/- 2.9% s.e., n=145)**, median relative width 106%.
Guarantee is marginal and assumes new listings resemble the training data.

## Limitations
- Small dataset (<1k rows); rare models and luxury brands are poorly covered - expect wider real errors there.
- No condition, location, owner-count, transmission or engine fields in this source; only partial hints parsed from listing text.
- Asking prices, circa 2019. Using the model for current-year valuation without recalibration is invalid.
- Explanations (SHAP) describe model behaviour, not causal effects.
