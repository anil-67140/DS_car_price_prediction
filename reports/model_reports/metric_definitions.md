# Metric definitions and limitations

| name | meaning | limits |
|---|---|---|
| Mean Absolute Error (INR) | Average size of the miss in rupees. Directly interpretable by a buyer or seller. | Treats all errors linearly; says nothing about relative error - a 50k miss is trivial on a 30 lakh car and huge on a 60k car. |
| Mean Squared Error (INR^2) | Average squared miss; the quantity least-squares models minimise. | Units are rupees squared (not interpretable); dominated by a few large errors. |
| Root Mean Squared Error (INR) | Typical miss in rupees with extra penalty for big misses. | Sensitive to outliers/expensive cars; RMSE >> MAE signals a heavy error tail. |
| R-squared | Share of price variance explained versus predicting the mean. | Scale-free but dominated by expensive cars; can look high while typical cars are badly mispriced; not comparable across datasets; can be negative out of sample. |
| Adjusted R-squared | R-squared penalised by the number of model inputs. | Only meaningful for (near-)linear models with n >> p. Computed with p = number of transformed input columns, so it is NaN when n <= p+1 and should NOT be used to rank non-parametric models. |
| Median Absolute Error (INR) | The miss that half of all predictions beat - robust to outliers. | Hides the tail: a model can have a great median and still be wildly wrong sometimes. |
| Mean Absolute Percentage Error (%) | Average relative miss; intuitive ('off by 15%'). | Undefined for zero prices; asymmetric (over-predictions can exceed 100% while under-predictions cap at 100%) so it rewards under-prediction; inflated by cheap cars. |
| Symmetric MAPE (%) | Relative error using the mean of actual and predicted in the denominator. | Bounded 0-200% but still asymmetric in subtle ways, and the definition varies between papers; use only for comparison within this report. |
| Root Mean Squared Log Error | RMSE on log(1+price): approximates typical RELATIVE error; ideal for right-skewed prices. | Requires non-negative prices/predictions; penalises under-prediction more than over; not in rupee units (0.20 is roughly a 20% typical miss). |
| Explained Variance Score | Like R-squared but ignores a constant bias in the predictions. | Can be high for a model that is systematically too high/low - always read with bias. |
| Maximum Error (INR) | Worst single miss on the evaluation set. | A single-observation statistic: very unstable and grows with sample size. |
| Mean Bias Error (INR) | Average signed error (prediction - actual). Positive = systematically over-prices. | Positive and negative errors cancel, so MBE near 0 does not mean accurate. |
| Mean Absolute Scaled Error | MAE divided by the MAE of a naive baseline (here: predict the TRAINING median price). MASE < 1 means the model beats that baseline. | Defined for time series with a seasonal-naive baseline; here the baseline is a constant median predictor, so it is a skill score vs 'no information', not a forecasting MASE. |