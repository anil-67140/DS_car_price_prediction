# EDA findings (training split only)

- **Price is strongly right-skewed** (skewness 2.81; mean ₹395,368 vs median ₹289,999). The middle car costs about ₹2.9 lakh while the top 1% exceed ₹18.9 lakh. After a log transform skewness drops to -0.04, so models are trained on log(price) and judged by relative error (RMSLE/MAPE) as well as rupees.

- **Brand drives price level**: Kruskal-Wallis across the 10 largest brands H = 108.3, p = 3.3e-19. Three brands account for 56% of listings, so the model is well informed about mass-market cars and thin on luxury brands (see per-segment errors).

- **Model coverage is sparse**: 128 model families, but 50% of them have at most 2 listings (median 2). A one-hot model column for a rare model cannot be learned reliably, which is why rare categories are pooled and why brand-level and engineered features matter.

- **Depreciation**: each additional year of age is associated with about 11.0% lower asking price (log-linear fit, R² = 0.33, 95% CI on the slope ±0.014). Age alone explains 33% of log-price variance; this is association, not a causal depreciation law, and the true curve is not perfectly linear.

- **Mileage is partly a proxy for age**: raw Spearman ρ(price, km) = -0.26, but after removing the age trend the residual correlation is only -0.05 (p = 0.23). Older cars have more km, so mileage adds less than its raw correlation suggests; odometer values under 1,000 km on cars >3 years old were treated as entry errors.

- **Diesel vs petrol**: median diesel ask ₹390,000 (95% CI 351,000–425,000) vs petrol ₹230,000 (210,000–250,000), a raw gap of +70% (Mann-Whitney p = 9.8e-16; a random diesel car out-prices a random petrol car 69% of the time). Median age is similar (5 vs 6 years), so age does not explain the gap. Holding model family and age fixed (regression with 65 model-family effects) the diesel premium is +8% - the difference between the two numbers is how much of the raw gap is simply that diesel buyers choose different (pricier) models. LPG has only 2 rows, so it cannot be modelled reliably.

- **What the data does not contain**: no transmission, owner-count, engine, power or location columns. Only 7% of listings state a gearbox in their text and 24% state an engine size. Those partial signals are used as clearly-named, nullable features (transmission_hint, engine_l) and NOT presented as complete transmission/engine data.

- **Multicollinearity**: highest VIF is 2.5 (kms_per_year). No serious multicollinearity among the numeric inputs (all VIF < 5). Age, odometer and km/year are related by construction, but not so tightly that OLS coefficients are unstable. The much larger instability risk is the high-cardinality, sparse one-hot brand/model columns, which is why regularised models and rare-category pooling are used.

- **Segments**: premium/luxury listings are only 28 of 581 rows (4.8%) but sell at 5.3x the median mass-market price. Their errors dominate RMSE; they are evaluated as a dedicated segment and are never removed as 'outliers'.