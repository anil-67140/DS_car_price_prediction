# Model selection

**Selected: `svr` on feature set `base_plus_parsed`** (tuned)

Best CV MAE: svr (93,436 INR, SE 3,400). 1 model(s) are statistically tied with it (MAE <= 96,836 and RMSE <= 193,929). Among those the simplest (interpretability tier 4, 24 KB) is 'svr'.

Selection rule: best CV MAE; every model within one standard error on MAE *and* RMSE is a tie; ties are broken by interpretability tier, then model size, then MAE. R² is reported but is not the criterion.

| model | variant | tuned | mae_inr | mae_se | rmse_inr | r2 | rmsle | tier | model_kb | predict_ms_per_row | eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| svr | base_plus_parsed | True | 93,435.928 | 3,399.822 | 179,974.708 | 0.752 | 0.381 | 4 | 24.400 | 0.185 | True |
| huber | base_plus_parsed | False | 97,341.341 | 3,517.555 | 213,892.170 | 0.623 | 0.401 | 1 | 5.100 | 0.131 | False |
| linear | base_plus_parsed | False | 100,512.559 | 3,645.144 | 192,704.680 | 0.714 | 0.395 | 1 | 5.700 | 0.144 | False |
| catboost | base_plus_parsed | True | 100,746.173 | 3,702.597 | 189,063.586 | 0.730 | 0.393 | 3 | 52.000 | 0.146 | False |
| extra_trees | target_encoded | True | 101,339.372 | 3,307.448 | 186,701.406 | 0.739 | 0.388 | 3 | 3,394.100 | 0.973 | False |
| extra_trees | target_encoded | False | 101,765.189 | 3,311.224 | 187,259.351 | 0.739 | 0.387 | 3 | 3,488.600 | 0.780 | False |
| svr | base_plus_parsed | False | 103,004.952 | 3,366.831 | 198,105.903 | 0.707 | 0.402 | 4 | 25.900 | 0.168 | False |
| ebm | base_plus_parsed | False | 105,321.301 | 3,373.684 | 201,388.764 | 0.701 | 0.393 | 1 | 226.600 | 0.158 | False |
| catboost | base_plus_parsed | False | 105,382.367 | 3,633.377 | 198,176.910 | 0.709 | 0.393 | 3 | 102.600 | 0.168 | False |
| catboost | target_encoded | True | 105,672.719 | 3,112.424 | 195,654.711 | 0.719 | 0.389 | 3 | 119.600 | 0.193 | False |
| catboost | target_encoded | False | 105,800.514 | 3,598.702 | 198,859.282 | 0.707 | 0.389 | 3 | 133.900 | 0.158 | False |
| xgboost | base_plus_parsed | False | 106,135.181 | 3,305.063 | 195,345.114 | 0.714 | 0.394 | 3 | 110.900 | 0.159 | False |
| gradient_boosting | target_encoded | False | 106,305.555 | 3,557.492 | 192,571.778 | 0.724 | 0.396 | 3 | 109.900 | 0.176 | False |
| xgboost | target_encoded | False | 106,332.046 | 3,596.259 | 194,511.287 | 0.720 | 0.396 | 3 | 141.400 | 0.162 | False |
| random_forest | target_encoded | False | 106,535.381 | 3,451.474 | 196,985.978 | 0.714 | 0.394 | 3 | 2,180.100 | 0.778 | False |
| hist_gradient_boosting | target_encoded | False | 107,119.384 | 3,758.936 | 194,236.149 | 0.719 | 0.404 | 3 | 184.100 | 0.194 | False |
| ridge | base_plus_parsed | False | 107,814.289 | 3,713.511 | 198,635.451 | 0.709 | 0.398 | 1 | 4.900 | 0.129 | False |
| lightgbm | target_encoded | False | 108,343.995 | 3,695.380 | 198,533.003 | 0.710 | 0.399 | 3 | 205.800 | 0.177 | False |
| gradient_boosting | base_plus_parsed | False | 108,733.542 | 3,107.473 | 196,335.211 | 0.712 | 0.401 | 3 | 93.700 | 0.162 | False |
| extra_trees | base_plus_parsed | False | 109,229.481 | 3,708.944 | 202,405.310 | 0.697 | 0.420 | 3 | 3,511.600 | 0.801 | False |
| ebm | target_encoded | False | 110,341.096 | 3,579.480 | 201,001.443 | 0.703 | 0.400 | 1 | 957.600 | 0.179 | False |
| elasticnet | base_plus_parsed | False | 111,051.001 | 3,393.196 | 197,320.160 | 0.711 | 0.410 | 1 | 4.700 | 0.178 | False |
| lasso | base_plus_parsed | False | 114,860.656 | 3,298.752 | 201,083.911 | 0.699 | 0.419 | 1 | 4.500 | 0.137 | False |
| random_forest | base_plus_parsed | False | 119,132.265 | 3,786.734 | 215,913.872 | 0.657 | 0.431 | 3 | 2,121.200 | 0.726 | False |
| svr | target_encoded | False | 121,719.771 | 3,979.519 | 242,812.333 | 0.582 | 0.427 | 4 | 23.000 | 0.161 | False |
| decision_tree | target_encoded | False | 124,321.123 | 2,766.183 | 216,444.715 | 0.651 | 0.469 | 2 | 9.100 | 0.139 | False |
| ridge | target_encoded | False | 124,459.264 | 3,163.520 | 239,876.911 | 0.582 | 0.415 | 1 | 5.300 | 0.133 | False |
| elasticnet | target_encoded | False | 125,252.515 | 3,157.152 | 242,689.463 | 0.571 | 0.416 | 1 | 5.300 | 0.148 | False |
| lasso | target_encoded | False | 125,714.803 | 3,189.251 | 244,310.103 | 0.566 | 0.416 | 1 | 5.300 | 0.152 | False |
| lightgbm | base_plus_parsed | False | 127,055.229 | 2,695.274 | 228,534.035 | 0.615 | 0.443 | 3 | 195.200 | 0.167 | False |
| hist_gradient_boosting | base_plus_parsed | False | 127,530.799 | 2,489.058 | 231,805.370 | 0.603 | 0.450 | 3 | 177.800 | 0.178 | False |
| linear | target_encoded | False | 127,989.769 | 3,210.183 | 250,161.450 | 0.540 | 0.418 | 1 | 5.500 | 0.177 | False |
| huber | target_encoded | False | 128,411.992 | 3,039.818 | 252,655.206 | 0.529 | 0.420 | 1 | 5.500 | 0.152 | False |
| knn | base_plus_parsed | False | 130,581.710 | 3,886.071 | 259,075.366 | 0.517 | 0.465 | 3 | 30.000 | 0.139 | False |
| knn | target_encoded | False | 136,690.690 | 4,012.796 | 271,637.575 | 0.473 | 0.482 | 3 | 25.600 | 0.162 | False |
| decision_tree | base_plus_parsed | False | 137,819.553 | 3,693.667 | 239,404.761 | 0.573 | 0.516 | 2 | 6.800 | 0.116 | False |
| mlp | target_encoded | False | 138,136.815 | 4,552.671 | 283,990.114 | 0.407 | 0.446 | 4 | 94.900 | 0.150 | False |
| mlp | base_plus_parsed | False | 153,680.866 | 23,805.116 | 433,098.061 | -1.952 | 0.486 | 4 | 213.100 | 0.153 | False |