# Dataset research and selection
**Selected: Quikr cars** (the project's original data; 892 raw rows, 726 after cleaning, snapshot year 2019). Provenance, sha256 and row counts are in `reports/dataset_sources.json`.
Fields: name, company, year, price, kms_driven, fuel_type. **Licence/terms are not stated by the source**; treat as research/educational use and verify before commercial use.
**Considered:** CarDekho v3 (richer: transmission, owners, seller type, engine). It is registered in `configs/datasets.yaml` as *unverified*; it could not be downloaded here (kaggle.com is blocked in the build sandbox), so no results are claimed for it, and the serving layer does not yet support its extra inputs.
**Absent from Quikr:** transmission, owners, engine, power, condition, location, listing date. None are fabricated.
**Biases:** asking (not sale) prices; one snapshot; Indian metro-skewed listings; popular brands dominate (rare brands have few rows).
