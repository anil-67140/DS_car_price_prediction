| step | before | after | removed | note |
|---|---:|---:|---:|---|
| raw | 892 | 892 | 0 | as loaded (all columns read as text) |
| recover shifted fuel/kms fields | 892 | 892 | 0 | 2 rows had fuel type in kms column: fuel recovered, kms set missing |
| drop non-numeric year (mis-scraped ad text) | 892 | 842 | 50 |  |
| drop missing price ('Ask For Price') | 842 | 819 | 23 | no target label -> unusable |
| drop implausible model year | 819 | 819 | 0 | valid range 1990..2019 |
| drop unknown/non-car brand | 819 | 819 | 0 | brand not in the supported canonical list (e.g. motorbike or ad text in brand field) |
| drop price outside hard limits | 819 | 819 | 0 | 10,000..50,000,000 INR |
| drop exact duplicate listings | 819 | 727 | 92 | 92 exact repeats of an earlier row; kept first occurrence |
| kms: unrealistic -> set missing (rows kept) | 727 | 727 | 0 | 0 above 600,000 km, 6 cars >= 3y old with < 1000 km (likely unit/entry error) |
| drop clear price typos (> 10x family median) | 727 | 726 | 1 | 2 flagged for review (> 6.0x), 1 dropped; valid luxury / rare cars deliberately kept |
| final | 726 | 726 | 0 | clean dataset |