# 1D-CNN SQI vs variance SQI — held-out PPG-DaLiA subjects

Both gates run on the same FFT-path HR estimates over 8-s windows. Variance gate rejects the bottom 10 % of windows by band-passed variance; CNN gate uses a tiny 1D-CNN (~1k parameters) trained on the other 13 subjects.

| Subject | n windows | variance MAE | variance accept | CNN MAE | CNN accept | Δ-MAE |
|---|---:|---:|---:|---:|---:|---:|
| S6 | 2622 | 13.70 | 90.0% | 5.76 | 71.7% | +7.94 |
| S15 | 3966 | 10.94 | 90.0% | 4.91 | 66.0% | +6.03 |
| **POOLED** | 6588 | 12.04 | 90.0% | 5.27 | 68.3% | +6.77 |
