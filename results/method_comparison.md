# Method comparison — peak detector vs FFT spectral HR

Same input signals, same FIR pre-stage, same SQI gate; only the C-level HR estimator differs. Metrics computed on the **estimable** subset (SQI-accepted ∧ embedded HR > 0); algorithm-failure rate reported separately.

| Metric | Peak detector | FFT spectral |
|---|---:|---:|
| n (estimable) | 772 | 794 |
| MAE (bpm) | 2.33 | 1.72 |
| RMSE (bpm) | 6.81 | 5.96 |
| % within ±5 bpm | 89.8 | 93.8 |
| % within ±3 bpm | 85.4 | 91.6 |
| Bland-Altman bias | -1.72 | -0.67 |
| BA 95% LoA (lower) | -14.64 | -12.28 |
| BA 95% LoA (upper) | +11.19 | +10.94 |
| Pearson r | 0.884 | 0.906 |
| Algorithm-failure % | 2.8 | 0.0 |

## Design trade-offs

- **Peak detector (time-domain):** continuous HR estimate (no bin quantisation), simple, requires no FFT memory. Fails when peak detection fails (low SNR, motion).
- **FFT spectral (frequency-domain):** HR resolution = fs_ds/N × 60 = 3.66 bpm/bin before parabolic interp (~0.4 bpm after). More robust to occasional missed peaks (energy is integrated across the cycle). Costs +2 kB flash (FFT code + twiddle + Hamming) and +1 kB SRAM (256-pt complex Q15 workspace).

The two methods are not directly substitutable: the peak detector gives instantaneous HR per beat (with the 1-sample resolution of the filtered signal), while the FFT averages over the whole window. The right choice depends on whether the application prefers latency (peak detector) or robustness (FFT).
