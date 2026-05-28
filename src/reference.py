#!/usr/bin/env python3
"""Golden reference for the embedded PPG heart-rate estimator.

Generates the fixed-point test vector + band-pass FIR coefficients
(firmware/generated/ppg_data.h) and a golden.json that the QEMU output is
validated against (src/validate.py).

Runs on the Python standard library alone (synthetic PPG + windowed-sinc FIR).
If numpy/scipy/wfdb are installed (see pyproject.toml), `--record DB/REC` instead
loads a real PhysioNet PPG record and designs the FIR with scipy.firwin.
"""
from __future__ import annotations
import argparse
import cmath
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "firmware" / "generated"


def q15(x: float) -> int:
    return max(-32768, min(32767, int(round(x * 32768.0))))


def refractory_samples(fs: float) -> int:
    """Peak-detector refractory window in samples (~0.33 s -> blocks dicrotic notch up to ~180 bpm).
    Single source of truth: emitted into firmware/generated/ppg_data.h as REFRACTORY_SAMPLES so
    the C side cannot drift from Python (truncation vs banker-rounding at non-default fs)."""
    return int(round(0.33 * fs))


def design_fir_bandpass(numtaps: int, f1: float, f2: float, fs: float) -> list[float]:
    """Windowed-sinc (Hamming) band-pass FIR, normalized to unity gain at band center."""
    M = numtaps - 1
    fc1, fc2 = f1 / fs, f2 / fs
    h = []
    for n in range(numtaps):
        k = n - M / 2.0
        if abs(k) < 1e-9:
            val = 2.0 * (fc2 - fc1)
        else:
            val = (math.sin(2 * math.pi * fc2 * k) - math.sin(2 * math.pi * fc1 * k)) / (math.pi * k)
        w = 0.54 - 0.46 * math.cos(2 * math.pi * n / M)
        h.append(val * w)
    fc = 0.5 * (f1 + f2) / fs
    g = abs(sum(h[n] * cmath.exp(-2j * math.pi * fc * n) for n in range(numtaps)))
    return [v / g for v in h] if g > 0 else h


def synth_ppg(fs: float, dur: float, hr_bpm: float) -> list[float]:
    n = int(round(fs * dur))
    f0 = hr_bpm / 60.0
    random.seed(0)
    x = []
    for i in range(n):
        t = i / fs
        s = math.sin(2 * math.pi * f0 * t)
        s += 0.4 * math.sin(2 * math.pi * 2 * f0 * t + 0.5)   # dicrotic-ish harmonic
        s += 0.3 * math.sin(2 * math.pi * 0.25 * t)           # baseline wander
        s += 0.05 * (2 * random.random() - 1)                 # mild noise
        x.append(s)
    m = max(abs(v) for v in x) or 1.0
    return [0.9 * v / m for v in x]


def fir_apply(x: list[float], h: list[float]) -> list[float]:
    nt = len(h)
    return [sum(h[k] * x[i - k] for k in range(nt)) for i in range(nt - 1, len(x))]


def detect_peaks(y: list[float], refractory: int) -> list[int]:
    """Return the list of peak indices: local maxima above max|y|/2, separated by
    >= refractory samples. Mirrors firmware/dsp_fixed.c:detect_peaks exactly."""
    maxabs = max((abs(v) for v in y), default=0.0)
    thr = maxabs / 2.0
    indices, last = [], -refractory
    for i in range(1, len(y) - 1):
        if y[i] > thr and y[i] >= y[i - 1] and y[i] > y[i + 1] and (i - last) >= refractory:
            indices.append(i)
            last = i
    return indices


def hr_x100_from_peaks(peak_indices: list[int], fs: float) -> int:
    """HR (× 100) from median inter-peak interval. Returns 0 if fewer than 2 valid
    intervals. Intervals outside [0.4 s, 1.5 s] (HR ∉ [40, 150] bpm) are rejected as
    outliers — the source of the algorithm's robustness vs naive count-based HR.
    Mirrors firmware/dsp_fixed.c:hr_x100_from_peaks exactly."""
    if len(peak_indices) < 2:
        return 0
    min_int = int(fs * 0.4)
    max_int = int((fs * 15) / 10)  # 1.5 s, via integer math matching C
    intervals = []
    for i in range(1, len(peak_indices)):
        d = peak_indices[i] - peak_indices[i - 1]
        if min_int <= d <= max_int:
            intervals.append(d)
    if not intervals:
        return 0
    intervals.sort()
    median_d = intervals[len(intervals) // 2]
    # HR_x100 = 60 * 100 * fs / median_d  (integer arithmetic mirroring C)
    return (6000 * int(round(fs))) // median_d


def emit_array(name: str, vals: list[int], ty: str = "int16_t") -> str:
    lines = ["  " + ", ".join(f"{v:6d}" for v in vals[i:i + 12]) + ","
             for i in range(0, len(vals), 12)]
    return f"static const {ty} {name} = {{\n" + "\n".join(lines) + "\n};\n"


def design_fft_tables(n_fft: int = 256, fs_in: float = 125.0, decim: int = 8,
                      hr_lo_hz: float = 0.7, hr_hi_hz: float = 3.5):
    """Generate Q15 Hamming window + twiddle table for an N-point radix-2 FFT.

    The PPG signal has already been band-passed to [hr_lo_hz, hr_hi_hz] Hz, so
    decimating by `decim` is anti-aliasing-free as long as fs_in/(2*decim) > hr_hi_hz.
    The decimated sample rate is fs_ds = fs_in / decim; bin width is fs_ds / N.

    Returns (fs_ds, k_min, k_max, hamming_q15, twiddle_q15) where twiddle_q15 is a
    list of (re, im) tuples for W_N^k = cos(2πk/N) − j sin(2πk/N), k=0..N/2-1.
    """
    n_log2 = (n_fft - 1).bit_length()
    if (1 << n_log2) != n_fft:
        raise ValueError(f"n_fft must be a power of 2, got {n_fft}")
    fs_ds = fs_in / decim
    if fs_ds / 2.0 <= hr_hi_hz:
        raise ValueError(
            f"decim={decim} too aggressive: fs_ds/2 = {fs_ds/2:.2f} Hz ≤ "
            f"hr_hi_hz = {hr_hi_hz} Hz — aliasing risk")
    k_min = max(1, int(round(hr_lo_hz * n_fft / fs_ds)))
    k_max = min(n_fft // 2 - 2, int(round(hr_hi_hz * n_fft / fs_ds)))
    # Hamming window over N samples
    hamming = [0.54 - 0.46 * math.cos(2 * math.pi * n / (n_fft - 1)) for n in range(n_fft)]
    hamming_q = [q15(v) for v in hamming]
    # Twiddle: W_N^k = exp(-j 2π k / N) = cos(2πk/N) − j sin(2πk/N), k=0..N/2-1
    twiddle = []
    for k in range(n_fft // 2):
        theta = 2 * math.pi * k / n_fft
        twiddle.append((q15(math.cos(theta)), q15(-math.sin(theta))))
    return fs_ds, k_min, k_max, hamming_q, twiddle


def _fft_radix2_stdlib(x: list[complex]) -> list[complex]:
    """Recursive Cooley-Tukey radix-2 DIT FFT in pure stdlib (cmath). O(N log N).
    Mirrors numpy.fft.fft(x) for power-of-2 N; used when numpy is unavailable so
    the default reference.py path keeps working on bare stdlib."""
    n = len(x)
    if n <= 1:
        return list(x)
    if n & (n - 1):
        raise ValueError(f"_fft_radix2_stdlib requires power-of-2 length, got {n}")
    even = _fft_radix2_stdlib(x[0::2])
    odd  = _fft_radix2_stdlib(x[1::2])
    t = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    return ([even[k] + t[k] for k in range(n // 2)]
          + [even[k] - t[k] for k in range(n // 2)])


def fft_hr_x100_reference(filt: list[float], fs_in: float, n_fft: int, decim: int,
                          hamming_q: list[int], k_min: int, k_max: int) -> int:
    """Float-pipeline FFT-based HR (×100): decimate, Hamming-window, FFT, find peak
    in [k_min, k_max], parabolic-interpolate. The C firmware mirrors this in Q15;
    the float result here is the golden reference for HR comparison.

    Uses numpy.fft if available, falling back to a stdlib radix-2 FFT so the
    default `python3 src/reference.py` path remains dependency-free."""
    fs_ds = fs_in / decim
    x_ds = [filt[i] for i in range(0, min(len(filt), n_fft * decim), decim)]
    if len(x_ds) >= n_fft:
        x_ds = x_ds[:n_fft]
    else:
        x_ds = x_ds + [0.0] * (n_fft - len(x_ds))
    win = [w / 32768.0 for w in hamming_q]
    x_windowed = [x_ds[i] * win[i] for i in range(n_fft)]
    try:
        import numpy as np
        X = np.fft.fft(np.array(x_windowed, dtype=float))
        mag2 = [float(abs(c)) ** 2 for c in X]
    except ImportError:
        X = _fft_radix2_stdlib([complex(v, 0.0) for v in x_windowed])
        mag2 = [(c.real * c.real + c.imag * c.imag) for c in X]
    # Peak bin in [k_min, k_max]
    k_peak = k_min
    best = mag2[k_min]
    for k in range(k_min + 1, k_max + 1):
        if mag2[k] > best:
            best = mag2[k]
            k_peak = k
    if k_peak <= 0 or k_peak >= n_fft - 1:
        return 0
    y_m1, y_0, y_p1 = mag2[k_peak - 1], mag2[k_peak], mag2[k_peak + 1]
    denom = y_m1 - 2 * y_0 + y_p1
    delta = 0.5 * (y_m1 - y_p1) / denom if denom != 0 else 0.0
    if delta < -0.5: delta = -0.5
    if delta >  0.5: delta =  0.5
    f_peak = (k_peak + delta) * fs_ds / n_fft
    hr_bpm = f_peak * 60.0
    return int(round(hr_bpm * 100))


def write_fft_data_header(hamming_q: list[int], twiddle: list[tuple[int, int]],
                          fs_in: float, fs_ds: float, decim: int,
                          k_min: int, k_max: int, path,
                          subharmonic_divisor: int = 3) -> None:
    """Emit firmware/generated/fft_data.h — Q15 Hamming + twiddle tables + bin range
    + sub-harmonic check threshold. AUTO-GENERATED contract between reference.py
    and the C side; never hand-edit.

    `subharmonic_divisor` sets the threshold for the sub-harmonic swap in
    `main_fft.c` (α = 1 / DIVISOR).
      DIVISOR=2 (α=0.5):  looser threshold. 6 of 9 bidmc47 2×-lock windows
                          fixed; 3 remain locked.
      DIVISOR=3 (α≈0.33): default. All 9 bidmc47 windows fixed; introduces
                          2 false 1/2× swaps on bidmc35 t=30 and bidmc40
                          t=30 (peak detector is correct at ~110 bpm; FFT
                          over-swaps to ~55). Net headline improvement
                          (MAE −6 %, RMSE −16 %, r +0.030) — the bidmc47
                          wins outweigh the bidmc35/40 losses.
    """
    n_fft = len(hamming_q)
    n_log2 = (n_fft - 1).bit_length()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("/* AUTO-GENERATED by src/reference.py - do not edit by hand. */\n")
        f.write("#ifndef FFT_DATA_H\n#define FFT_DATA_H\n")
        f.write("#include <stdint.h>\n#include \"../dsp_fft.h\"\n\n")
        f.write(f"#define FFT_N                    {n_fft}\n")
        f.write(f"#define FFT_N_LOG2               {n_log2}\n")
        f.write(f"#define FFT_DECIM                {decim}\n")
        f.write(f"#define FFT_FS_IN_HZ             {int(round(fs_in))}\n")
        f.write(f"#define FFT_FS_DS_X1000          {int(round(fs_ds * 1000))}\n")
        f.write(f"#define FFT_K_MIN                {k_min}\n")
        f.write(f"#define FFT_K_MAX                {k_max}\n")
        f.write(f"#define FFT_SUBHARMONIC_DIVISOR  {subharmonic_divisor}\n\n")
        f.write(emit_array(f"fft_window_q15[FFT_N]", hamming_q))
        f.write("\n")
        f.write(f"static const cplx_q15_t fft_twiddle_q15[FFT_N / 2] = {{\n")
        for i in range(0, len(twiddle), 6):
            row = ", ".join(f"{{{r:6d},{im:6d}}}" for r, im in twiddle[i:i + 6])
            f.write(f"  {row},\n")
        f.write("};\n")
        f.write("\n#endif /* FFT_DATA_H */\n")


def load_record(record_id: str, dur: float = 10.0, taps: int = 101,
                f1: float = 0.7, f2: float = 3.5, cache_dir: str | None = None):
    """Load a PhysioNet PPG record and design a matching band-pass FIR.
    Returns (x: list[float], fs: float, h: list[float]).
    Requires numpy/scipy/wfdb (declared optional in pyproject.toml). If cache_dir is
    given and the record's .hea file lives there, no PhysioNet network access is needed."""
    import numpy as np
    import wfdb
    from scipy.signal import firwin
    rec_path = record_id
    if cache_dir is not None:
        local = Path(cache_dir) / record_id
        if local.with_suffix('.hea').exists():
            rec_path = str(local)
    rec = wfdb.rdrecord(rec_path)
    names = [s.lower() for s in rec.sig_name]
    idx = next((i for i, nm in enumerate(names) if "pleth" in nm or "ppg" in nm), 0)
    fs = float(rec.fs)
    sig = np.asarray(rec.p_signal[:, idx], dtype=float)
    sig = sig[~np.isnan(sig)]
    n = min(len(sig), int(fs * dur))
    sig = sig[:n] - sig[:n].mean()
    m = float(np.max(np.abs(sig))) or 1.0
    x = (0.9 * sig / m).tolist()
    h = firwin(taps, [f1, f2], fs=fs, pass_zero=False).tolist()
    return x, fs, h


def write_ppg_data_header(ppg_q: list[int], fir_q: list[int], fs: float, n: int,
                          refractory: int, ref_hr_x100: int, path) -> None:
    """Emit firmware/generated/ppg_data.h. Pure I/O — no DSP. Single source of truth
    for the Python<->C contract (PPG_FS, PPG_N, FIR_NTAPS, REFRACTORY_SAMPLES, REF_HR_X100,
    plus the ppg_q15[] and fir_q15_coef[] arrays)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("/* AUTO-GENERATED by src/reference.py - do not edit by hand. */\n")
        f.write("#ifndef PPG_DATA_H\n#define PPG_DATA_H\n#include <stdint.h>\n\n")
        f.write(f"#define PPG_FS              {int(round(fs))}\n")
        f.write(f"#define PPG_N               {n}\n")
        f.write(f"#define FIR_NTAPS           {len(fir_q)}\n")
        f.write(f"#define REFRACTORY_SAMPLES  {refractory}\n")
        f.write(f"#define REF_HR_X100         {ref_hr_x100}\n\n")
        f.write(emit_array("ppg_q15[PPG_N]", ppg_q))
        f.write("\n")
        f.write(emit_array("fir_q15_coef[FIR_NTAPS]", fir_q))
        f.write("\n#endif /* PPG_DATA_H */\n")


def design_resp_tables(num_taps: int, cutoff_hz: float, fs_in: float, decim: int,
                       freqs_hz: list[float]) -> tuple[list[int], int, list[int], list[int]]:
    """Design the BW-path RR tables.

    Returns (fir_rr_q, fs_ds_x1000, cos_q15_list, freqs_x1000_list):
      - `fir_rr_q`         : Q15 lowpass FIR coefficients (Hamming-window sinc).
      - `fs_ds_x1000`      : decimated sample rate * 1000, as int (for header constant).
      - `cos_q15_list`     : q15(cos(2π·f/fs_ds)) per candidate frequency.
      - `freqs_x1000_list` : frequency in mHz, integer (for runtime BrPM conversion).
    """
    # Lowpass FIR, normalized to DC gain = 1
    M = num_taps - 1
    fc = cutoff_hz / fs_in
    h: list[float] = []
    for n in range(num_taps):
        k = n - M / 2.0
        if abs(k) < 1e-12:
            v = 2.0 * fc
        else:
            v = math.sin(2.0 * math.pi * fc * k) / (math.pi * k)
        w = 0.54 - 0.46 * math.cos(2.0 * math.pi * n / M)
        h.append(v * w)
    s = sum(h) or 1.0
    h = [v / s for v in h]
    fir_rr_q = [q15(v) for v in h]

    fs_ds = fs_in / decim
    cos_q15_list = [q15(math.cos(2.0 * math.pi * f / fs_ds)) for f in freqs_hz]
    freqs_x1000_list = [int(round(f * 1000.0)) for f in freqs_hz]
    return fir_rr_q, int(round(fs_ds * 1000.0)), cos_q15_list, freqs_x1000_list


def write_resp_data_header(fir_rr_q: list[int], fs_in: float, decim: int,
                           fs_ds_x1000: int,
                           cos_q15_list: list[int], freqs_x1000_list: list[int],
                           path) -> None:
    """Emit firmware/generated/resp_data.h — BW-path RR tables.

    AUTO-GENERATED contract between reference.py and firmware/main_rr.c +
    firmware/dsp_resp.c; never hand-edit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("/* AUTO-GENERATED by src/reference.py - do not edit by hand. */\n")
        f.write("#ifndef RESP_DATA_H\n#define RESP_DATA_H\n#include <stdint.h>\n\n")
        f.write(f"#define FIR_RR_NTAPS         {len(fir_rr_q)}\n")
        f.write(f"#define DECIM_RR             {decim}\n")
        f.write(f"#define RESP_FS_IN_HZ        {int(round(fs_in))}\n")
        f.write(f"#define RESP_FS_DS_X1000     {fs_ds_x1000}\n")
        f.write(f"#define GOERTZEL_K           {len(freqs_x1000_list)}\n\n")
        f.write(emit_array("fir_rr_coef[FIR_RR_NTAPS]", fir_rr_q))
        f.write("\n")
        f.write(emit_array("goertzel_cos_q15[GOERTZEL_K]", cos_q15_list))
        f.write("\n")
        f.write(emit_array("goertzel_freqs_x1000[GOERTZEL_K]", freqs_x1000_list, ty="uint16_t"))
        f.write("\n#endif /* RESP_DATA_H */\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fs", type=float, default=125.0)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--hr", type=float, default=72.0, help="synthetic ground-truth HR (bpm)")
    ap.add_argument("--taps", type=int, default=101)
    ap.add_argument("--f1", type=float, default=0.7, help="band-pass low cutoff (Hz)")
    ap.add_argument("--f2", type=float, default=3.5, help="band-pass high cutoff (Hz)")
    ap.add_argument("--record", default=None, help="PhysioNet record (needs wfdb/numpy/scipy)")
    ap.add_argument("--fft-n", type=int, default=256,
                    help="FFT size for the spectral HR path (power of 2; default 256)")
    ap.add_argument("--fft-decim", type=int, default=8,
                    help="decimation factor before FFT (must keep fs/(2*decim) > f2; default 8)")
    args = ap.parse_args()

    fs, gt_hr, used_real = args.fs, args.hr, False

    if args.record:
        try:
            x, fs, h = load_record(args.record, args.dur, args.taps, args.f1, args.f2)
            used_real, gt_hr = True, float("nan")
        except Exception as e:  # noqa: BLE001 - fall back gracefully
            print(f"[reference] real-data path unavailable ({e}); using synthetic")
            x = synth_ppg(fs, args.dur, args.hr)
            h = design_fir_bandpass(args.taps, args.f1, args.f2, fs)
    else:
        x = synth_ppg(fs, args.dur, args.hr)
        h = design_fir_bandpass(args.taps, args.f1, args.f2, fs)

    n = len(x)
    refractory = refractory_samples(fs)
    yf = fir_apply(x, h)
    peaks = detect_peaks(yf, refractory)
    L = len(yf)
    hr_x100 = hr_x100_from_peaks(peaks, fs)
    ref_hr = hr_x100 / 100.0

    ppg_q = [q15(v) for v in x]
    fir_q = [q15(v) for v in h]

    hdr = GEN / "ppg_data.h"
    write_ppg_data_header(ppg_q, fir_q, fs, n, refractory, int(round(ref_hr * 100)), hdr)

    # FFT path — emit tables and compute the float-reference FFT HR for golden.json
    fs_ds, k_min, k_max, hamming_q, twiddle = design_fft_tables(
        n_fft=args.fft_n, fs_in=fs, decim=args.fft_decim,
        hr_lo_hz=args.f1, hr_hi_hz=args.f2,
    )
    fft_hdr = GEN / "fft_data.h"
    write_fft_data_header(hamming_q, twiddle, fs, fs_ds, args.fft_decim, k_min, k_max, fft_hdr)
    fft_hr_x100 = fft_hr_x100_reference(yf, fs, args.fft_n, args.fft_decim,
                                         hamming_q, k_min, k_max)
    fft_hr = fft_hr_x100 / 100.0

    # BW-only RR path. 24 candidate frequencies, 6-30 BrPM at 1-BrPM
    # resolution. The C side reads these tables via firmware/generated/resp_data.h.
    rr_freqs_hz = [0.10 + 0.40 * k / 23 for k in range(24)]
    rr_decim = 32
    fir_rr_q, fs_ds_x1000, cos_q15_list, freqs_x1000_list = design_resp_tables(
        num_taps=51, cutoff_hz=0.5, fs_in=fs, decim=rr_decim, freqs_hz=rr_freqs_hz)
    resp_hdr = GEN / "resp_data.h"
    write_resp_data_header(fir_rr_q, fs, rr_decim, fs_ds_x1000,
                            cos_q15_list, freqs_x1000_list, resp_hdr)

    golden = {
        "fs": fs, "n": n, "taps": len(fir_q), "valid_len": L,
        "refractory": refractory, "peaks_float": len(peaks),
        "ref_hr_float": ref_hr, "gt_hr": gt_hr, "used_real_data": used_real,
        "fft_n": args.fft_n, "fft_decim": args.fft_decim, "fft_fs_ds": fs_ds,
        "fft_k_min": k_min, "fft_k_max": k_max,
        "fft_hr_float": fft_hr,
    }
    (GEN / "golden.json").write_text(json.dumps(golden, indent=2))

    print(f"[reference] {'real' if used_real else 'synthetic'} PPG | fs={fs:.0f} N={n} taps={len(fir_q)}")
    print(f"[reference] peak-detector path: peaks={len(peaks)}  HR={ref_hr:.2f} bpm  (ground truth={gt_hr})")
    print(f"[reference] FFT path (N={args.fft_n}, decim={args.fft_decim}, fs_ds={fs_ds:.3f}): HR={fft_hr:.2f} bpm")
    print(f"[reference] RR path (BW-only, 51-tap LP @ 0.5 Hz, decim={rr_decim}, "
          f"fs_ds={fs_ds_x1000/1000.0:.3f} Hz, {len(rr_freqs_hz)} Goertzel bins)")
    print(f"[reference] wrote {hdr.relative_to(ROOT)}, {fft_hdr.relative_to(ROOT)}, "
          f"{resp_hdr.relative_to(ROOT)}, and golden.json")


if __name__ == "__main__":
    main()
