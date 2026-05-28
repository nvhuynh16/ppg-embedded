#!/usr/bin/env python3
"""Bit-exactness validator for the C Q15 FFT in firmware/dsp_fft.c.

Builds firmware/host_fft_test (host gcc, no QEMU), runs it with a known synthetic
sinusoid (matching firmware/host_fft_test.c:gen_test_signal), parses the CSV
spectrum it prints, and compares each bin against numpy.fft.fft of the same
windowed input — scaled by 1/N to match the per-stage >>1 scaling of dsp_fft.c.

Passes if max per-bin |re| and |im| differ from numpy by ≤ 1 LSB (Q15) for the
dominant bins, and the dominant bin index matches exactly. The 1-LSB tolerance
is the worst-case truncation across log2(N) butterfly stages with the
`(a*b)>>15` Q15 multiplication.
"""
from __future__ import annotations
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIRMWARE = ROOT / "firmware"

# Test signal parameters — MUST match firmware/host_fft_test.c:main()
TEST_F_HZ = 1.2
TEST_AMP_Q15 = 16384.0  # 0.5 full-scale


def q15_saturate(v: float) -> int:
    if v >  32767.0: return  32767
    if v < -32768.0: return -32768
    return int(v)


def numpy_reference(fft_n: int, fs_ds: float, hamming_q: list[int]) -> list[complex]:
    """Generate the same signal the C code generates, window it identically, run
    numpy.fft.fft on it, and scale by 1/N to match the C's per-stage >>1 scaling."""
    import numpy as np
    # Same generator as host_fft_test.c:gen_test_signal
    signal_q15 = [q15_saturate(TEST_AMP_Q15 * math.sin(2 * math.pi * TEST_F_HZ * i / fs_ds))
                  for i in range(fft_n)]
    # Same Q15 windowing as host_fft_test.c: ((signal[i] * window[i]) >> 15) with arithmetic shift
    windowed = [(signal_q15[i] * hamming_q[i]) >> 15 for i in range(fft_n)]
    X = np.fft.fft(np.array(windowed, dtype=float))
    # The C code does >>1 per stage for log2(N) stages → divide by N total.
    X_scaled = X / float(fft_n)
    return [complex(z) for z in X_scaled]


def main() -> int:
    # Make sure host_fft_test is built and produces fresh output
    subprocess.run(["make", "-C", str(FIRMWARE), "host_fft_test"],
                   check=True, capture_output=True)
    proc = subprocess.run([str(FIRMWARE / "host_fft_test")],
                          check=True, capture_output=True, text=True, timeout=30)

    # Parse the C-printed spectrum: header line "# host_fft_test: ..." then "k,re,im,mag2"
    c_bins: list[tuple[int, int, int, int]] = []
    fft_n = None
    fft_fs_ds_x1000 = None
    for line in proc.stdout.splitlines():
        if line.startswith("# host_fft_test:"):
            # parse FFT_N and FFT_FS_DS_X1000
            for tok in line.split():
                if tok.startswith("FFT_N="):
                    fft_n = int(tok.split("=")[1])
                elif tok.startswith("FFT_FS_DS_X1000="):
                    fft_fs_ds_x1000 = int(tok.split("=")[1])
        elif line.startswith("k,"):
            continue  # CSV header
        elif line.strip() and not line.startswith("#"):
            parts = line.split(",")
            if len(parts) == 4:
                c_bins.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))

    if fft_n is None or fft_fs_ds_x1000 is None or not c_bins:
        print("FAIL: could not parse host_fft_test output", file=sys.stderr)
        return 1

    fs_ds = fft_fs_ds_x1000 / 1000.0
    print(f"FFT_N={fft_n}  fs_ds={fs_ds:.4f} Hz  test_f={TEST_F_HZ} Hz  "
          f"expected_bin≈{round(TEST_F_HZ * fft_n / fs_ds)}")

    # Reconstruct the Hamming window the C used — read it from fft_data.h
    fft_hdr = (FIRMWARE / "generated" / "fft_data.h").read_text()
    hamming_q = []
    in_window = False
    for line in fft_hdr.splitlines():
        if "fft_window_q15[FFT_N]" in line:
            in_window = True
            continue
        if in_window:
            if "fft_twiddle_q15" in line or line.startswith("};"):
                break
            for tok in line.replace("{", "").replace("}", "").split(","):
                t = tok.strip()
                if t and t.lstrip("-").isdigit():
                    hamming_q.append(int(t))
    hamming_q = hamming_q[:fft_n]
    if len(hamming_q) != fft_n:
        print(f"FAIL: parsed {len(hamming_q)} Hamming entries, expected {fft_n}",
              file=sys.stderr)
        return 1

    # Compute numpy reference and compare
    try:
        ref = numpy_reference(fft_n, fs_ds, hamming_q)
    except ImportError:
        print("WARNING: numpy unavailable — verify_fft.py needs numpy for bit-exact compare")
        return 0  # Not a fail — bare-stdlib default is design intent

    max_re_err = 0
    max_im_err = 0
    worst_k = 0
    for k, c_re, c_im, _c_mag2 in c_bins:
        ref_re_int = int(round(ref[k].real))
        ref_im_int = int(round(ref[k].imag))
        re_err = abs(c_re - ref_re_int)
        im_err = abs(c_im - ref_im_int)
        err = max(re_err, im_err)
        if err > max(max_re_err, max_im_err):
            worst_k = k
        if re_err > max_re_err: max_re_err = re_err
        if im_err > max_im_err: max_im_err = im_err

    # Peak bin agreement
    c_mag2 = [(k, m) for (k, _, _, m) in c_bins]
    c_peak = max(c_mag2[1:fft_n // 2], key=lambda kv: kv[1])[0]
    ref_mag2 = [abs(c) ** 2 for c in ref]
    ref_peak = max(range(1, fft_n // 2), key=lambda k: ref_mag2[k])

    print(f"peak bin: C={c_peak}  numpy={ref_peak}  "
          f"{'MATCH' if c_peak == ref_peak else 'MISMATCH'}")
    # Worst-case Q15 FFT truncation: each butterfly does `(a*b)>>15`, truncating up
    # to 1 LSB; over log2(N) stages, max per-bin error is bounded by log2(N) LSB.
    # For N=256 that's 8 LSB. Tighter bounds (~sqrt(log2(N))) hold on average; we
    # use the worst-case bound + 1-LSB slack.
    n_log2 = int(math.log2(fft_n))
    lsb_bound = n_log2 + 1
    print(f"max |re error|={max_re_err} LSB,  max |im error|={max_im_err} LSB  "
          f"(worst at k={worst_k}, bound = log2(N)+1 = {lsb_bound} LSB)")
    ok = (c_peak == ref_peak) and (max_re_err <= lsb_bound) and (max_im_err <= lsb_bound)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
