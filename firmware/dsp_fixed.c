#include "dsp_fixed.h"

int fir_q15(const int16_t *x, int n, const int16_t *h, int ntaps, int32_t *y) {
    int out = 0;
    for (int i = ntaps - 1; i < n; i++) {
        int64_t acc = 0;                 /* widen to avoid overflow across taps */
        for (int k = 0; k < ntaps; k++) {
            acc += (int64_t)h[k] * (int64_t)x[i - k];
        }
        y[out++] = (int32_t)(acc >> 15); /* back to Q15 scale */
    }
    return out;
}

int detect_peaks(const int32_t *y, int len, int refractory,
                 int *peak_indices, int max_peaks) {
    int32_t maxabs = 0;
    for (int i = 0; i < len; i++) {
        int32_t a = y[i] < 0 ? -y[i] : y[i];
        if (a > maxabs) maxabs = a;
    }
    int32_t thr = maxabs / 2;
    int count = 0;
    int last = -refractory;
    for (int i = 1; i < len - 1; i++) {
        if (y[i] > thr && y[i] >= y[i - 1] && y[i] > y[i + 1] &&
            (i - last) >= refractory) {
            if (peak_indices && count < max_peaks) {
                peak_indices[count] = i;
            }
            count++;
            last = i;
        }
    }
    return count;
}

/* In-place insertion sort. n is small (typical ≤ ~50 intervals/window), so O(n²) is fine. */
static void insertion_sort_int(int *arr, int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

uint32_t hr_x100_from_peaks(const int *peak_indices, int n_peaks, int fs) {
    if (n_peaks < 2) return 0;
    /* Intervals are sample-counts between consecutive peaks. Physiological gate:
     * 0.4 s → 150 bpm upper bound, 1.5 s → 40 bpm lower bound. Outliers (motion,
     * missed/spurious peaks) are rejected before taking the median. */
    static int intervals[256];
    int n_int = 0;
    int min_int = (fs * 4) / 10;   /* 0.4 s */
    int max_int = (fs * 15) / 10;  /* 1.5 s */
    for (int i = 1; i < n_peaks; i++) {
        int d = peak_indices[i] - peak_indices[i - 1];
        if (d >= min_int && d <= max_int && n_int < (int)(sizeof(intervals) / sizeof(intervals[0]))) {
            intervals[n_int++] = d;
        }
    }
    if (n_int == 0) return 0;
    insertion_sort_int(intervals, n_int);
    int med = intervals[n_int / 2];
    /* HR_x100 = 60 * 100 * fs / median_interval_samples */
    return (uint32_t)(((int64_t)6000 * fs) / med);
}
