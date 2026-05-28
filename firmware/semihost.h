/* Minimal ARM semihosting (Cortex-M / Thumb, BKPT 0xAB).
 * Used for console output and exit under QEMU (-semihosting-config). No libc needed. */
#ifndef SEMIHOST_H
#define SEMIHOST_H

#include <stdint.h>

static inline int __semihost(int op, const void *arg) {
    register int r0 __asm__("r0") = op;
    register const void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt #0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

/* SYS_WRITE0 (0x04): print a NUL-terminated string. */
static inline void sh_write0(const char *s) { (void)__semihost(0x04, s); }

/* SYS_EXIT (0x18): clean application exit (ADP_Stopped_ApplicationExit = 0x20026). */
static inline void sh_exit(void) {
    register int r0 __asm__("r0") = 0x18;
    register void *r1 __asm__("r1") = (void *)0x20026;
    __asm__ volatile("bkpt #0xAB" : : "r"(r0), "r"(r1) : "memory");
    for (;;) { }
}

/* Print a uint32_t as decimal via SYS_WRITE0. No libc itoa, no printf — build
 * a NUL-terminated string in a small stack buffer right-to-left, hand to
 * sh_write0. Extracted from main.c/main_fft.c/main_rr.c (was triplicated). */
static inline void sh_print_uint(uint32_t v) {
    char buf[12];
    int i = 11;
    buf[i--] = '\0';
    if (v == 0) {
        buf[i--] = '0';
    } else {
        while (v) { buf[i--] = (char)('0' + (v % 10u)); v /= 10u; }
    }
    sh_write0(&buf[i + 1]);
}

#endif /* SEMIHOST_H */
