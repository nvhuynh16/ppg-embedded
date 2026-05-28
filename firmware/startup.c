/* Minimal startup + vector table for LM3S6965 (Cortex-M3) under QEMU.
 * No CMSIS, no crt0: copy .data, zero .bss, call main(). */
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);

void Reset_Handler(void);
void Default_Handler(void);

void Reset_Handler(void) {
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;     /* init .data from flash */
    for (dst = &_sbss; dst < &_ebss; ) *dst++ = 0u;  /* zero .bss */
    main();
    for (;;) { }                               /* main() exits via semihosting */
}

void Default_Handler(void) { for (;;) { } }

/* Cortex-M3 vector table: initial SP, reset, then system exceptions. */
__attribute__((section(".isr_vector"), used))
void (*const g_pfnVectors[])(void) = {
    (void (*)(void))(&_estack),  /* 0  Initial Stack Pointer */
    Reset_Handler,               /* 1  Reset */
    Default_Handler,             /* 2  NMI */
    Default_Handler,             /* 3  HardFault */
    Default_Handler,             /* 4  MemManage */
    Default_Handler,             /* 5  BusFault */
    Default_Handler,             /* 6  UsageFault */
    (void (*)(void))0,           /* 7  reserved */
    (void (*)(void))0,           /* 8  reserved */
    (void (*)(void))0,           /* 9  reserved */
    (void (*)(void))0,           /* 10 reserved */
    Default_Handler,             /* 11 SVCall */
    Default_Handler,             /* 12 DebugMonitor */
    (void (*)(void))0,           /* 13 reserved */
    Default_Handler,             /* 14 PendSV */
    Default_Handler              /* 15 SysTick */
};
