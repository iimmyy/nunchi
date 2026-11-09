// nunchi: built for AVX2 processors (IMI_ARCH avx2) and started on an older one, it says so plainly instead of
// crashing. Every other file may use AVX2 anywhere, also in code that runs before main(), so this file is built
// without AVX (CMakeLists) and runs first, as the earliest constructor.
#if defined(NUNCHI_NEEDS_AVX2) && (defined(__x86_64__) || defined(_M_X64))
#include <cstdio>
#include <cstdlib>

__attribute__((constructor(101))) static void nunchi_cpu_check() {
    __builtin_cpu_init();   // the CPU checks are not set up yet this early
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return;
    printf("This computer's processor is too old for Nunchi. It needs AVX2: an Intel processor from 2013 or newer, or\n"
           "an AMD one from 2015 or newer. Press Enter to close.\n");
    fflush(stdout);
    getchar();
    exit(1);
}
#endif
