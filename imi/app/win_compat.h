// nunchi on Windows: <windows.h>, plus the part of Windows' power API that old mingw-w64 headers lack.
// Windows 11 slows a program down to save power (EcoQoS) when it decides the program is in the background, for example
// when its window is minimised behind the screen page. Programs opt out with SetProcessInformation (the whole program,
// app/main.cpp) and SetThreadInformation (one thread, ggml's own threads). Both are in kernel32 since Windows 8, and
// the import library of every mingw-w64 has them, but the headers of mingw-w64 before version 10 (Ubuntu 22.04 ships
// version 8) stop their lists of settings before the power one. This adds that part as Microsoft documents it, only
// where the headers lack it; with newer headers it adds nothing. ggml's ggml-cpu.c gets this file ahead of its own text
// (CMakeLists), so it compiles here too.
#pragma once
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00   // Windows 10, the oldest Windows nunchi runs on
#endif
#include <windows.h>

#if defined(__MINGW64_VERSION_MAJOR) && !defined(THREAD_POWER_THROTTLING_CURRENT_VERSION)
#ifdef __cplusplus
extern "C" {
#endif
// the old headers' THREAD_INFORMATION_CLASS stops after ThreadAbsoluteCpuPriority (1); Microsoft's value for this is 3
#define ThreadPowerThrottling ((THREAD_INFORMATION_CLASS) 3)
#define THREAD_POWER_THROTTLING_CURRENT_VERSION 1
#define THREAD_POWER_THROTTLING_EXECUTION_SPEED 0x1
#define THREAD_POWER_THROTTLING_VALID_FLAGS     THREAD_POWER_THROTTLING_EXECUTION_SPEED
typedef struct _THREAD_POWER_THROTTLING_STATE {
    ULONG Version;
    ULONG ControlMask;
    ULONG StateMask;
} THREAD_POWER_THROTTLING_STATE;
#if _WIN32_WINNT < 0x0602   // the old headers declare it only from Windows 8 on
WINBASEAPI WINBOOL WINAPI SetThreadInformation(HANDLE hThread, THREAD_INFORMATION_CLASS ThreadInformationClass,
                                               LPVOID ThreadInformation, DWORD ThreadInformationSize);
#endif
#ifdef __cplusplus
}
#endif
#endif

#if defined(__MINGW64_VERSION_MAJOR) && !defined(PROCESS_POWER_THROTTLING_CURRENT_VERSION)
#ifdef __cplusplus
extern "C" {
#endif
// the old headers' PROCESS_INFORMATION_CLASS stops after ProcessMemoryPriority (0); Microsoft's value for this is 4
#define ProcessPowerThrottling ((PROCESS_INFORMATION_CLASS) 4)
#define PROCESS_POWER_THROTTLING_CURRENT_VERSION        1
#define PROCESS_POWER_THROTTLING_EXECUTION_SPEED        0x1
#define PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION 0x4
typedef struct _PROCESS_POWER_THROTTLING_STATE {
    ULONG Version;
    ULONG ControlMask;
    ULONG StateMask;
} PROCESS_POWER_THROTTLING_STATE;
#if _WIN32_WINNT < 0x0602   // the old headers declare it only from Windows 8 on
WINBASEAPI WINBOOL WINAPI SetProcessInformation(HANDLE hProcess, PROCESS_INFORMATION_CLASS ProcessInformationClass,
                                                LPVOID ProcessInformation, DWORD ProcessInformationSize);
#endif
#ifdef __cplusplus
}
#endif
#endif
#endif
