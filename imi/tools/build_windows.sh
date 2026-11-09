#!/bin/bash
# nunchi.exe for Windows, built on Linux (Ubuntu 22.04 or newer) with mingw-w64:
#   sudo apt-get install g++-mingw-w64-x86-64-posix        (once)
#   imi/tools/build_windows.sh GGML_DIR [BUILD_DIR]
# GGML_DIR: the ggml/ folder of our llama.cpp fork. The program runs on Windows 10 and 11 with an AVX2 processor
# (Intel from 2013, AMD from 2015, or newer) and is linked statically, so it needs no DLLs next to it.
# Then imi/tools/package.sh BUILD_DIR/nunchi.exe ... makes the folder for the room's computer.
set -e
ggml="$1"; out="${2:-build-win}"
if [ -z "$ggml" ]; then echo "usage: build_windows.sh GGML_DIR [BUILD_DIR]"; exit 1; fi
here="$(cd "$(dirname "$0")/.." && pwd)"
cmake -S "$here" -B "$out" -DCMAKE_TOOLCHAIN_FILE="$here/cmake/mingw-w64.cmake" -DGGML_DIR="$ggml" \
      -DIMI_NATIVE=OFF -DIMI_ARCH=avx2 -DCMAKE_BUILD_TYPE=Release
cmake --build "$out" --target nunchi -j"$(nproc)"
ls -la "$out/nunchi.exe"
