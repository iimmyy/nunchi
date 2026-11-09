# Building nunchi.exe for Windows on Linux (Ubuntu's mingw-w64 package):
#   cmake -S imi -B build-win -DCMAKE_TOOLCHAIN_FILE=imi/cmake/mingw-w64.cmake -DGGML_DIR=<fork>/ggml -DIMI_NATIVE=OFF \
#         -DIMI_ARCH=avx2 -DCMAKE_BUILD_TYPE=Release && cmake --build build-win --target nunchi
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc-posix)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++-posix)
set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
