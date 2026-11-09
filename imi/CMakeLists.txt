# imi.cpp: an engine built for Seoul (LFM3-Omni) alone. Our own design (graphs, caches, frame loop, pipeline);
# the innermost math loops come from ggml for now and are replaced one at a time when ours is faster on the floor box.
cmake_minimum_required(VERSION 3.16)
project(imi CXX C)
set(CMAKE_CXX_STANDARD 17)
set(GGML_DIR "" CACHE PATH "ggml source (the ggml/ folder of our llama.cpp fork)")
if (NOT GGML_DIR)
  message(FATAL_ERROR "pass -DGGML_DIR=<llama.cpp fork>/ggml")
endif()
# CPU only, like the church's Dell: no GPU, no vendor BLAS
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
# same CPU kernels and thread scheduler as the llama.cpp build we race against
set(GGML_OPENMP ON CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
add_subdirectory(${GGML_DIR} ggml)
add_executable(imi-gen gen.cpp model.cpp)
target_link_libraries(imi-gen PRIVATE ggml ggml-cpu)
