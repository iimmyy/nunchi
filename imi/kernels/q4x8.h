// imi kernel: Seoul's 4-bit (Q4_0) tables times one input, in our own 8-row layout (see q4x8.cpp).
#pragma once
#include "ggml.h"
#include <atomic>
#include <cstdint>
#include <vector>

struct imi_q4x8 {
    int rows = 0, cols = 0, nb = 0;
    std::vector<uint8_t> codes;   // [rows/8][nb][4 x 32 bytes]: 2 rows per byte (low / high 4 bits), codes 0..15
    std::vector<float>   scales;  // [rows/8][nb][8]
};

bool imi_q4x8_supported(const ggml_tensor * w);          // a Q4_0 table this CPU can run in our layout
imi_q4x8 * imi_q4x8_from(const ggml_tensor * w);          // converts a loaded Q4_0 table
// one use of a table in a step: the table plus a work counter (threads take 32-row chunks as they free up)
struct imi_q4x8_call { const imi_q4x8 * w; std::atomic<int> next{0}; };
ggml_tensor * imi_q4x8_mul(ggml_context * ctx, imi_q4x8_call * call, ggml_tensor * x); // [cols] f32 -> [rows] f32
