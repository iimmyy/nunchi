// imi's own table formats and kernels.
//
// 4-bit tables (Seoul's Q4_0 and Q4_1 layer tables) in our 8-row layout: for each group of 8 rows and each block of
// 32 columns, 128 bytes of codes in 4 chunks of 32 bytes; in chunk t, bytes 4r..4r+3 are row r's columns 4t..4t+3
// (low 4 bits) and 16+4t..16+4t+3 (high 4 bits). Then the 8 rows' scales as 16-bit floats. The input is cut to 8-bit per 32
// columns once per table read, so  row . x = sum_blocks d * dx * (sum code * xq) + correction, where the correction
// depends only on the input (Q4_0: -8 * d * dx * sum xq) or on the row's offset m (Q4_1: m * dx * sum xq).
// Each multiply lands in its row's own lanes (no folding of rows), and scaling is one multiply-add for 8 rows.
//
// The word table stays in the file's 6-bit layout (Q6_K); its kernel reads it as stored.
#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

struct ggml_tensor;

float    imi_h2f(uint16_t h);
uint16_t imi_f2h(float f);

// memory for big read-only blocks: 2 MB aligned, huge pages where the OS allows
void * imi_big_alloc(size_t bytes);
void   imi_big_free(void * p);

struct imi_q4 {
    int rows = 0, cols = 0, nb = 0;
    bool has_m = false;           // Q4_1: weight = d * code + m; Q4_0: weight = d * (code - 8)
    uint8_t  * codes = nullptr;   // [rows/8][nb][128]
    uint16_t * d = nullptr;       // [rows/8][nb][8]
    uint16_t * m = nullptr;       // [rows/8][nb][8], Q4_1 only
    void * mem = nullptr;
    size_t bytes = 0;
    // rows, in order, taken from (table, row) pairs: every source must be 4-bit with the same width and type
    bool build(const std::vector<std::pair<const ggml_tensor *, int>> & order);
    void release();
};

// an input cut to 8-bit per 32 values
struct imi_q8 {
    int n = 0;
    std::vector<int8_t> q;
    std::vector<float> d, s;      // per block: scale, scale * sum of codes
    void from(const float * x, int n);
};

// many inputs cut to 8-bit per 32 values, block-major ([block][input][32]) so one block of all inputs sits together
struct imi_q8b {
    int n = 0, cols = 0, nb = 0;
    std::vector<int8_t> q;
    std::vector<float> d, s;      // [block][input]
    void shape(int n_in, int n_cols);
    void set(int i, const float * x);   // input i (different threads may set different inputs)
};
// one group of 8 rows times all inputs of X -> out[input][8]  (X.n <= 64)
void imi_q4_group_batch(const imi_q4 & W, const imi_q8b & X, int g, float * out);

// one group of 8 rows times the input -> out[8]
void imi_q4_group(const imi_q4 & W, const imi_q8 & X, int g, float * out);
// two inputs, one read of the table (two jobs riding one step)
void imi_q4_group2(const imi_q4 & W, const imi_q8 & X, const imi_q8 & X2, int g, float * out, float * out2);

// 8-bit tables (the voice generator's Q8_0) for many inputs at once (the 12 codebook positions of a pass).
// Our layout: groups of 8 rows; inside a group the 8 rows' weights for column k sit together (k-major), and each block
// of 32 columns has the 8 rows' scales. One step of the kernel turns 8 weights into floats (times their scales) and
// multiplies them into all inputs' accumulators, so inputs stay exact floats (no 8-bit cut of the input) and the
// 8 rows' sums come out in one register per input: no horizontal sums.
struct imi_q8g {
    int rows = 0, rows_pad = 0, cols = 0, nb = 0;
    int8_t   * codes = nullptr;   // [rows_pad/8][cols][8]
    uint16_t * d = nullptr;       // [rows_pad/8][nb][8]
    void * mem = nullptr;
    size_t bytes = 0;
    // rows of Q8_0 tables in order; rows past a multiple of 8 are padded with zeros
    bool build(const std::vector<std::pair<const ggml_tensor *, int>> & order, int plane = 0);
    void release();
};
// group g times C inputs; xt = inputs transposed, [cols][C]; out = [C][8]
void imi_q8g_group(const imi_q8g & W, const float * xt, int C, int g, float * out);
void imi_q8_0_dequant_row(const void * row, float * y, int cols);

// the word table (Q6_K, rows of 256-value blocks)
struct imi_q8k {
    int n = 0;
    std::vector<int8_t> q;
    std::vector<float> d;
    std::vector<int16_t> bsums;   // per 16 values
    void from(const float * x, int n);
};
float imi_q6k_row(const void * row, const imi_q8k & X, int nblocks);
void  imi_q6k_dequant_row(const void * row, float * y, int nblocks);
