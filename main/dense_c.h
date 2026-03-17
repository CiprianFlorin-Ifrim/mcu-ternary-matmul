#pragma once

#include <stdint.h>
#include "matrix_utils.h"

// ---------------------------------------------------------------------------
// Approach 1 -- dense INT8 matrix-vector multiply, plain C.
//
// No intrinsics, no SIMD, no special instructions. The compiler is free to
// optimise but no explicit vectorisation is requested. This is the baseline
// that any standard embedded ML framework would produce without hardware
// awareness.
//
// y[rows] = W[rows x cols] * x[cols]
// Accumulator is INT32 to avoid overflow across cols additions.
// ---------------------------------------------------------------------------

// MatmulFn-compatible wrapper (for use with time_matmul)
void dense_matmul_c_fn(const void *W, const int8_t *x, int32_t *y,
                       int rows, int cols, void *user_data);

// Direct call version
void dense_matmul_c(const int8_t *W, const int8_t *x, int32_t *y,
                    int rows, int cols);
