#pragma once

#include <stdint.h>
#include "matrix_utils.h"

// ---------------------------------------------------------------------------
// Approach 3 -- bitmask sparse matrix-vector multiply, scalar C.
//
// The ternary weight matrix is stored as two bitmask arrays (pos_mask and
// neg_mask) rather than an INT8 array. Each bit encodes whether the
// corresponding weight is +1 or -1. Zero weights have no bit set in either
// mask and are skipped entirely -- they cost zero compute.
//
// The inner loop uses __builtin_ctz (count trailing zeros) to find the
// position of the next set bit, gathers the corresponding activation,
// adds or subtracts it from the accumulator, then clears that bit and
// continues. This is a standard sparse bit-manipulation loop.
//
// No multiplications are performed -- only additions and subtractions.
// The saving comes entirely from skipping zero weights.
//
// At 70% zeros: processes ~30% of weights vs dense.
// At 90% zeros: processes ~10% of weights vs dense.
// The break-even vs dense PIE depends on __builtin_ctz overhead vs
// PIE throughput -- this benchmark measures exactly that crossover.
// ---------------------------------------------------------------------------

// MatmulFn-compatible wrapper
void sparse_matmul_bitmask_fn(const void *bm, const int8_t *x, int32_t *y,
                              int rows, int cols, void *user_data);

// Direct call version -- takes a BitmaskMatrix pointer
void sparse_matmul_bitmask(const BitmaskMatrix *bm, const int8_t *x, int32_t *y);
