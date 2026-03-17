#pragma once

#include <stdint.h>
#include "matrix_utils.h"

// ---------------------------------------------------------------------------
// Approach 2 -- dense INT8 matrix-vector multiply using ESP32-P4 PIE.
//
// Uses the Processor Instruction Extensions (PIE) custom SIMD extension
// introduced in the ESP32-P4. Eight 128-bit Q registers allow 16 INT8
// values to be loaded in a single cycle via esp.vld.128.ip. The MAC
// accumulator (QACC, 256-bit on P4) accumulates 16 INT8 multiply-add
// results per instruction via esp.vmulas.s8.qacc.
//
// The inner loop processes 16 elements per iteration, giving 16x throughput
// compared to scalar when memory bandwidth is the bottleneck.
//
// Reference: ESP32-P4 Technical Reference Manual, PIE chapter.
// Instruction prefix: esp. (vs ee. on ESP32-S3)
// ---------------------------------------------------------------------------

// MatmulFn-compatible wrapper
void dense_matmul_pie_fn(const void *W, const int8_t *x, int32_t *y,
                         int rows, int cols, void *user_data);

// Direct call version -- implemented in dense_pie.S
// W must be 16-byte aligned for optimal PIE performance.
// cols must be a multiple of 16 -- caller pads if needed.
void dense_matmul_pie(const int8_t *W, const int8_t *x, int32_t *y,
                      int rows, int cols);
