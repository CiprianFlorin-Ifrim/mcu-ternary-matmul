#pragma once

#include <stdint.h>
#include "matrix_utils.h"

// ---------------------------------------------------------------------------
// Approach 4 -- block-sparse PIE hybrid matrix-vector multiply.
//
// Combines the zero-skipping benefit of sparse computation with the
// vectorised throughput of PIE SIMD. The weight matrix is divided into
// 16-element blocks (matching the PIE Q register width of 128 bits = 16
// INT8 values). Before processing each block, a precomputed block-presence
// flag is checked -- if all 16 weights in the block are zero, the block is
// skipped entirely with no memory access and no compute.
//
// For non-zero blocks, 16 weights and 16 activations are loaded in single
// PIE load instructions and processed with the QACC MAC accumulator.
//
// This approach is theoretically optimal when:
//   - Sparsity is structured enough that whole 16-element blocks are zero
//   - Zero fraction is high enough that block-skip overhead is worth paying
//
// A block-presence array (one byte per 16-weight block) is precomputed from
// the INT8 weight matrix and stored alongside it in the BlockSparseMatrix
// struct. Precomputation cost is paid once, not at inference time.
// ---------------------------------------------------------------------------

// Block size matches PIE Q register width: 16 x INT8 = 128 bits
#define BLOCK_SIZE  16

// Block-sparse matrix representation
typedef struct {
    int            rows;
    int            cols;
    int            cols_padded;      // cols rounded up to multiple of BLOCK_SIZE
    int8_t        *W_padded;         // weight matrix, cols padded to BLOCK_SIZE multiple
    uint8_t       *block_nonzero;    // 1 if block has any non-zero weights, 0 if all zero
    int            n_blocks_per_row; // cols_padded / BLOCK_SIZE
} BlockSparseMatrix;

// Build a BlockSparseMatrix from a dense INT8 weight matrix.
// Allocates internal arrays -- caller must call block_sparse_free().
BlockSparseMatrix block_sparse_from_dense(const int8_t *W, int rows, int cols);

// Free internal allocations.
void block_sparse_free(BlockSparseMatrix *bs);

// MatmulFn-compatible wrapper
void sparse_matmul_pie_fn(const void *bs, const int8_t *x, int32_t *y,
                          int rows, int cols, void *user_data);

// Direct call version -- implemented partly in sparse_pie.S
void sparse_matmul_pie(const BlockSparseMatrix *bs, const int8_t *x, int32_t *y);
