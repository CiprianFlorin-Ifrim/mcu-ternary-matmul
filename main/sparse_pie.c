#include "sparse_pie.h"
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Build block-sparse representation from dense INT8 weight matrix.
// ---------------------------------------------------------------------------

BlockSparseMatrix block_sparse_from_dense(const int8_t *W, int rows, int cols)
{
    BlockSparseMatrix bs;
    bs.rows            = rows;
    bs.cols            = cols;
    bs.cols_padded     = (cols + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
    bs.n_blocks_per_row = bs.cols_padded / BLOCK_SIZE;

    int total_padded   = rows * bs.cols_padded;
    int total_blocks   = rows * bs.n_blocks_per_row;

    // Allocate padded weight matrix -- zero-fill so padding is neutral
    bs.W_padded      = (int8_t  *)calloc(total_padded, sizeof(int8_t));
    bs.block_nonzero = (uint8_t *)calloc(total_blocks, sizeof(uint8_t));

    // Copy original weights into padded layout
    for (int i = 0; i < rows; i++) {
        memcpy(bs.W_padded + i * bs.cols_padded,
               W + i * cols,
               cols);
    }

    // Build block-presence flags -- one flag per 16-element block
    for (int i = 0; i < rows; i++) {
        for (int b = 0; b < bs.n_blocks_per_row; b++) {
            int           block_start = i * bs.cols_padded + b * BLOCK_SIZE;
            const int8_t *block       = bs.W_padded + block_start;
            uint8_t       nonzero     = 0;

            for (int k = 0; k < BLOCK_SIZE; k++) {
                if (block[k] != 0) {
                    nonzero = 1;
                    break;
                }
            }

            bs.block_nonzero[i * bs.n_blocks_per_row + b] = nonzero;
        }
    }

    return bs;
}


void block_sparse_free(BlockSparseMatrix *bs)
{
    free(bs->W_padded);
    free(bs->block_nonzero);
    bs->W_padded      = NULL;
    bs->block_nonzero = NULL;
}


// ---------------------------------------------------------------------------
// Approach 4 -- block-sparse PIE hybrid.
//
// For each row, check each 16-element block's nonzero flag.
// Skip the block entirely if all-zero.
// For non-zero blocks, call the PIE inner loop (in sparse_pie.S).
// Accumulate results into INT32 output.
//
// The C-level block dispatch loop allows clean zero-block skipping without
// the complexity of encoding it in assembly. The PIE routine handles only
// the non-zero block multiply-accumulate.
// ---------------------------------------------------------------------------

// Forward declaration of PIE inner loop from sparse_pie.S
// Computes: acc += dot(w_block[0..15], x_block[0..15])
// where w_block and x_block are 16-element INT8 arrays.
extern int32_t pie_block_dot(const int8_t *w_block, const int8_t *x_block);


void sparse_matmul_pie(const BlockSparseMatrix *bs, const int8_t *x, int32_t *y)
{
    int rows            = bs->rows;
    int n_blocks        = bs->n_blocks_per_row;
    int cols_padded     = bs->cols_padded;

    for (int i = 0; i < rows; i++) {
        int32_t       acc         = 0;
        const int8_t *w_row       = bs->W_padded      + i * cols_padded;
        const uint8_t *flags_row  = bs->block_nonzero + i * n_blocks;

        for (int b = 0; b < n_blocks; b++) {
            // -- Skip all-zero blocks with no memory access ---------------
            if (!flags_row[b]) continue;

            const int8_t *w_block = w_row + b * BLOCK_SIZE;
            const int8_t *x_block = x    + b * BLOCK_SIZE;

            // -- PIE-accelerated dot product of one 16-element block ------
            acc += pie_block_dot(w_block, x_block);
        }

        y[i] = acc;
    }
}


// ---------------------------------------------------------------------------
// MatmulFn wrapper
// ---------------------------------------------------------------------------

void sparse_matmul_pie_fn(const void *bs, const int8_t *x, int32_t *y,
                          int rows, int cols, void *user_data)
{
    (void)rows;
    (void)cols;
    (void)user_data;
    sparse_matmul_pie((const BlockSparseMatrix *)bs, x, y);
}
