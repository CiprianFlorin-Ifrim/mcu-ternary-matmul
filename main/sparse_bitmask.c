#include "sparse_bitmask.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Approach 3 -- bitmask sparse matrix-vector multiply.
//
// For each row, iterate over the 32-bit words of pos_mask and neg_mask.
// Within each word, use __builtin_ctz to find the lowest set bit,
// compute its global column index, gather x[col], accumulate, clear the bit.
// Repeat until the word is zero, then move to the next word.
//
// __builtin_ctz maps to the RISC-V CTZ instruction (part of Zbb extension)
// on recent GCC/LLVM toolchains targeting RISC-V, giving single-cycle
// bit position lookup.
// ---------------------------------------------------------------------------

void sparse_matmul_bitmask(const BitmaskMatrix *bm, const int8_t *x, int32_t *y)
{
    int rows     = bm->rows;
    int cols     = bm->cols;
    int words_per_row = (cols + 31) / 32;

    for (int i = 0; i < rows; i++) {
        int32_t      acc        = 0;
        int          word_base  = i * words_per_row;

        for (int w = 0; w < words_per_row; w++) {
            int col_base = w * 32;

            // -- Process +1 weights in this 32-bit word -------------------
            uint32_t pos_word = bm->pos_mask[word_base + w];
            while (pos_word) {
                int bit = __builtin_ctz(pos_word);    // index of lowest set bit
                int col = col_base + bit;
                if (col < cols) {
                    acc += (int32_t)x[col];           // weight = +1: add activation
                }
                pos_word &= pos_word - 1;             // clear lowest set bit
            }

            // -- Process -1 weights in this 32-bit word -------------------
            uint32_t neg_word = bm->neg_mask[word_base + w];
            while (neg_word) {
                int bit = __builtin_ctz(neg_word);
                int col = col_base + bit;
                if (col < cols) {
                    acc -= (int32_t)x[col];           // weight = -1: subtract activation
                }
                neg_word &= neg_word - 1;
            }
        }

        y[i] = acc;
    }
}


// ---------------------------------------------------------------------------
// MatmulFn wrapper -- user_data is unused; bm pointer is passed as W.
// ---------------------------------------------------------------------------

void sparse_matmul_bitmask_fn(const void *bm, const int8_t *x, int32_t *y,
                              int rows, int cols, void *user_data)
{
    (void)rows;
    (void)cols;
    (void)user_data;
    sparse_matmul_bitmask((const BitmaskMatrix *)bm, x, y);
}
