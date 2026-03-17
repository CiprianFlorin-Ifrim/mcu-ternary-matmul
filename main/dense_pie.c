#include "dense_pie.h"
#include "dense_c.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Approach 2 -- dense PIE wrapper.
//
// Fast path: if cols is already a multiple of 16 AND both W and x are
// 16-byte aligned (guaranteed when allocated via matrix_utils.c), the
// assembly is called directly with zero allocation overhead.
//
// Slow path: cols not a multiple of 16 (e.g. cols=15) -- pad to the next
// multiple of 16 using aligned_alloc so the assembly still gets aligned
// pointers. Falls back to dense_c on allocation failure.
// ---------------------------------------------------------------------------

static inline int pad16(int n)
{
    return (n + 15) & ~15;
}

static inline int is_aligned16(const void *p)
{
    return ((uintptr_t)p & 15) == 0;
}

void dense_matmul_pie_fn(const void *W, const int8_t *x, int32_t *y,
                         int rows, int cols, void *user_data)
{
    (void)user_data;

    int cols_padded = pad16(cols);

    // Fast path -- no allocation needed
    if (cols_padded == cols &&
        is_aligned16(W) &&
        is_aligned16(x)) {
        dense_matmul_pie((const int8_t *)W, x, y, rows, cols);
        return;
    }

    // Slow path -- pad and align
    int    W_size = rows * cols_padded;
    int    x_size = cols_padded;

    int8_t *W_pad = (int8_t *)aligned_alloc(16, W_size);
    int8_t *x_pad = (int8_t *)aligned_alloc(16, x_size);

    if (!W_pad || !x_pad) {
        free(W_pad);
        free(x_pad);
        dense_matmul_c((const int8_t *)W, x, y, rows, cols);
        return;
    }

    memset(W_pad, 0, W_size);
    memset(x_pad, 0, x_size);

    for (int i = 0; i < rows; i++) {
        memcpy(W_pad + i * cols_padded,
               (const int8_t *)W + i * cols,
               cols);
    }
    memcpy(x_pad, x, cols);

    dense_matmul_pie(W_pad, x_pad, y, rows, cols_padded);

    free(W_pad);
    free(x_pad);
}
