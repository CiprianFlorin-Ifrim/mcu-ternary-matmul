#include "dense_c.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Approach 1 -- dense INT8 matrix-vector multiply, plain C.
//
// Straightforward row-major traversal. The inner loop accumulates into a
// local INT32 register which the compiler keeps in a GP register.
// No unrolling or vectorisation hints -- the compiler decides.
// ---------------------------------------------------------------------------

void dense_matmul_c(const int8_t *W, const int8_t *x, int32_t *y, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        int32_t acc         = 0;
        const int8_t *w_row = W + i * cols;

        for (int j = 0; j < cols; j++) {
            acc += (int32_t)w_row[j] * (int32_t)x[j];
        }

        y[i] = acc;
    }
}


// ---------------------------------------------------------------------------
// MatmulFn wrapper -- matches the function pointer signature expected by
// time_matmul() so this approach can be timed identically to the others.
// ---------------------------------------------------------------------------

void dense_matmul_c_fn(const void *W, const int8_t *x, int32_t *y,
                       int rows, int cols, void *user_data)
{
    (void)user_data;
    dense_matmul_c((const int8_t *)W, x, y, rows, cols);
}
