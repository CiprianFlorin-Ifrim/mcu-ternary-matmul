#pragma once

#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// Matrix configuration -- every combination of rows x cols we benchmark.
// Covers the full range of transformer operation shapes:
//   - Square:            attention score matrices, SSM state matrices (Mamba)
//   - Rectangular tall: projection layers where T > D
//   - Rectangular wide: FFN up projections where D_out > D_in
//   - Mamba SSM shapes: very small D x N where N is state dim (16 or 32)
//   - Realistic layers: exact shapes from our trained 64-dim ternary model
// ---------------------------------------------------------------------------

typedef struct {
    int         rows;
    int         cols;
    const char *label;    // human-readable name for CSV output
} MatrixConfig;

// Full matrix configuration table -- defined in matrix_utils.c
extern const MatrixConfig MATRIX_CONFIGS[];
extern const int          N_MATRIX_CONFIGS;

// Zero fractions to test at each matrix size
extern const float ZERO_FRACS[];
extern const int   N_ZERO_FRACS;

// Number of timing runs to average over per combination
#define N_TIMING_RUNS   1000

// ---------------------------------------------------------------------------
// Bitmask matrix -- stores ternary weight matrix as two bitmask arrays.
// pos_mask[i] has bit j set if W[row][col] == +1
// neg_mask[i] has bit j set if W[row][col] == -1
// Everything not set in either mask is zero by definition.
// Word index = (row * cols + col) / 32
// Bit  index = (row * cols + col) % 32
// ---------------------------------------------------------------------------

typedef struct {
    int        rows;
    int        cols;
    int        n_words;       // ceil(rows * cols / 32)
    uint32_t  *pos_mask;      // +1 positions
    uint32_t  *neg_mask;      // -1 positions
} BitmaskMatrix;

// ---------------------------------------------------------------------------
// Timing result -- holds mean, min, max over N_TIMING_RUNS
// ---------------------------------------------------------------------------

typedef struct {
    float mean_us;
    float min_us;
    float max_us;
} TimingResult;

// ---------------------------------------------------------------------------
// Function declarations
// ---------------------------------------------------------------------------

// Generate a random INT8 weight matrix with a controlled zero fraction.
// Allocates and returns a rows x cols INT8 array (caller must free).
// For ternary matrices, non-zero values are exactly -1 or +1.
int8_t *matrix_generate(int rows, int cols, float zero_frac);

// Convert an INT8 ternary weight matrix to bitmask representation.
// Allocates internal mask arrays -- caller must call bitmask_free().
BitmaskMatrix bitmask_from_dense(const int8_t *W, int rows, int cols);

// Free bitmask matrix internal allocations.
void bitmask_free(BitmaskMatrix *bm);

// Generate a random INT8 activation vector of length n.
// Allocates and returns the vector (caller must free).
int8_t *vector_generate(int n);

// Time a matrix-vector multiply function over N_TIMING_RUNS runs.
// fn_ptr is called as: fn_ptr(W, x, y, rows, cols, user_data)
typedef void (*MatmulFn)(const void *W, const int8_t *x, int32_t *y,
                         int rows, int cols, void *user_data);

TimingResult time_matmul(MatmulFn fn, const void *W, const int8_t *x,
                         int32_t *y, int rows, int cols, void *user_data);

// Compute the actual zero fraction of a weight matrix (for verification).
float matrix_zero_frac(const int8_t *W, int rows, int cols);
