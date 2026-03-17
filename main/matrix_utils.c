#include "matrix_utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_timer.h"
#include "esp_random.h"

// ---------------------------------------------------------------------------
// Matrix size configuration table.
// Organised into groups matching transformer operation shapes.
// ---------------------------------------------------------------------------

const MatrixConfig MATRIX_CONFIGS[] = {

    // -- Mamba SSM state matrices (D x N, N = state dim 16 or 32) ----------
    { 16,  16,  "mamba_state_16x16"   },
    { 32,  32,  "mamba_state_32x32"   },
    { 64,  16,  "mamba_ssm_64x16"     },
    { 64,  32,  "mamba_ssm_64x32"     },
    { 128, 16,  "mamba_ssm_128x16"    },
    { 128, 32,  "mamba_ssm_128x32"    },
    { 256, 16,  "mamba_ssm_256x16"    },
    { 256, 32,  "mamba_ssm_256x32"    },

    // -- Square matrices (attention scores, symmetric projections) ----------
    { 32,  32,  "square_32x32"        },
    { 64,  64,  "square_64x64"        },
    { 128, 128, "square_128x128"      },
    { 256, 256, "square_256x256"      },

    // -- Rectangular tall (T > D, projection layers with long sequences) ----
    { 128, 64,  "tall_128x64"         },
    { 256, 64,  "tall_256x64"         },
    { 256, 128, "tall_256x128"        },
    { 512, 64,  "tall_512x64"         },
    { 512, 128, "tall_512x128"        },

    // -- Rectangular wide (FFN up projections, D_out > D_in) ---------------
    { 64,  128, "wide_64x128"         },
    { 64,  256, "wide_64x256"         },
    { 128, 256, "wide_128x256"        },
    { 128, 512, "wide_128x512"        },
    { 256, 512, "wide_256x512"        },

    // -- Realistic layer shapes from trained 64-dim ternary model -----------
    { 64,  64,  "model_attn_proj"     },
    { 64,  128, "model_ffn_up"        },
    { 128, 64,  "model_ffn_down"      },
    { 64,  15,  "model_output_proj"   },

    // -- SwiGLU gate shapes -------------------------------------------------
    { 64,  128, "swiglu_gate"         },
    { 64,  128, "swiglu_up"           },

};

const int N_MATRIX_CONFIGS = sizeof(MATRIX_CONFIGS) / sizeof(MATRIX_CONFIGS[0]);

const float ZERO_FRACS[] = { 0.0f, 0.30f, 0.50f, 0.70f, 0.90f, 0.95f };
const int   N_ZERO_FRACS = sizeof(ZERO_FRACS) / sizeof(ZERO_FRACS[0]);


// ---------------------------------------------------------------------------
// Matrix generation -- ternary weights with controlled zero fraction.
// Allocated with 16-byte alignment so PIE VLD instructions hit aligned
// addresses and dense_pie_fn can call the assembly directly without copying.
// ---------------------------------------------------------------------------

static inline size_t align16_size(int n)
{
    // aligned_alloc requires size to be a multiple of the alignment
    return (size_t)(((n + 15) / 16) * 16);
}

int8_t *matrix_generate(int rows, int cols, float zero_frac)
{
    int    n    = rows * cols;
    int8_t *W   = (int8_t *)aligned_alloc(16, align16_size(n));
    if (!W) return NULL;

    for (int i = 0; i < n; i++) {
        float r = (float)(esp_random() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        if (r < zero_frac) {
            W[i] = 0;
        } else {
            W[i] = (esp_random() & 1) ? 1 : -1;
        }
    }

    return W;
}


// ---------------------------------------------------------------------------
// Bitmask conversion
// ---------------------------------------------------------------------------

BitmaskMatrix bitmask_from_dense(const int8_t *W, int rows, int cols)
{
    BitmaskMatrix bm;
    bm.rows    = rows;
    bm.cols    = cols;
    bm.n_words = (rows * cols + 31) / 32;
    bm.pos_mask = (uint32_t *)calloc(bm.n_words, sizeof(uint32_t));
    bm.neg_mask = (uint32_t *)calloc(bm.n_words, sizeof(uint32_t));

    for (int i = 0; i < rows * cols; i++) {
        int word = i / 32;
        int bit  = i % 32;
        if (W[i] > 0) {
            bm.pos_mask[word] |= (1u << bit);
        } else if (W[i] < 0) {
            bm.neg_mask[word] |= (1u << bit);
        }
    }

    return bm;
}


void bitmask_free(BitmaskMatrix *bm)
{
    free(bm->pos_mask);
    free(bm->neg_mask);
    bm->pos_mask = NULL;
    bm->neg_mask = NULL;
}


// ---------------------------------------------------------------------------
// Activation vector generation -- 16-byte aligned, random INT8 values.
// ---------------------------------------------------------------------------

int8_t *vector_generate(int n)
{
    int8_t *x = (int8_t *)aligned_alloc(16, align16_size(n));
    if (!x) return NULL;

    for (int i = 0; i < n; i++) {
        x[i] = (int8_t)(esp_random() & 0xFF);
    }

    return x;
}


// ---------------------------------------------------------------------------
// Timing harness
// ---------------------------------------------------------------------------

TimingResult time_matmul(MatmulFn fn, const void *W, const int8_t *x,
                         int32_t *y, int rows, int cols, void *user_data)
{
    TimingResult result;
    float        total_us = 0.0f;
    float        min_us   = 1e12f;
    float        max_us   = 0.0f;

    for (int run = 0; run < N_TIMING_RUNS; run++) {
        int64_t t0      = esp_timer_get_time();
        fn(W, x, y, rows, cols, user_data);
        int64_t elapsed = esp_timer_get_time() - t0;

        float us  = (float)elapsed;
        total_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }

    result.mean_us = total_us / N_TIMING_RUNS;
    result.min_us  = min_us;
    result.max_us  = max_us;

    return result;
}


// ---------------------------------------------------------------------------
// Utility -- actual zero fraction of a weight matrix.
// ---------------------------------------------------------------------------

float matrix_zero_frac(const int8_t *W, int rows, int cols)
{
    int n    = rows * cols;
    int zero = 0;
    for (int i = 0; i < n; i++) {
        if (W[i] == 0) zero++;
    }
    return (float)zero / (float)n;
}
