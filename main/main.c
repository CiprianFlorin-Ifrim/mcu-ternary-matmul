#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "matrix_utils.h"
#include "dense_c.h"
#include "dense_pie.h"
#include "sparse_bitmask.h"
#include "sparse_pie.h"
#include "results.h"

static const char *TAG = "ternary_matmul";

// ---------------------------------------------------------------------------
// run_all_benchmarks -- iterates over every combination of:
//   matrix config x zero fraction x approach
// and prints one CSV row per combination.
//
// Memory for weight matrices, activation vectors, and output vectors is
// allocated and freed per matrix config to keep peak SRAM usage bounded.
// The bitmask and block-sparse structures are rebuilt for each zero fraction.
// ---------------------------------------------------------------------------

static void run_all_benchmarks(void)
{
    results_print_header();

    for (int cfg_idx = 0; cfg_idx < N_MATRIX_CONFIGS; cfg_idx++) {
        const MatrixConfig *cfg  = &MATRIX_CONFIGS[cfg_idx];
        int                 rows = cfg->rows;
        int                 cols = cfg->cols;

        ESP_LOGI(TAG, "Matrix %s (%d x %d)", cfg->label, rows, cols);

        // Allocate output vector and activation vector once per matrix config
        int32_t *y = (int32_t *)malloc(rows * sizeof(int32_t));
        int8_t  *x = vector_generate(cols);

        if (!y || !x) {
            ESP_LOGE(TAG, "Allocation failed for %s -- skipping", cfg->label);
            free(y);
            free(x);
            continue;
        }

        for (int zf_idx = 0; zf_idx < N_ZERO_FRACS; zf_idx++) {
            float zero_frac = ZERO_FRACS[zf_idx];

            // -- Generate weight matrix for this zero fraction ------------
            int8_t *W = matrix_generate(rows, cols, zero_frac);
            if (!W) {
                ESP_LOGE(TAG, "Weight alloc failed for %s zf=%.2f", cfg->label, zero_frac);
                continue;
            }

            float actual_zf = matrix_zero_frac(W, rows, cols);

            // -- Build derived representations ----------------------------
            BitmaskMatrix     bm = bitmask_from_dense(W, rows, cols);
            BlockSparseMatrix bs = block_sparse_from_dense(W, rows, cols);

            // -- Approach 1: dense plain C ---------------------------------
            {
                TimingResult t = time_matmul(dense_matmul_c_fn,
                                             W, x, y, rows, cols, NULL);
                results_print_row("dense_c", rows, cols, cfg->label,
                                  zero_frac, actual_zf, t);
            }

            // -- Approach 2: dense PIE ------------------------------------
            {
                TimingResult t = time_matmul(dense_matmul_pie_fn,
                                             W, x, y, rows, cols, NULL);
                results_print_row("dense_pie", rows, cols, cfg->label,
                                  zero_frac, actual_zf, t);
            }

            // -- Approach 3: bitmask sparse scalar C ----------------------
            {
                TimingResult t = time_matmul(sparse_matmul_bitmask_fn,
                                             &bm, x, y, rows, cols, NULL);
                results_print_row("sparse_bitmask", rows, cols, cfg->label,
                                  zero_frac, actual_zf, t);
            }

            // -- Approach 4: block-sparse PIE hybrid ----------------------
            {
                TimingResult t = time_matmul(sparse_matmul_pie_fn,
                                             &bs, x, y, rows, cols, NULL);
                results_print_row("sparse_pie", rows, cols, cfg->label,
                                  zero_frac, actual_zf, t);
            }

            // -- Cleanup --------------------------------------------------
            bitmask_free(&bm);
            block_sparse_free(&bs);
            free(W);

            // Small delay to allow UART to flush between zero fractions
            vTaskDelay(pdMS_TO_TICKS(10));
        }

        free(y);
        free(x);

        ESP_LOGI(TAG, "Done with %s", cfg->label);
    }

    ESP_LOGI(TAG, "All benchmarks complete.");
}


// ---------------------------------------------------------------------------
// app_main -- ESP-IDF entry point.
// ---------------------------------------------------------------------------

void app_main(void)
{
    ESP_LOGI(TAG, "mcu-ternary-matmul benchmark starting");
    ESP_LOGI(TAG, "CPU freq: %d MHz", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    ESP_LOGI(TAG, "Matrix configs: %d", N_MATRIX_CONFIGS);
    ESP_LOGI(TAG, "Zero fractions: %d", N_ZERO_FRACS);
    ESP_LOGI(TAG, "Timing runs per combination: %d", N_TIMING_RUNS);
    ESP_LOGI(TAG, "---");
    ESP_LOGI(TAG, "CSV output begins below:");
    ESP_LOGI(TAG, "---");

    // Small startup delay to allow serial monitor to connect
    vTaskDelay(pdMS_TO_TICKS(500));

    run_all_benchmarks();
}
