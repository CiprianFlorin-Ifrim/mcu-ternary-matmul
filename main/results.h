#pragma once

#include <stdio.h>
#include "matrix_utils.h"

// ---------------------------------------------------------------------------
// CSV output formatting for UART.
//
// All results are printed as CSV rows to make parsing straightforward in
// the Python analysis script. The header is printed once at startup.
//
// CSV columns:
//   approach      -- one of: dense_c, dense_pie, sparse_bitmask, sparse_pie
//   rows          -- matrix row count
//   cols          -- matrix col count
//   label         -- descriptive shape label from MatrixConfig
//   zero_frac_req -- requested zero fraction (may differ slightly from actual)
//   zero_frac_act -- actual zero fraction measured after matrix generation
//   mean_us       -- mean time in microseconds over N_TIMING_RUNS runs
//   min_us        -- minimum observed time
//   max_us        -- maximum observed time
//   n_runs        -- number of timing runs (always N_TIMING_RUNS)
// ---------------------------------------------------------------------------

#define RESULTS_CSV_HEADER \
    "approach,rows,cols,label,zero_frac_req,zero_frac_act," \
    "mean_us,min_us,max_us,n_runs\n"

static inline void results_print_header(void)
{
    printf(RESULTS_CSV_HEADER);
}

static inline void results_print_row(
    const char     *approach,
    int             rows,
    int             cols,
    const char     *label,
    float           zero_frac_req,
    float           zero_frac_act,
    TimingResult    t)
{
    printf("%s,%d,%d,%s,%.2f,%.4f,%.2f,%.2f,%.2f,%d\n",
           approach,
           rows,
           cols,
           label,
           zero_frac_req,
           zero_frac_act,
           t.mean_us,
           t.min_us,
           t.max_us,
           N_TIMING_RUNS);
}
