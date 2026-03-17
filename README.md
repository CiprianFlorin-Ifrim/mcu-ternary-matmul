# mcu-ternary-matmul

Exploring whether ternary weight matrices ({-1, 0, +1}) can be computed
efficiently on microcontroller hardware, specifically the ESP32-P4, using
its PIE (Processor Instruction Extensions) SIMD capabilities and sparse
bitmask representations.

This is not a full transformer implementation. The focus is on the fundamental
matrix-vector multiply operation that underlies every transformer layer, tested
at the specific shapes and sparsity levels produced by trained ternary models.

---

## Motivation

Ternary neural networks store weights as -1, 0, or +1 rather than float32 or
int8. At sufficient model capacity they match float32 accuracy on structured
tasks (see companion repo `ternary-transformer-lab`). The question this project
answers is whether the resulting weight sparsity (typically 40-70% zeros) can
be exploited on real embedded hardware to produce faster or more memory-efficient
inference than a dense int8 approach.

---

## Approaches Tested

| Approach         | Description                                                   |
|------------------|---------------------------------------------------------------|
| `dense_c`        | Dense INT8 matmul, plain C, no intrinsics. Compiler baseline. |
| `dense_pie`      | Dense INT8 matmul using ESP32-P4 PIE SIMD (128-bit Q regs).   |
| `sparse_bitmask` | Ternary weights as two bitmasks, scalar bit-manipulation loop. |
| `sparse_pie`     | 16-element block-sparse with PIE for non-zero blocks.          |

---

## Hardware

- **Board**: Guition JC-ESP32P4-M3 DEV (non-official, v1.3 silicon)
- **Chip**: ESP32-P4 revision v1.3
- **CPU**: Dual-core RISC-V HP at 360 MHz (v1.x silicon cap; v3.x reaches 400 MHz)
- **SRAM**: 768 KB on-chip (all matrices allocated here, no PSRAM involved)
- **PIE**: 8 × 128-bit Q registers, 2 × 256-bit QACC accumulators
- **Flash**: 16 MB Boya (generic driver, no functional impact)
- **PSRAM**: 32 MB octal (available but not used in these benchmarks)

### Hardware notes

The ESP32-P4 has two silicon revision families: v1.x and v3.x. They are
firmware-incompatible — a binary built targeting v3.x will refuse to boot on
v1.x silicon, producing a continuous `invalid header` boot loop. This is
controlled via `CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y` in `sdkconfig`.

v1.x silicon is officially rated at 360 MHz. An experimental option
(`CONFIG_ESP_FORCE_400MHZ_ON_REV_LESS_V3`) exists in ESP-IDF to push v1.x
to 400 MHz but requires Espressif qualification of the specific chip batch.
Enabling it on unqualified chips causes a clock initialisation assert
(`esp_clk_init clk.c:141`) and a boot loop. All results in this project are
from 360 MHz v1.3 silicon.

---

## PIE Assembly Implementation

### Design decisions

The PIE extension provides 128-bit SIMD load, store, and multiply-accumulate
instructions over Q registers (q0-q7) and a 256-bit QACC accumulator. The key
instructions used are:

- `esp.vld.128.ip qu, rs1, imm` — load 128 bits (16 INT8 values) into Q register,
  post-increment address by `imm`
- `esp.vmulas.s8.qacc qu, qv` — 16-way INT8 MAC of two Q registers into QACC
- `esp.zero.qacc` — zero the QACC accumulator
- `esp.srcmb.s8.qacc qu, rs1, 1` — horizontal reduction of QACC into Q register
- `esp.vst.128.ip qu, rs1, imm` — store Q register to memory

The address register constraint (`rs1`) only accepts specific GP registers.
The assembler accepts `x28`/`x29` (caller-saved `t3`/`t4`) reliably. All other
registers were found through trial-and-error since the ESP32-P4 TRM PIE chapter
was not publicly available at time of writing.

QACC horizontal reduction into a GP register via `esp.movi.32.a` was found to
have undocumented register constraints. The reliable workaround is a stack
round-trip: store the Q register to a 16-byte stack allocation via
`esp.vst.128.ip`, then load the first word with a standard `lw`.

### dense_pie assembly paths

The `dense_matmul_pie` function dispatches to one of three internal paths based
on the `cols` argument:

**Cached activation path (cols == 64)**
The activation vector (64 bytes = 4 Q registers) is loaded once into q4-q7
before the outer row loop. Each row only requires loading 4 weight Q registers
and 4 MAC instructions. For a matrix with R rows this saves 4R VLD instructions
compared to reloading activations per row.

**Bulk + tail path (cols > 16, any multiple of 16)**
The inner loop is 4-pair unrolled: 8 VLD instructions (4 weight + 4 activation
blocks, interleaved) followed by 4 MAC instructions, processing 64 elements per
iteration. After `cols / 64` bulk iterations, a tail loop handles the remaining
`(cols % 64) / 16` blocks one pair at a time. No padding zeros are introduced.

**Tail-only path (cols == 16 or 32)**
For small matrices the bulk loop has zero iterations and the tail loop processes
the entire row in 1 or 2 iterations respectively.

### Alignment

`matrix_utils.c` allocates all weight matrices and activation vectors with
`aligned_alloc(16, ...)` guaranteeing 16-byte alignment. `dense_pie.c` checks
alignment before each call — if cols is already a multiple of 16 and both
pointers are 16-byte aligned (the common case for all benchmark shapes except
`model_output_proj` with cols=15), the assembly is called directly with zero
allocation overhead. The slow path (pad and copy into fresh aligned buffers)
only fires for the cols=15 case.

---

## Experiments

Three benchmark runs were conducted, each with identical matrix configs and
zero fractions but with progressively improved assembly implementations.

### Experiment 1 — Baseline

**Assembly:** Single-pair inner loop (16 elements per iteration). No
path specialisation. `pad16` wrapper with `calloc` (no alignment guarantee).

**Key implementation detail:**
```
.Lcol_loop:
    esp.vld.128.ip  q0, x28, 16
    esp.vld.128.ip  q1, x29, 16
    esp.vmulas.s8.qacc  q0, q1
    addi    t0, t0, -16
    bgtz    t0, .Lcol_loop
```

**Fastest approach at 95% zero fraction:**

| Label                | Rows | Cols | Fastest      | Mean µs |
|----------------------|------|------|--------------|---------|
| mamba_state_16x16    |   16 |   16 | dense_pie    |    2.00 |
| mamba_state_32x32    |   32 |   32 | dense_pie    |    3.50 |
| mamba_ssm_64x16      |   64 |   16 | dense_pie    |    5.15 |
| mamba_ssm_64x32      |   64 |   32 | dense_pie    |    6.73 |
| mamba_ssm_128x16     |  128 |   16 | dense_pie    |    9.40 |
| mamba_ssm_128x32     |  128 |   32 | dense_pie    |   11.51 |
| mamba_ssm_256x16     |  256 |   16 | dense_pie    |   17.99 |
| mamba_ssm_256x32     |  256 |   32 | dense_pie    |   22.23 |
| square_32x32         |   32 |   32 | dense_pie    |    3.51 |
| square_64x64         |   64 |   64 | dense_pie    |    8.38 |
| square_128x128       |  128 |  128 | dense_pie    |   24.38 |
| square_256x256       |  256 |  256 | dense_pie    |   85.76 |
| model_attn_proj      |   64 |   64 | dense_pie    |    8.36 |
| model_ffn_up         |   64 |  128 | dense_pie    |   12.51 |
| model_ffn_down       |  128 |   64 | dense_pie    |   15.81 |
| model_output_proj    |   64 |   15 | sparse_pie   |    7.45 |
| tall_128x64          |  128 |   64 | dense_pie    |   15.81 |
| tall_256x64          |  256 |   64 | dense_pie    |   30.75 |
| tall_256x128         |  256 |  128 | dense_pie    |   47.83 |
| tall_512x64          |  512 |   64 | dense_pie    |   60.62 |
| tall_512x128         |  512 |  128 | dense_pie    |   97.42 |
| wide_64x128          |   64 |  128 | dense_pie    |   12.77 |
| wide_64x256          |   64 |  256 | dense_pie    |   21.68 |
| wide_128x256         |  128 |  256 | dense_pie    |   42.52 |
| wide_128x512         |  128 |  512 | dense_pie    |   77.92 |
| wide_256x512         |  256 |  512 | dense_pie    |  198.87 |
| swiglu_gate          |   64 |  128 | dense_pie    |   12.71 |
| swiglu_up            |   64 |  128 | dense_pie    |   12.67 |

**Notable finding:** dense_pie wins on 27 of 28 shapes regardless of sparsity
level. The only exception is `model_output_proj` (cols=15, non-multiple-of-16)
where the padding overhead inverts the result.

---

### Experiment 2 — Optimised assembly (regression)

**Assembly:** 4-pair unrolled loop (64 elements per iteration). `pad64`
wrapper replacing `pad16`. `aligned_alloc` but no fast path — every call
paid allocation overhead.

**Intended improvements:** 4x fewer loop iterations on large matrices,
reduced branch overhead.

**Actual result:** Regression across all shapes. sparse_pie won on every
shape. Root cause: the `pad64` wrapper called `aligned_alloc` + `memset` +
`memcpy` on every one of the 1000 timing iterations, making allocation
overhead dominate the measurement entirely.

**Lesson:** Benchmark wrappers that allocate on every timed call invalidate
the timing. The fast path (call assembly directly when already aligned and
padded) is essential to measure actual compute performance.

---

### Experiment 3 — Full optimisation

**Assembly:** Three-path dispatch (cached activation, bulk+tail, tail-only).
**Wrapper:** Fast path calling assembly directly when aligned; slow path only
for cols=15. All matrices allocated with `aligned_alloc(16)` in
`matrix_utils.c`.

**Fastest approach at 95% zero fraction:**

| Label                | Rows | Cols | Fastest      | Mean µs | vs Exp 1 |
|----------------------|------|------|--------------|---------|----------|
| mamba_state_16x16    |   16 |   16 | dense_pie    |    2.19 |  +0.19µs |
| mamba_state_32x32    |   32 |   32 | dense_pie    |    3.89 |  +0.39µs |
| mamba_ssm_64x16      |   64 |   16 | dense_pie    |    5.77 |  +0.62µs |
| mamba_ssm_64x32      |   64 |   32 | dense_pie    |    6.77 |  +0.04µs |
| mamba_ssm_128x16     |  128 |   16 | dense_pie    |   10.54 |  +1.14µs |
| mamba_ssm_128x32     |  128 |   32 | dense_pie    |   12.67 |  +1.16µs |
| mamba_ssm_256x16     |  256 |   16 | dense_pie    |   20.25 |  +2.26µs |
| mamba_ssm_256x32     |  256 |   32 | dense_pie    |   24.39 |  +2.16µs |
| square_32x32         |   32 |   32 | dense_pie    |    3.94 |  +0.43µs |
| square_64x64         |   64 |   64 | dense_pie    |    5.14 |   -3.24µs|
| square_128x128       |  128 |  128 | dense_pie    |   19.01 |   -5.37µs|
| square_256x256       |  256 |  256 | dense_pie    |   58.56 |  -27.20µs|
| model_attn_proj      |   64 |   64 | dense_pie    |    5.22 |   -3.14µs|
| model_ffn_up         |   64 |  128 | dense_pie    |   10.03 |   -2.48µs|
| model_ffn_down       |  128 |   64 | dense_pie    |    9.57 |   -6.24µs|
| model_output_proj    |   64 |   15 | sparse_pie   |    7.53 |  +0.08µs |
| tall_128x64          |  128 |   64 | dense_pie    |    9.55 |   -6.26µs|
| tall_256x64          |  256 |   64 | dense_pie    |   18.01 |  -12.74µs|
| tall_256x128         |  256 |  128 | dense_pie    |   37.21 |  -10.62µs|
| tall_512x64          |  512 |   64 | dense_pie    |   35.09 |  -25.53µs|
| tall_512x128         |  512 |  128 | dense_pie    |   75.74 |  -21.68µs|
| wide_64x128          |   64 |  128 | dense_pie    |   10.03 |   -2.74µs|
| wide_64x256          |   64 |  256 | dense_pie    |   14.99 |   -6.69µs|
| wide_128x256         |  128 |  256 | dense_pie    |   29.08 |  -13.44µs|
| wide_128x512         |  128 |  512 | dense_pie    |   50.46 |  -27.46µs|
| wide_256x512         |  256 |  512 | dense_pie    |  143.33 |  -55.54µs|
| swiglu_gate          |   64 |  128 | dense_pie    |   10.05 |   -2.66µs|
| swiglu_up            |   64 |  128 | dense_pie    |   10.01 |   -2.66µs|

**Observations:**

Shapes with cols=64 improved by roughly 1.6x due to the cached activation path
(load x once before the row loop rather than once per row). `model_ffn_down`
(128×64) improved from 15.81µs to 9.57µs — a 39% reduction.

Large matrices improved by 1.3-1.4x from the 4-pair bulk unrolled loop.
`square_256x256` went from 85.76µs to 58.56µs. `wide_256x512` went from
198.87µs to 143.33µs.

Small matrix shapes (cols=16, cols=32) regressed by 0.2-1.2µs due to added
branch overhead for the path dispatcher. For these tiny matrices the dispatch
cost is comparable to the actual computation cost. In a real inference workload
these shapes would be negligible in total runtime.

---

## Overall Findings

### dense_pie dominates at all sparsity levels

Across all three experiments and all zero fractions from 0% to 95%, dense_pie
is the fastest approach on 27 of 28 shapes. Sparsity never tips the balance
toward sparse approaches for cols≥16 aligned matrices.

The reason is throughput. `esp.vmulas.s8.qacc` processes 16 INT8 MAC operations
per cycle. The bitmask approach processes at most 1 multiply per cycle and has
significant overhead per non-zero element from the `__builtin_ctz` bit
extraction loop. Even at 95% zeros, the 5% of non-zero computations in the
bitmask approach take longer than processing all 100% of elements with PIE.

### sparse_bitmask is slower than dense_c in almost every case

The bit manipulation overhead completely outweighs the zero-skipping benefit.
At 0% zeros, sparse_bitmask is roughly 4-8x slower than dense_c on large
matrices. Even at 95% zeros it rarely matches dense_c, and never matches
dense_pie.

### The only sparse win is model_output_proj (cols=15)

This shape has a non-multiple-of-16 column count which forces dense_pie to
pad to 16 and pay allocation overhead on every call. sparse_pie wins here
because the block-sparse C dispatch avoids this entirely. In production
deployment the output projection would be padded once at model load time,
eliminating this case.

### Speedup of dense_pie over dense_c at 0% zeros

| Matrix size  | dense_c (µs) | dense_pie (µs) | Speedup |
|--------------|-------------|----------------|---------|
| 32×32        |       24.7  |           3.5  |   7.1×  |
| 64×64        |       94.1  |           5.1  |  18.4×  |
| 128×128      |      369.4  |          19.0  |  19.4×  |
| 256×256      |     1468.4  |          58.6  |  25.1×  |
| 128×512      |     1463.9  |          50.5  |  29.0×  |
| 256×512      |     2969.7  |         143.3  |  20.7×  |

The speedup grows with matrix size as the benefit of processing 16 elements
per cycle over 1 element per iteration compounds.

---

## sdkconfig and Versioning Notes

Getting the project to boot on v1.x silicon required resolving several
non-obvious configuration issues, documented here for reproducibility.

### Bootloader address

The ESP32-P4 bootloader loads at address `0x2000`, not `0x0` as on older
ESP32 variants. Flashing with the address hardcoded to `0x0` produces a
continuous `invalid header` boot loop.

### Silicon revision targeting

ESP-IDF v5.5.3 defaults to targeting v3.x silicon. Flashing a v3.x-targeted
binary to a v1.x chip fails with `Image requires chip rev >= v3.1`.
The fix is adding to `sdkconfig`:

```
CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y
```

This option is only present in ESP-IDF v5.5.3 and later. Manually setting
`CONFIG_ESP32P4_REV_MIN_FULL=100` as an alternative does not fully resolve
the issue because `sdkconfig.defaults` is ignored when `sdkconfig` already
exists. The reliable workflow is:

```bash
rm sdkconfig
rm -rf build
idf.py build
```

### 400 MHz on v1.x silicon

`CONFIG_ESP_FORCE_400MHZ_ON_REV_LESS_V3=y` exists in menuconfig for v1.x
silicon but is clearly marked experimental and requires Espressif qualification
of the specific chip batch. Enabling it on unqualified chips causes a hard
crash at boot with `assert failed: esp_clk_init clk.c:141`. All benchmarks
in this project run at 360 MHz, which is the rated maximum for v1.x.

### sdkconfig.defaults vs sdkconfig

`sdkconfig.defaults` is only read during a fresh project configuration (when
no `sdkconfig` exists). Any subsequent build reuses the existing `sdkconfig`
regardless of changes to `sdkconfig.defaults`. Always delete both `sdkconfig`
and `build/` when changing fundamental settings like silicon revision targeting
or CPU frequency.

### Working sdkconfig settings for v1.x silicon

```
CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=360
CONFIG_BOOTLOADER_LOG_LEVEL_NONE=y
CONFIG_PARTITION_TABLE_OFFSET=0xA000
CONFIG_ESP_TASK_WDT_EN=n
CONFIG_ESP_INT_WDT=n
CONFIG_COMPILER_OPTIMIZATION_PERF=y
CONFIG_ESP_CONSOLE_UART_BAUDRATE=921600
```

---

## Matrix Shapes Tested

| Group              | Shapes                                              |
|--------------------|-----------------------------------------------------|
| Mamba SSM          | 16×16, 32×32, 64×16, 64×32, 128×16, 128×32, 256×16, 256×32 |
| Square             | 32×32, 64×64, 128×128, 256×256                      |
| Rectangular tall   | 128×64, 256×64, 256×128, 512×64, 512×128            |
| Rectangular wide   | 64×128, 64×256, 128×256, 128×512, 256×512           |
| Realistic layers   | 64×64, 64×128, 128×64, 64×15                        |
| SwiGLU             | 64×128 (×2)                                         |

Zero fractions: 0%, 30%, 50%, 70%, 90%, 95%
Timing runs per combination: 1000
Total combinations: 28 shapes × 6 zero fractions × 4 approaches = 672

---

## Repository Structure

```
mcu-ternary-matmul/
  main/
    main.c              Test harness, timing loop, CSV output over UART
    matrix_utils.c/h    Matrix generation (aligned_alloc), bitmask conversion,
                        timing harness
    dense_c.c/h         Approach 1: dense plain C
    dense_pie.c/h/.S    Approach 2: dense PIE assembly with three dispatch paths
    sparse_bitmask.c/h  Approach 3: bitmask sparse scalar C
    sparse_pie.c/h/.S   Approach 4: block-sparse PIE hybrid
    results.h           CSV output formatting
    CMakeLists.txt
  analysis/
    parse_results.py    CSV parser and plot generator
    requirements.txt
  sdkconfig             Working configuration for v1.3 silicon at 360 MHz
  CMakeLists.txt
  README.md
  .gitignore
```

---

## Setup

### Requirements

- ESP-IDF v5.5.3 or later (required for `CONFIG_ESP32P4_SELECTS_REV_LESS_V3`)
- ESP32-P4 board with v1.x or v3.x silicon (separate firmware required for each)
- Python 3.x with pip for the analysis script

### Build and flash

```bash
# Activate ESP-IDF environment
source ~/.espressif/v5.5.3/esp-idf/export.sh

# First time only -- set target and generate sdkconfig
idf.py set-target esp32p4
idf.py menuconfig   # verify CONFIG_ESP32P4_SELECTS_REV_LESS_V3 matches your silicon

# Build
idf.py build

# Flash using flash_args for correct addresses
cd build && python -m esptool --chip esp32p4 --port /dev/cu.usbmodem1101 \
    --baud 921600 write_flash @flash_args
```

### Monitor

```bash
idf.py -p /dev/cu.usbmodem1101 monitor
```

### Analyse results

Save the CSV output from the monitor (remove ESP-IDF log lines, keep only
the CSV rows), then:

```bash
cd analysis
pip install -r requirements.txt
python parse_results.py --file ../results.csv --out plots/
```

---

## Related Work

Companion project `ternary-transformer-lab` trains float32 and ternary
transformers on synthetic mathematical tasks and benchmarks their accuracy,
convergence behaviour, and inference speed on MPS (Apple Silicon). The matrix
shapes in this project correspond directly to the layer sizes of the 64-dim
ternary model trained in that project.