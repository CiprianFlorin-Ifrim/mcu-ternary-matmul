    .section .text
    .align  4

    .global dense_matmul_pie
    .type   dense_matmul_pie, @function

// dense_matmul_pie(const int8_t *W, const int8_t *x, int32_t *y, int rows, int cols)
//
// cols must be a multiple of 16 (enforced by the C wrapper pad16()).
//
// Two execution paths:
//   cols == 64 -> cached path: load x once into q4-q7, reload only weights per row.
//   otherwise  -> remainder path: 4-pair unrolled bulk loop (cols/64 iterations)
//                 followed by single-pair tail loop ((cols%64)/16 iterations).
//                 Handles any multiple-of-16 cols with zero padding.
//
// Register allocation:
//   s0 = W row pointer (advances each row)
//   s1 = x activation vector (constant)
//   s2 = y output pointer
//   s3 = rows remaining
//   s4 = cols
//   s5 = full_iters  = cols >> 6  (number of 64-element chunks)
//   s6 = tail_iters  = (cols & 63) >> 4  (remaining 16-element blocks)
//   x28/x29 = PIE address registers (caller-saved, no save needed)

dense_matmul_pie:
    addi    sp, sp, -32
    sw      s0,  0(sp)
    sw      s1,  4(sp)
    sw      s2,  8(sp)
    sw      s3, 12(sp)
    sw      s4, 16(sp)
    sw      s5, 20(sp)
    sw      s6, 24(sp)

    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3
    mv      s4, a4

    beqz    s3, .Ldone

    // Precompute loop counts from cols
    srli    s5, s4, 6           // full_iters = cols / 64
    andi    t0, s4, 63          // cols % 64
    srli    s6, t0, 4           // tail_iters = (cols % 64) / 16

    // Select cached path if cols == 64 exactly
    li      t1, 64
    bne     s4, t1, .Lremainder_path

    // ------------------------------------------------------------------
    // Cached activation path (cols == 64)
    // Load x once into q4-q7 before the row loop.
    // ------------------------------------------------------------------
    mv      x28, s1
    esp.vld.128.ip  q4, x28, 16
    esp.vld.128.ip  q5, x28, 16
    esp.vld.128.ip  q6, x28, 16
    esp.vld.128.ip  q7, x28, 16

.Lcached_row_loop:
    esp.zero.qacc
    mv      x28, s0

    esp.vld.128.ip  q0, x28, 16
    esp.vld.128.ip  q1, x28, 16
    esp.vld.128.ip  q2, x28, 16
    esp.vld.128.ip  q3, x28, 16

    esp.vmulas.s8.qacc  q0, q4
    esp.vmulas.s8.qacc  q1, q5
    esp.vmulas.s8.qacc  q2, q6
    esp.vmulas.s8.qacc  q3, q7

    li      x28, 0
    esp.srcmb.s8.qacc   q2, x28, 1
    addi    sp, sp, -16
    mv      x28, sp
    esp.vst.128.ip  q2, x28, 0
    lw      t0, 0(sp)
    addi    sp, sp, 16

    sw      t0, 0(s2)
    addi    s2, s2, 4
    add     s0, s0, s4
    addi    s3, s3, -1
    bgtz    s3, .Lcached_row_loop

    j       .Ldone

    // ------------------------------------------------------------------
    // Remainder path (any multiple-of-16 cols)
    // Bulk loop processes 64 elements/iter, tail loop handles the rest.
    // ------------------------------------------------------------------
.Lremainder_path:
.Lrow_loop:
    esp.zero.qacc
    mv      x28, s0
    mv      x29, s1
    mv      t1,  s5             // bulk iteration counter

    beqz    t1, .Ltail_start

.Lbulk_loop:
    esp.vld.128.ip  q0, x28, 16
    esp.vld.128.ip  q4, x29, 16
    esp.vld.128.ip  q1, x28, 16
    esp.vld.128.ip  q5, x29, 16
    esp.vld.128.ip  q2, x28, 16
    esp.vld.128.ip  q6, x29, 16
    esp.vld.128.ip  q3, x28, 16
    esp.vld.128.ip  q7, x29, 16

    esp.vmulas.s8.qacc  q0, q4
    esp.vmulas.s8.qacc  q1, q5
    esp.vmulas.s8.qacc  q2, q6
    esp.vmulas.s8.qacc  q3, q7

    addi    t1, t1, -1
    bgtz    t1, .Lbulk_loop

.Ltail_start:
    mv      t1, s6              // tail iteration counter
    beqz    t1, .Lextract

.Ltail_loop:
    esp.vld.128.ip  q0, x28, 16
    esp.vld.128.ip  q1, x29, 16
    esp.vmulas.s8.qacc  q0, q1

    addi    t1, t1, -1
    bgtz    t1, .Ltail_loop

.Lextract:
    li      x28, 0
    esp.srcmb.s8.qacc   q2, x28, 1
    addi    sp, sp, -16
    mv      x28, sp
    esp.vst.128.ip  q2, x28, 0
    lw      t0, 0(sp)
    addi    sp, sp, 16

    sw      t0, 0(s2)
    addi    s2, s2, 4
    add     s0, s0, s4
    addi    s3, s3, -1
    bgtz    s3, .Lrow_loop

.Ldone:
    lw      s0,  0(sp)
    lw      s1,  4(sp)
    lw      s2,  8(sp)
    lw      s3, 12(sp)
    lw      s4, 16(sp)
    lw      s5, 20(sp)
    lw      s6, 24(sp)
    addi    sp, sp, 32
    ret

    .size dense_matmul_pie, . - dense_matmul_pie
