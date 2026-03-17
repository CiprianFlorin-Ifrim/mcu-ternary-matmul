.section .text
    .align  4

    .global pie_block_dot
    .type   pie_block_dot, @function

pie_block_dot:
    esp.zero.qacc

    mv      x28, a0
    mv      x29, a1

    esp.vld.128.ip  q0, x28, 0
    esp.vld.128.ip  q1, x29, 0
    esp.vmulas.s8.qacc  q0, q1

    // Store q2 to stack and load scalar result
    li      x28, 0
    esp.srcmb.s8.qacc   q2, x28, 1
    addi    sp, sp, -16
    mv      x28, sp
    esp.vst.128.ip  q2, x28, 0
    lw      a0, 0(sp)
    addi    sp, sp, 16

    ret

    .size pie_block_dot, . - pie_block_dot