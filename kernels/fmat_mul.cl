
#define GROUP_HEIGHT 64
#define GROUP_WIDTH  1

/*
 * the number of columns in dst that each thread is responsible for, i.e. each
 * thread calculates a 1 x STRIDE block of dst
 */
#define GROUP_STRIDE 16

/*
 * the number of threads that work on a single block together, with one per row
 * in dst
 */
#define THREADS_PER_BLOCK GROUP_HEIGHT

/*
 * number of registers dedicated to storing accumulations of destination
 * values, i.e. the width of the block in dst being written to
 */
#define N_DST_REGS GROUP_STRIDE


#define M2_BUF_HEIGHT GROUP_STRIDE
#define M2_BUF_WIDTH  N_DST_REGS


__attribute__((reqd_work_group_size(GROUP_HEIGHT, GROUP_WIDTH, 1)))
__attribute__((xcl_zero_global_work_offset))
__kernel void fmat_mul(__global float * dst, __global float * m1,
        __global float * m2, uint m1_w) {

    __private float16 dst_regs;
    __private float m1_buf;
    __local float m2_buf[M2_BUF_HEIGHT * M2_BUF_WIDTH];

    uint g_r = get_group_id(0) * GROUP_HEIGHT;
    uint g_c = get_group_id(1) * GROUP_STRIDE;

    uint r = get_global_id(0);
#define c g_c

    uint m1_h = get_global_size(0);
    uint m2_w = get_global_size(1) * GROUP_STRIDE;
#define m2_h  m1_w
#define dst_h m1_h
#define dst_w m2_w

    // pointers to the locations in the given matrices we will be reading
    // from and writing to
    __global float * dst_ptr;
    __global float * m1_ptr;
    __global float * m2_ptr;

    const __global float * m2_ptr_end = m2_ptr + (m2_w * m2_h);

    event_t copy_ev;

    // clear accumulation registers
    dst_regs = (float16) (0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                          0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

    dst_ptr = dst + (r * dst_w) + c;
    m1_ptr  = m1 + (r * m1_w);
    m2_ptr  = m2 + c;

    while (((ulong) m2_ptr) < ((ulong) m2_ptr_end)) {
        // copy a 16x16 block from m2 into the local "cache"
        copy_ev = async_work_group_strided_copy(
                (__local float16 *) m2_buf,
                (__global float16 *) m2_ptr,
                M2_BUF_HEIGHT,
                m2_w / M2_BUF_WIDTH,
                (event_t) 0);
        wait_group_events(1, &copy_ev);

        #pragma unroll
        for (uint i = 0; i < M2_BUF_HEIGHT; i++) {
            m1_buf = m1_ptr[i];

            dst_regs.s0 = m1_buf * m2_buf[i * M2_BUF_HEIGHT +  0];
            dst_regs.s1 = m1_buf * m2_buf[i * M2_BUF_HEIGHT +  1];
            dst_regs.s2 = m1_buf * m2_buf[i * M2_BUF_HEIGHT +  2];
            dst_regs.s3 = m1_buf * m2_buf[i * M2_BUF_HEIGHT +  3];
            dst_regs.s4 = m1_buf * m2_buf[i * M2_BUF_HEIGHT +  4];
            dst_regs.s5 = m1_buf * m2_buf[i * M2_BUF_HEIGHT +  5];
            dst_regs.s6 = m1_buf * m2_buf[i * M2_BUF_HEIGHT +  6];
            dst_regs.s7 = m1_buf * m2_buf[i * M2_BUF_HEIGHT +  7];
            dst_regs.s8 = m1_buf * m2_buf[i * M2_BUF_HEIGHT +  8];
            dst_regs.s9 = m1_buf * m2_buf[i * M2_BUF_HEIGHT +  9];
            dst_regs.sa = m1_buf * m2_buf[i * M2_BUF_HEIGHT + 10];
            dst_regs.sb = m1_buf * m2_buf[i * M2_BUF_HEIGHT + 11];
            dst_regs.sc = m1_buf * m2_buf[i * M2_BUF_HEIGHT + 12];
            dst_regs.sd = m1_buf * m2_buf[i * M2_BUF_HEIGHT + 13];
            dst_regs.se = m1_buf * m2_buf[i * M2_BUF_HEIGHT + 14];
            dst_regs.sf = m1_buf * m2_buf[i * M2_BUF_HEIGHT + 15];
        }

        m1_ptr += M2_BUF_HEIGHT;
        m2_ptr += M2_BUF_HEIGHT * m2_w;

        // wait for all threads in work group to reach here before looping
        // and writing to m2 again
        barrier(0);
    }

    vstore16(dst_regs, 0, dst_ptr);

    /*float tot = 0;

    for (uint k = 0; k < m1_w; k++) {
        tot += m1[r * m1_w + k] * m2[k * m2_w + c];
    }

    dst[r * dst_w + c] = tot;*/
}

