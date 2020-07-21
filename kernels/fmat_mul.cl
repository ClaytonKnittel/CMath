

/*
 * macro for defining a vector type of given size (i.e. int16)
 */
#define _DEF_VEC_T(type, size) type ## size
#define DEF_VEC_T(type, size) _DEF_VEC_T(type, size)



//#define GROUP_HEIGHT 64
#define GROUP_WIDTH  1

/*
 * the number of columns in dst that each thread is responsible for, i.e. each
 * thread calculates a 1 x STRIDE block of dst
 */
//#define GROUP_STRIDE 16

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

#define m2_block_row_t DEF_VEC_T(float, M2_BUF_WIDTH)

#define vload  DEF_VEC_T(vload,  M2_BUF_WIDTH)
#define vstore DEF_VEC_T(vstore, M2_BUF_WIDTH)


__attribute__((reqd_work_group_size(GROUP_HEIGHT, GROUP_WIDTH, 1)))
//__attribute__((xcl_zero_global_work_offset))
__kernel void fmat_mul(__global float * dst, __global float * m1,
        __global float * m2, uint m1_w) {

    __private m2_block_row_t dst_regs;
    __private m2_block_row_t m1_buf;
    __local float m2_buf[M2_BUF_HEIGHT * M2_BUF_WIDTH];

    // height and width of the matrix
#define r get_global_id(0)
#define c (get_group_id(1) * GROUP_STRIDE)

    //uint m1_h = get_global_size(0);
    uint m2_w = get_global_size(1) * GROUP_STRIDE;
#define m2_h  m1_w
//#define dst_h m1_h
#define dst_w m2_w

    const __global float * m2_end = m2 + (m2_w * m2_h);

    event_t copy_ev;

    // clear accumulation registers
    dst_regs = 0.f;

    m1  = m1 + (r * m1_w);
    m2  = m2 + c;

    while (((ulong) m2) < ((ulong) m2_end)) {
        // copy a 16x16 block from m2 into the local "cache"
        copy_ev = async_work_group_strided_copy(
                (__local  m2_block_row_t *) m2_buf,
                (__global m2_block_row_t *) m2,
                M2_BUF_HEIGHT,
                m2_w / M2_BUF_WIDTH,
                (event_t) 0);
        wait_group_events(1, &copy_ev);

        /*uint idx = r - g_r;
        for (uint offset = 0; offset < M2_BUF_WIDTH; offset++) {
            m2_buf[idx * M2_BUF_WIDTH + offset] = m2[idx * m2_w + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);*/

        m1_buf = vload(0, m1);

        /*
        #pragma unroll
        for (uint i = 0; i < M2_BUF_HEIGHT; i++) {
            __private float m1_val = m1_buf[i];

            dst_regs += m1_val * ((__local m2_block_row_t *) m2_buf)[i];
        }*/

        // explicit loop unroll
        dst_regs += (m1_buf.s0 * ((__local m2_block_row_t *) m2_buf)[ 0]);
                  + (m1_buf.s1 * ((__local m2_block_row_t *) m2_buf)[ 1]);
                  + (m1_buf.s2 * ((__local m2_block_row_t *) m2_buf)[ 2]);
                  + (m1_buf.s3 * ((__local m2_block_row_t *) m2_buf)[ 3]);
                  + (m1_buf.s4 * ((__local m2_block_row_t *) m2_buf)[ 4]);
                  + (m1_buf.s5 * ((__local m2_block_row_t *) m2_buf)[ 5]);
                  + (m1_buf.s6 * ((__local m2_block_row_t *) m2_buf)[ 6]);
                  + (m1_buf.s7 * ((__local m2_block_row_t *) m2_buf)[ 7]);
                  + (m1_buf.s8 * ((__local m2_block_row_t *) m2_buf)[ 8]);
                  + (m1_buf.s9 * ((__local m2_block_row_t *) m2_buf)[ 9]);
                  + (m1_buf.sa * ((__local m2_block_row_t *) m2_buf)[10]);
                  + (m1_buf.sb * ((__local m2_block_row_t *) m2_buf)[11]);
                  + (m1_buf.sc * ((__local m2_block_row_t *) m2_buf)[12]);
                  + (m1_buf.sd * ((__local m2_block_row_t *) m2_buf)[13]);
                  + (m1_buf.se * ((__local m2_block_row_t *) m2_buf)[14]);
                  + (m1_buf.sf * ((__local m2_block_row_t *) m2_buf)[15]);

        m1 += M2_BUF_HEIGHT;
        m2 += M2_BUF_HEIGHT * m2_w;

        // wait for all threads in work group to reach here before looping
        // and writing to m2 again
        barrier(0);
    }

    vstore(dst_regs, 0, dst + (r * dst_w) + c);

}

