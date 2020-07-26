/*
 * Algorithm is based on:
 * https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf
 * https://www.cise.ufl.edu/~sahni/papers/strassen.pdf
 */


/*
 * macro for defining a vector type of given size (i.e. int16)
 */
#define _DEF_VEC_T(type, size) type ## size
#define DEF_VEC_T(type, size) _DEF_VEC_T(type, size)



//#define GROUP_HEIGHT
#define GROUP_WIDTH  1

/*
 * the number of columns in dst that each thread is responsible for, i.e. each
 * thread calculates a 1 x STRIDE block of dst
 */
//#define GROUP_STRIDE

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
#define m2_w (get_global_size(1) * GROUP_STRIDE)
#define m2_h  m1_w
#define dst_w m2_w
//#define dst_h m1_h

    const __global float * m2_end = m2 + (m2_h * m2_w);

    event_t copy_ev;

    // clear accumulation registers
    dst_regs = 0.f;

    m1 += (r * m1_w);
    m2 += c;

    while (((ulong) m2) < ((ulong) m2_end)) {
        // copy a 16x16 block from m2 into the local "cache"
        copy_ev = async_work_group_strided_copy(
                (__local  m2_block_row_t *) m2_buf,
                (__global m2_block_row_t *) m2,
                M2_BUF_HEIGHT,
                m2_w / M2_BUF_WIDTH,
                (event_t) 0);
        wait_group_events(1, &copy_ev);

        /*
        uint idx = get_local_id(0);
        ((__local float8 *) m2_buf)[(idx >> 1) + (idx & 1)]
            = ((__global float8 *) m2)[(idx >> 1) * m2_w / M2_BUF_WIDTH + (idx & 1)];
        barrier(CLK_LOCAL_MEM_FENCE);*/

        m1_buf = vload(0, m1);

#define r01 dst_regs
                                 r01 +=
            m1_buf.s0 * ((__local m2_block_row_t *) m2_buf)[ 0] +
            m1_buf.s1 * ((__local m2_block_row_t *) m2_buf)[ 1];
        __private m2_block_row_t r23 =
            m1_buf.s2 * ((__local m2_block_row_t *) m2_buf)[ 2] +
            m1_buf.s3 * ((__local m2_block_row_t *) m2_buf)[ 3];
        __private m2_block_row_t r45 =
            m1_buf.s4 * ((__local m2_block_row_t *) m2_buf)[ 4] +
            m1_buf.s5 * ((__local m2_block_row_t *) m2_buf)[ 5];
        __private m2_block_row_t r67 =
            m1_buf.s6 * ((__local m2_block_row_t *) m2_buf)[ 6] +
            m1_buf.s7 * ((__local m2_block_row_t *) m2_buf)[ 7];
        __private m2_block_row_t r89 =
            m1_buf.s8 * ((__local m2_block_row_t *) m2_buf)[ 8] +
            m1_buf.s9 * ((__local m2_block_row_t *) m2_buf)[ 9];
        __private m2_block_row_t rab =
            m1_buf.sa * ((__local m2_block_row_t *) m2_buf)[10] +
            m1_buf.sb * ((__local m2_block_row_t *) m2_buf)[11];
        __private m2_block_row_t rcd =
            m1_buf.sc * ((__local m2_block_row_t *) m2_buf)[12] +
            m1_buf.sd * ((__local m2_block_row_t *) m2_buf)[13];
        __private m2_block_row_t ref =
            m1_buf.se * ((__local m2_block_row_t *) m2_buf)[14] +
            m1_buf.sf * ((__local m2_block_row_t *) m2_buf)[15];

        dst_regs = ((r01 + r23) + (r45 + r67)) + ((r89 + rab) + (rcd + ref));


        m1 += M2_BUF_HEIGHT;
        m2 += m2_w * M2_BUF_HEIGHT;

        // wait for all threads in work group to reach here before looping
        // and writing to m2 again
        barrier(0);
    }

    vstore(dst_regs, 0, dst + (r * dst_w) + c);

}

