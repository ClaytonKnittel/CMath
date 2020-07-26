
#define WIDTH (64 / 16)

#define row_t float16

__kernel void add(__global float * dst, __global float * m1,
        __global float * m2 /*, __private uint width*/) {

    __private row_t m1_seg, m2_seg;

#define width WIDTH

    {
        __private uint start_pos = 16 * get_global_id(0) * width;
        dst += start_pos;
        m1  += start_pos;
        m2  += start_pos;
    }

    __private uint col = 0;

#pragma unroll
    while (col < width) {
        m1_seg = vload16(0, m1);
        m2_seg = vload16(0, m2);

        m1_seg += m2_seg;
        vstore16(m1_seg, 0, dst);

        dst += 16;
        m1  += 16;
        m2  += 16;
        col++;
    }
}

