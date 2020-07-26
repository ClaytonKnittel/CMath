

#define row_t float16

__kernel void sub(__global float * dst, __global float * m1,
        __global float * m2, __private uint width) {

    __private row_t m1_seg, m2_seg;

    {
        __private uint start_pos = get_global_id(0) * width;
        dst += start_pos;
        m1  += start_pos;
        m2  += start_pos;
    }

    __private uint col = 0;

    while (col < width) {
        m1_seg = vload16(0, m1);
        m2_seg = vload16(0, m2);

        m1_seg -= m2_seg;
        vstore16(m1_seg, 0, dst);

        dst += 16;
        m1  += 16;
        m2  += 16;
        col += 16;
    }
}

