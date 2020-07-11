
// assume data comes in groups of 16
__kernel void fvec_add(__global float * dst, __global float * v1,
        __global float * v2) {

    int index = 16 * get_global_id(0);
    dst[index +  0] = v1[index +  0] + v2[index +  0];
    dst[index +  1] = v1[index +  1] + v2[index +  1];
    dst[index +  2] = v1[index +  2] + v2[index +  2];
    dst[index +  3] = v1[index +  3] + v2[index +  3];
    dst[index +  4] = v1[index +  4] + v2[index +  4];
    dst[index +  5] = v1[index +  5] + v2[index +  5];
    dst[index +  6] = v1[index +  6] + v2[index +  6];
    dst[index +  7] = v1[index +  7] + v2[index +  7];
    dst[index +  8] = v1[index +  8] + v2[index +  8];
    dst[index +  9] = v1[index +  9] + v2[index +  9];
    dst[index + 10] = v1[index + 10] + v2[index + 10];
    dst[index + 11] = v1[index + 11] + v2[index + 11];
    dst[index + 12] = v1[index + 12] + v2[index + 12];
    dst[index + 13] = v1[index + 13] + v2[index + 13];
    dst[index + 14] = v1[index + 14] + v2[index + 14];
    dst[index + 15] = v1[index + 15] + v2[index + 15];
}

