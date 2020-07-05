
__kernel void fvec_add(__global float * dst, __global float * v1,
        __global float * v2) {

    int index = get_global_id(0);
    dst[index] = v1[index] + v2[index];
}

