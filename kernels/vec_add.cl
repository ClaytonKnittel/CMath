
__kernel void vec_add(__global double * dst, __global double * v1,
        __global double * v2) {

    int index = get_global_id(0);
    dst[index] = v1[index] + v2[index];
}

