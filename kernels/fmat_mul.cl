

__kernel void fmat_mul(__global float * dst, __global float * m1,
        __global float * m2, uint m1_w) {

    uint r = get_global_id(0);
    uint c = get_global_id(1);

    //uint m1_h = get_global_size(0);
    uint m2_w = get_global_size(1);

    float tot = 0;

    for (uint k = 0; k < m1_w; k++) {
        tot += m1[r * m1_w + k] * m2[k * m2_w + c];
    }

    dst[r * m2_w + c] = tot;
}

