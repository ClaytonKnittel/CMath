

#define N GROUP_SIZE

#define __vload(n) vload ## n
#define _vload(n) __vload(n)
#define vload _vload(N)

#define __vstore(n) vstore ## n
#define _vstore(n) __vstore(n)
#define vstore _vstore(N)

#define __dtype(n) float ## n
#define _dtype(n) __dtype(n)

#if N == 1
#define dtype float
#else
#define dtype _dtype(N)
#endif


// assume data comes in groups of 16
__kernel void fvec_add(__global dtype * dst, __global dtype * v1,
        __global dtype * v2) {

#if N != 1
    dtype p1, p2;
#endif
    uint index = get_global_id(0);

#if N != 1
    p1 = vload(index, (__global float *) v1);
    p2 = vload(index, (__global float *) v2);
    p1 += p2;
    vstore(p1, index, (__global float *) dst);
#else
    dst[index] = v1[index] + v2[index];
#endif

}

