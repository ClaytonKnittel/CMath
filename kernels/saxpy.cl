
__kernel void saxpy_kernel(float alpha, __global float * A,
        __global float * B, __global float * C) {

    int index = get_global_id(0);
    float a = A[index];
    float b = B[index];
    float res = alpha * a + b;
    C[index] = res;
}

