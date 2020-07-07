
#include <assert.h>

#include <pmath/cl.h>
#include <pmath/vec.h>


int main(int argc, char *argv[]) {
    _load_fvec_add();

#define SIZE 1024
    float * a = (float*) malloc(SIZE * sizeof(float));
    float * b = (float*) malloc(SIZE * sizeof(float));
    float * c = (float*) malloc(SIZE * sizeof(float));

    for (int i = 0; i < SIZE; i++) {
        a[i] = i + 1;
        b[i] = i + 2;
    }

    cl_mem a_cl = cl_create_buffer(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            SIZE * sizeof(float), a);

    cl_mem b_cl = cl_create_buffer(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            SIZE * sizeof(float), b);

    cl_mem c_cl = cl_create_buffer(CL_MEM_WRITE_ONLY, SIZE * sizeof(float),
            NULL);

    cl_set_param(fvec_add, 0, sizeof(c_cl), &c_cl);
    cl_set_param(fvec_add, 1, sizeof(a_cl), &a_cl);
    cl_set_param(fvec_add, 2, sizeof(b_cl), &b_cl);

    cl_execute_op(fvec_add, SIZE);
    cl_read_buffer(c_cl, 0, SIZE * sizeof(float), c);

    for (int i = 0; i < SIZE; i++) {
        assert(c[i] == 2 * i + 3);
    }

    cl_delete_buffer(c_cl);
    cl_delete_buffer(b_cl);
    cl_delete_buffer(a_cl);

    free(c);
    free(b);
    free(a);

    return 0;
}

