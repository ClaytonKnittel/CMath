#ifndef _CL_H
#define _CL_H

#include <stddef.h>
#include <stdint.h>

#include <OpenCL/opencl.h>


// list of all operations
typedef enum operation {
    vec_add,
    fvec_add,
    fmat_mul,
    n_operations
} operation_t;



// forward declaration
struct __int_cl_context;

/*
 * returns a pointer to the global cl context, initializing it if it has not
 * yet been initialized
 */
struct __int_cl_context * __cl_get_global_context();


int __int_cl_load_op(operation_t op, const char * program_name,
        const char * kernel_name, const char * opts,
        struct __int_cl_context * ctxt);


/*
 * to be called before using any operation, can be called multiple times on a
 * single op, but only initializes the op the first time. Should initialize the
 * program and kernel objects for the given operation
 *
 * callback is the operation initializer, which loads the program and kernel
 * objects for the op
 */
static __attribute__((always_inline)) int cl_load_op(operation_t op,
        const char * program_name, const char * kernel_name,
        const char * opts) {
    struct __int_cl_context * global_context;

    global_context = __cl_get_global_context();

    return __int_cl_load_op(op, program_name, kernel_name, opts,
            global_context);
}


/*
 * decompiles operation source code binary and puts it in buf
 */
void cl_get_op_binary(operation_t op, unsigned char * buf, size_t buf_len,
        size_t * write_size);


/*
 * alias for clCreateBuffer
 */
cl_mem cl_create_buffer(int flags, size_t n_bytes, void * ptr);

/*
 * read memory from cl buffer into dst, blocks until prerequisite operations
 * have been complete and the memory transfer is finished
 */
void cl_read_buffer(cl_mem cl_buf, size_t offset, size_t n_bytes, void * dst);

/*
 * alias for clReleaseMemObject
 */
void cl_delete_buffer(cl_mem buf);



/*
 * alias for clSetKernelArg
 */
void cl_set_param(operation_t op, uint32_t param_idx, size_t arg_size,
        const void * arg);


int cl_execute_op(operation_t op_idx, uint32_t n_dims, size_t * global_sizes,
        size_t * local_sizes);


void cl_finish();


#endif /* _CL_H */
