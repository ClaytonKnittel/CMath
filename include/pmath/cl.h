#ifndef _CL_H
#define _CL_H

#include <stddef.h>
#include <stdint.h>


// list of all operations
typedef enum operation {
    fvec_add,
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
        const char * kernel_name, struct __int_cl_context * ctxt);


/*
 * to be called before using any operation, can be called multiple times on a
 * single op, but only initializes the op the first time. Should initialize the
 * program and kernel objects for the given operation
 *
 * callback is the operation initializer, which loads the program and kernel
 * objects for the op
 */
static __attribute__((always_inline)) int cl_load_op(operation_t op,
        const char * program_name, const char * kernel_name) {
    struct __int_cl_context * global_context;

    global_context = __cl_get_global_context();

    return __int_cl_load_op(op, program_name, kernel_name, global_context);
}


/*
 * alias for clSetKernelArg
 */
void cl_set_param(operation_t op, uint32_t param_idx, size_t arg_size,
        const void * arg);


static int cl_execute_op() {
    return 0;
}


#endif /* _CL_H */
