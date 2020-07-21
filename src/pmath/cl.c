
#include <stdio.h>

#include <OpenCL/opencl.h>


#include <pmath/cl.h>



#define USE_DEVICE 2


// universal flag
#define INITIALIZED 0x1


struct __int_op {
    cl_program prog;
    cl_kernel kernel;
    int flags;
};



/*
 * we have one context per device
 */
struct __int_cl_context {
    cl_context context;

    cl_uint num_devices;
    cl_device_id * device_ids;
    cl_command_queue * command_queues;

    int flags;

    struct __int_op ops[n_operations];
};


/*
 * default global context, to be initialized once by the first API call
 */
struct __int_cl_context __global_context = {
    .flags = 0
};





/*
 * initializes given cl_context, called automatically by all of the below
 * functions if the global context has not yet been initialized
 */
static int _cl_init(struct __int_cl_context * c) {

    cl_uint num_platforms;
    cl_platform_id * platforms = NULL;

    // set up the platform
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms =
        (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, NULL);

    // get the devices list
    cl_uint num_devices;
    cl_device_id * device_ids = NULL;

    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    device_ids = (cl_device_id *)
        malloc(sizeof(cl_device_id)*num_devices);
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices,
            device_ids, NULL);


    cl_context context;
    // create the OpenCL context
    cl_context_properties props[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) platforms[0],
        0
    };
    context = clCreateContext(props, num_devices, device_ids, NULL, NULL, &err);

    if (context == NULL) {
        fprintf(stderr, "Unable to initialize OpenCL context, ");
        switch (err) {
            case CL_INVALID_PLATFORM:
                fprintf(stderr, "platform value is invalid\n");
                break;
            case CL_INVALID_VALUE:
                fprintf(stderr, "no devices could be found\n");
                break;
            case CL_DEVICE_NOT_AVAILABLE:
                fprintf(stderr, "device not available\n");
                break;
            case CL_OUT_OF_HOST_MEMORY:
                fprintf(stderr, "the host was unable to allocate OpenCL "
                        "resources\n");
                break;
        }
        free(device_ids);
        return -1;
    }

    cl_command_queue * command_queues =
        (cl_command_queue *) malloc(num_devices * sizeof(cl_command_queue));

    // Create a command queue for each device
    for (int device_n = 0; device_n < num_devices; device_n++) {
        cl_command_queue command_queue = clCreateCommandQueue(context,
                device_ids[device_n], 0, &err);

        if (err != CL_SUCCESS) {
            fprintf(stderr, "failed to create command queue");
            clReleaseContext(context);
            free(command_queues);
            free(device_ids);
            return -1;
        }

        command_queues[device_n] = command_queue;
    }

    c->context = context;

    c->num_devices = num_devices;
    c->device_ids = device_ids;
    c->command_queues = command_queues;
    c->flags = INITIALIZED;
    __builtin_memset(c->ops, 0, n_operations * sizeof(struct __int_op));

    for (int i = 0; i < num_devices; i++) {
        cl_print_device_info(device_ids[i]);
    }

    return 0;
}


struct __int_cl_context * __cl_get_global_context() {
    struct __int_cl_context * gc = &__global_context;

    if (__builtin_expect(!(gc->flags & INITIALIZED), 0)) {
        _cl_init(gc);
    }

    return gc;
}



int __int_cl_load_op(operation_t op_idx, const char * program_name,
        const char * kernel_name, const char * opts,
        struct __int_cl_context * ctxt) {

    struct __int_op * op = &ctxt->ops[op_idx];

    if (__builtin_expect(!(op->flags & INITIALIZED), 0)) {
        // TODO cb & set initialized flag
        int err = 0;

        op->prog = clCreateProgramWithSource(ctxt->context, 1, &program_name,
                NULL, &err);

        if (err) {
            fprintf(stderr, "Error loading program \"%s\"\n", program_name);
            return err;
        }

        err = clBuildProgram(op->prog, ctxt->num_devices, ctxt->device_ids, opts, NULL, NULL);

        if (err != CL_SUCCESS) {
            char * buf;
            size_t len;

            for (int device_n = 0; device_n < ctxt->num_devices; device_n++) {

                clGetProgramBuildInfo(op->prog, ctxt->device_ids[device_n],
                        CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

                buf = (char *) malloc(len * sizeof(char));

                clGetProgramBuildInfo(op->prog, ctxt->device_ids[device_n],
                        CL_PROGRAM_BUILD_LOG, len, buf, NULL);

                fprintf(stderr, "Error building program \"%s\", reason: %s\n",
                        program_name, buf);

                free(buf);
            }

            return err;
        }

        op->kernel = clCreateKernel(op->prog, kernel_name, &err);

        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to create kernel \"%s\" in program "
                    "\"%s\"\n", kernel_name, program_name);
            clReleaseProgram(op->prog);
            return err;
        }

        op->flags |= INITIALIZED;
    }

    // if op was already initialized, don't need to do anything
    return 0;
}


void cl_get_op_binary(operation_t op_idx, unsigned char * buf, size_t buf_len,
        size_t * write_size) {
    struct __int_cl_context * global_context;
    struct __int_op * op;

    global_context = __cl_get_global_context();
    op = &global_context->ops[op_idx];

    size_t sizes[2];
    cl_int err = clGetProgramInfo(op->prog, CL_PROGRAM_BINARY_SIZES,
            2 * sizeof(size_t), sizes, NULL);

    if (buf_len < sizes[1]) {
        fprintf(stderr, "Buffer length %zu too short for binary size %zu\n",
                buf_len, sizes[1]);
        return;
    }

    unsigned char * bufs[2] = {
        NULL, buf
    };

    *write_size = sizes[1];

    err = clGetProgramInfo(op->prog, CL_PROGRAM_BINARIES,
            2 * sizeof(unsigned char *), bufs, NULL);

    if (err) {
        fprintf(stderr, "Failed to get binaries for op %d\n", op_idx);
    }
}


cl_mem cl_create_buffer(int flags, size_t n_bytes, void * ptr) {
    struct __int_cl_context * global_context;
    cl_int err;

    global_context = __cl_get_global_context();

    cl_mem mem = clCreateBuffer(global_context->context, flags, n_bytes,
            ptr, &err);

    if (err != CL_SUCCESS) {
        const char * perms;
        if (flags & CL_MEM_READ_ONLY) {
            if (flags & CL_MEM_WRITE_ONLY) {
                perms = "read/write";
            }
            else {
                perms = "read";
            }
        }
        else if (flags & CL_MEM_WRITE_ONLY) {
            perms = "write";
        }
        fprintf(stderr, "Unable to create %s buffer of size %lu\n",
                perms, n_bytes);
        return NULL;
    }

    return mem;
}

void cl_read_buffer(cl_mem cl_buf, size_t offset, size_t n_bytes, void * dst) {
    struct __int_cl_context * global_context;
    cl_int err;
    cl_event event;

    global_context = __cl_get_global_context();

    err = clEnqueueReadBuffer(global_context->command_queues[USE_DEVICE],
            cl_buf,
            // true for blocking read
            CL_TRUE,
            offset, n_bytes, dst,
            0, NULL,
            &event);
}

void cl_delete_buffer(cl_mem buf) {
    clReleaseMemObject(buf);
}


void cl_set_param(operation_t op_idx, uint32_t param_idx, size_t arg_size,
        const void * arg) {
    struct __int_cl_context * global_context;
    struct __int_op * op;

    global_context = __cl_get_global_context();
    op = &global_context->ops[op_idx];

    clSetKernelArg(op->kernel, param_idx, arg_size, arg);
}



static void _cl_print_execute_op_err(cl_int err) {
    fprintf(stderr, "failed to enqueue kernel: ");
    switch (err) {
        case CL_INVALID_PROGRAM_EXECUTABLE:
            fprintf(stderr, "no executable has been built in the program "
                    "object for the device associated with the command "
                    "queue\n");
            break;
        case CL_INVALID_COMMAND_QUEUE:
            fprintf(stderr, "the command queue is not valid\n");
            break;
        case CL_INVALID_KERNEL:
            fprintf(stderr, "the kernel object is not valid\n");
            break;
        case CL_INVALID_CONTEXT:
            fprintf(stderr, "the command queue and kernel are not "
                    "associated with the same context\n");
            break;
        case CL_INVALID_KERNEL_ARGS:
            fprintf(stderr, "kernel arguments have not been set\n");
            break;
        case CL_INVALID_WORK_DIMENSION:
            fprintf(stderr, "the dimension is not between 1 and 3\n");
            break;
        case CL_INVALID_GLOBAL_WORK_SIZE:
            fprintf(stderr, "the global work size is NULL or exceeds the "
                    "range supported by the compute device\n");
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            fprintf(stderr, "the local work size is not evenly divisible "
                    "with the global work size or the value specified "
                    "exceeds the range supported by the compute device\n");
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            fprintf(stderr, "the number of work-items specified in any of "
                    "local_work_size[0], ... local_work_size[work_dim - 1] "
                    "is greater than the corresponding values specified by "
                    "CL_DEVICE_MAX_WORK_ITEM_SIZES[0], ..., "
                    "CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1]\n");
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            fprintf(stderr, "the reserved global offset parameter is not "
                    "set to NULL");
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            fprintf(stderr, "the events list is empty (NULL) but the "
                    "number of events is greater than 0; or number of "
                    "events is 0 but the event list is not NULL; or the "
                    "events list contains invalid event objects\n");
            break;
        case CL_OUT_OF_HOST_MEMORY:
            fprintf(stderr, "the host is unable to allocate OpenCL "
                    "resources\n");
            break;
        case CL_OUT_OF_RESOURCES:
            fprintf(stderr, "insufficient resources to execute the "
                    "kernel\n");
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            fprintf(stderr, "there was a failure to allocate memory for "
                    "data store associated with image or buffer objects "
                    "specified as arguments to the kernel\n");
            break;
    }
}


int cl_execute_op(operation_t op_idx, uint32_t n_dims, size_t * global_sizes,
        size_t * local_sizes) {

    struct __int_cl_context * global_context;
    struct __int_op * op;
    cl_int err;
    cl_event event;

    global_context = __cl_get_global_context();
    op = &global_context->ops[op_idx];

    err = clEnqueueNDRangeKernel(global_context->command_queues[USE_DEVICE], op->kernel,
            n_dims, NULL, global_sizes, local_sizes,
            0, NULL,
            &event);

    if (err != CL_SUCCESS) {
        _cl_print_execute_op_err(err);
        return err;
    }

    return 0;
}


void cl_finish() {
    struct __int_cl_context * global_context;

    global_context = __cl_get_global_context();

    clFlush(global_context->command_queues[USE_DEVICE]);
    clFinish(global_context->command_queues[USE_DEVICE]);
}

