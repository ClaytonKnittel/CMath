
#include <OpenCL/opencl.h>

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include <tester.h>

#define SIZE 16384


//OpenCL kernel which is run for every work item created.
const char *saxpy_kernel =
"__kernel                                   \n"
"void saxpy_kernel(float alpha,     \n"
"                  __global float *A,       \n"
"                  __global float *B,       \n"
"                  __global float *C)       \n"
"{                                          \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    C[index] = alpha* A[index] + B[index]; \n"
"}                                          \n";



static void print_device_info(cl_device_id id) {
    char name[128];
    char vendor[128];
    char version[128];

    clGetDeviceInfo(id, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(id, CL_DEVICE_VENDOR, 128, vendor, NULL);
    clGetDeviceInfo(id, CL_DRIVER_VERSION, 128, version, NULL);
    printf("%s : %s (%s)\n", vendor, name, version);
}

int test() {

    cl_platform_id * platforms = NULL;
    cl_uint     num_platforms;
    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id *)
        malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

    //Get the devices list and choose the device you want to run on
    cl_device_id     *device_list = NULL;
    cl_uint           num_devices;

    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);
    device_list = (cl_device_id *)
        malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

    printf("Num devices: %d\n", num_devices);

    for (int i = 0; i < num_devices; i++) {
        print_device_info(device_list[i]);
    }

    // Create one OpenCL context for each device in the platform
    cl_context context;
    context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

    // Create memory buffers on the device for each vector
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,  SIZE * sizeof(float), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,  SIZE * sizeof(float), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(float), NULL, &clStatus);

    float alpha = 2.f;
    float * a = (float*) malloc(SIZE * sizeof(float));
    float * b = (float*) malloc(SIZE * sizeof(float));
    float * c = (float*) malloc(SIZE * sizeof(float));

    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Copy the Buffer A and B to the device
    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, SIZE * sizeof(float), a, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, SIZE * sizeof(float), b, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,(const char **)&saxpy_kernel, NULL, &clStatus);

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &clStatus);

    // Set the arguments of the kernel
    clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);

    // Execute the OpenCL kernel on the list
    size_t global_size = SIZE; // Process the entire lists
    size_t local_size = 64;           // Process one item at a time
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    // Read the cl memory C_clmem on device to the host variable C
    clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, SIZE * sizeof(float), c, 0, NULL, NULL);

    // Clean up and wait for all the comands to complete.
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);

    // Display the result to the screen
    //for(int i = 0; i < SIZE; i++)
    //    printf("%f * %f + %f = %f\n", alpha, a[i], b[i], c[i]);

    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseMemObject(C_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(a);
    free(b);
    free(c);
    free(platforms);
    free(device_list);


    return 0;
}

