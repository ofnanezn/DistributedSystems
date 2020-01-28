#include <stdio.h>
#include <stdlib.h> 
#include <CL/cl.h>
 
#define MAX_SOURCE_SIZE (0x100000)
 
void printMatrix(int *arr, int matrix_size) {
  int i, j;
  for (i = 0 ; i < matrix_size ; ++i ) {
    for (j = 0 ; j < matrix_size ; ++j ) {
        printf("%d ", arr[i*matrix_size+j]);
    }
    printf("\n" );
  }
  printf("\n" );
}

void checkResult(int *arr, int expected_value, int size) {
    int error = 0;
    for(unsigned int i = 0;  i < size; i++){
        arr[i] == expected_value ? error += 0 : error += 1;        
    }
    printf("%d errors were found\n", error );
}

int main(void) {
    // Create the two input vectors
    int i;
    const int MATRIX_SIZE = 2048;
    int *A = (int*)malloc(sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);
    int *B = (int*)malloc(sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);
    int *C = (int*)malloc(sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);


    for(i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++) {
        A[i] = i % MATRIX_SIZE;
        B[i] = MATRIX_SIZE - (i/MATRIX_SIZE) - 1;        
    }

    //printf("-----------------Matrix A-----------------\n");
    //printMatrix(A, MATRIX_SIZE);

    //printf("-----------------Matrix B-----------------\n");
    //printMatrix(B, MATRIX_SIZE);
 
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("vector_mul_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
 
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1,&device_id, &ret_num_devices);
 
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE*MATRIX_SIZE * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE*MATRIX_SIZE * sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, MATRIX_SIZE*MATRIX_SIZE * sizeof(int), NULL, &ret);
 
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, MATRIX_SIZE*MATRIX_SIZE * sizeof(int), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, MATRIX_SIZE*MATRIX_SIZE * sizeof(int), B, 0, NULL, NULL);
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_mul", &ret);
 
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &MATRIX_SIZE);
 
    // Execute the OpenCL kernel on the list
    size_t global_item_size = MATRIX_SIZE*MATRIX_SIZE; 
    size_t local_item_size = 32; 
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
 
    // Read the memory buffer C on the device to the local variable C
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), C, 0, NULL, NULL);
 
    // Display the result to the screen
    // printf("-----------------Matrix C-----------------\n");
    // for(unsigned int i = 0; i < 2*MATRIX_SIZE; i++ ){
    //     printf("C[%d]: %d + %d = %d\n",i,A[i], B[i], C[i] );
    // }

    //expected value for 128x128 matrix => 341376
    //expected value for 1024x1024 matrix => 178433024
    //expected value for 2048x2048 matrix => 1429559296


    checkResult(C, 1429559296, MATRIX_SIZE);
 
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    return 0;
}