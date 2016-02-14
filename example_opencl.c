// Copyright (c) 2016 Dominik Zeromski <dzeromsk@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>

#define DATA_SIZE (64)

const char source[] = {
    "__kernel void sum(                                                     \n"
    "   __global int* input,                                                \n"
    "   __global int* output)                                               \n"
    "{                                                                      \n"
    "   int i = get_global_id(0);                                           \n"
    "   output[i] = input[i] + input[i];                                    \n"
    "}                                                                      \n"

};

int main(int argc, char **argv) {
  int err = 0; // error code returned from api calls

  int data[DATA_SIZE];    // original data set given to device
  int results[DATA_SIZE]; // results returned from device
  unsigned int correct;   // number of correct results returned

  size_t global; // global domain size for our calculation
  size_t local;  // local domain size for our calculation

  cl_platform_id platform;
  cl_device_id device_id;    // compute device id
  cl_context context;        // compute context
  cl_command_queue commands; // compute command queue
  cl_program program;        // compute program
  cl_kernel kernel;          // compute kernel

  cl_mem input;  // device memory used for the input array
  cl_mem output; // device memory used for the output array

  int i = 0;
  unsigned int count = DATA_SIZE;
  for (i = 0; i < count; i++)
    data[i] = i;

  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to list platforms!\n");
    return EXIT_FAILURE;
  }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to create a device group!\n");
    return EXIT_FAILURE;
  }

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context) {
    printf("Error: Failed to create a compute context!\n");
    return EXIT_FAILURE;
  }

  commands = clCreateCommandQueue(context, device_id, 0, &err);
  if (!commands) {
    printf("Error: Failed to create a command commands!\n");
    return EXIT_FAILURE;
  }

  program =
      clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  if (!program) {
    printf("Error: Failed to create compute program!\n");
    return EXIT_FAILURE;
  }

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];

    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                          sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    return EXIT_FAILURE;
  }

  kernel = clCreateKernel(program, "sum", &err);
  if (!kernel || err != CL_SUCCESS) {
    printf("Error: Failed to create compute kernel!\n");
    return EXIT_FAILURE;
  }

  input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * count, NULL,
                         NULL);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * count, NULL,
                          NULL);
  if (!input || !output) {
    printf("Error: Failed to allocate device memory!\n");
    return EXIT_FAILURE;
  }

  err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(int) * count,
                             data, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to write to source array!\n");
    return EXIT_FAILURE;
  }

  err = 0;
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    return EXIT_FAILURE;
  }

  global = count;
  err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0,
                               NULL, NULL);
  if (err) {
    printf("Error: Failed to execute kernel!\n");
    return EXIT_FAILURE;
  }

  clFinish(commands);

  err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(int) * count,
                            results, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to read output array! %d\n", err);
    return EXIT_FAILURE;
  }

  correct = 0;
  for (i = 0; i < count; i++) {
    if (results[i] == data[i] + data[i])
      correct++;
  }

  fprintf(stderr, "Computed '%d/%d' correct values!\n", correct, count);

  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return 0;
}
