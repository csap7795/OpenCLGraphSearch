#pragma once

#include <CL/cl.h>
#include <stdio.h>
#include <assert.h>
#include <stdarg.h>
#ifndef _WIN32
#include <alloca.h>
#endif


#define MAX_KERNEL_SOURCE 1024*1024*4

#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)

// check __err for ocl success and print message in case of error
#define CLU_ERRCHECK(__err, __message, ...) \
if(__err != CL_SUCCESS) { \
	fprintf(stderr, "OpenCL Assertion failure in %s#%d:\n", __FILE__, __LINE__); \
	fprintf(stderr, "Error code: %s\n", cluErrorString(__err)); \
	fprintf(stderr, __message "\n", ##__VA_ARGS__); \
	exit(-1); \
}

// ------------------------------------------------------------------------------------------------ declarations

/* used for making globalSize a multiple of localSize */
unsigned round_up_globalSize(unsigned globalSize, unsigned localSize);

// initialize opencl device "num" -- devices are numbered sequentially across all platforms
// if supplied, "command_queue" and "context" are filled with an initialized context and command queue on the device
 cl_device_id cluInitDevice(size_t num, cl_context *out_context, cl_command_queue *out_queue);

// get string with basic information about the ocl device "device" with id "id"
const char* cluGetDeviceDescription(const cl_device_id device, unsigned id);

// loads and builds program from "fn" on the supplied context and device, with the options string "options"
// aborts and reports the build log in case of compiler errors
 cl_program cluBuildProgramFromFile(cl_context context, cl_device_id device_id, const char* fn, const char* options);

// sets "num_arg" arguments for kernel "kernel"
// additional arguments need to follow this order: arg0_size, arg0, arg1_size, arg1, ...
 void cluSetKernelArguments(const cl_kernel kernel, const cl_uint num_args, ...);

// return string representation of ocl error code "err"
 const char* cluErrorString(cl_int err);

// return string representation of ocl device type "type"
const char* cluDeviceTypeString(cl_device_type type);


