#ifndef SCAN_H_INCLUDED
#define SCAN_H_INCLUDED

#include <stdio.h>
#include <CL/cl.h>

typedef unsigned uint;

//static void build_kernel(size_t device_num);


void scan_parallel(const cl_uint *input, cl_uint* output, uint length, uint device);
void sum_scan(const cl_uint *input, cl_uint* output, uint length, uint device);



unsigned long scan(const cl_uint *input, cl_uint* output, uint length, uint device);

unsigned long scan_serial(const uint *input, uint *output, int length);

#endif // SCAN_H_INCLUDED
