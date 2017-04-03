#ifndef FLOYD_WARHSALL_H_INCLUDED
#define FLOYD_WARHSALL_H_INCLUDED

#include <stdlib.h>
#include <CL/cl.h>
#include <stdbool.h>

void parallel_floyd_warshall(cl_float** matrix, unsigned length, size_t device_num);

void serial_floyd_warshall(cl_float** matrix, unsigned length);

cl_float** createMatrix(cl_float** matrix, unsigned length);

bool verify(cl_float** mat1, cl_float** mat2, unsigned length);

#endif // FLOYD_WARHSALL_H_INCLUDED
