#ifndef FLOYD_WARHSALL_H_INCLUDED
#define FLOYD_WARHSALL_H_INCLUDED

#include <stdlib.h>
#include <CL/cl.h>
#include <stdbool.h>

//Calculates the Floyd Warshall Algorithm in parallel on global Memory in row major fashion
void parallel_floyd_warshall_global(cl_float** in_matrix, cl_float** out_matrix, cl_uint** out_path, unsigned length, size_t device_num);
//Calculates the Floyd Warshall Algorithm, works on transposed matrices so tha data can be accessed in a column major fashion
void parallel_floyd_warshall_global_gpu(cl_float** in_matrix, cl_float** out_matrix, cl_uint** out_path, unsigned length, size_t device_num);
//Calculates the Floyd Warshall Algorithm on local memory, using tiles
void parallel_floyd_warshall_workgroup(cl_float** matrix, cl_float** out_matrix, cl_uint** out_path, unsigned length, size_t device_num);


#endif // FLOYD_WARHSALL_H_INCLUDED
