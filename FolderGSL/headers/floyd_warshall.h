#ifndef FLOYD_WARHSALL_H_INCLUDED
#define FLOYD_WARHSALL_H_INCLUDED

#include <stdlib.h>
#include <CL/cl.h>
#include <stdbool.h>

// All 3 Functions calculate APSP on a graph in matrix format, where in_matrix(i,j) denotes the weight of the edge between
// node i and node j. Length is the number of rows/columns of the matrix. The calculated costs are saved in out_matrix,
// The calculated paths in out_path, where out_path(i,j) is a vertex between the path from i to j.

//Calculates the Floyd Warshall Algorithm in parallel on global Memory in row major fashion
void parallel_floyd_warshall_row(cl_float** in_matrix, cl_float** out_matrix, cl_uint** out_path, unsigned length, size_t device_num, unsigned long *time);
//Calculates the Floyd Warshall Algorithm, works on transposed matrices
//so tha data can be accessed in a column major fashion
void parallel_floyd_warshall_column(cl_float** in_matrix, cl_float** out_matrix, cl_uint** out_path, unsigned length, size_t device_num, unsigned long *time);
//Calculates the Floyd Warshall Algorithm on local memory, using tiles.
void parallel_floyd_warshall_workgroup(cl_float** matrix, cl_float** out_matrix, cl_uint** out_path, unsigned length, size_t device_num, unsigned long *time);


#endif // FLOYD_WARHSALL_H_INCLUDED
