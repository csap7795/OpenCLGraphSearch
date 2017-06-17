#ifndef TEST_TRANSPOSE_H_INCLUDED
#define TEST_TRANSPOSE_H_INCLUDED

#include <graph.h>

unsigned long measure_time_transpose(Graph* graph, unsigned device_id);
void benchmark_transpose(Graph* graph);
void benchmark_transpose_serial(Graph* graph);
void verify_transpose_parallel(Graph* graph);
bool verify_transpose(Graph* graph, unsigned device);
bool have_same_neighbors(cl_uint *arr1, cl_uint *arr2, unsigned length);

#endif // TEST_TRANSPOSE_H_INCLUDED
