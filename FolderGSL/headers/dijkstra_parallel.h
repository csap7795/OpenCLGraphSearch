#ifndef DIJKSTRA_PARALLEL_H_INCLUDED
#define DIJKSTRA_PARALLEL_H_INCLUDED

#include <graph.h>
#include <stdbool.h>

void dijkstra_parallel(Graph* graph, unsigned source, unsigned device_num, cl_float* out_cost, cl_uint* out_path, unsigned long *time);//bool* check_cycles, bool* negative_cycles);

#endif // DIJKSTRA_PARALLEL_H_INCLUDED
