#ifndef DIJKSTRA_PARALLEL_H_INCLUDED
#define DIJKSTRA_PARALLEL_H_INCLUDED

#include <graph.h>
#include <stdbool.h>

unsigned long dijkstra_parallel(Graph* graph, unsigned source, unsigned device_num, cl_float* out_cost, cl_uint* out_path, bool* check_cycles, bool* negative_cycles);
//float* dijkstra_serial(Graph* graph, unsigned source);

#endif // DIJKSTRA_PARALLEL_H_INCLUDED
