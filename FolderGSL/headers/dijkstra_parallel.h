#ifndef DIJKSTRA_PARALLEL_H_INCLUDED
#define DIJKSTRA_PARALLEL_H_INCLUDED

#include <graph.h>
#include <stdbool.h>

// Calculates the shortest path from source node 'source' to all other nodes in Graph 'graph' in parallel using device 'device num'
// The shortest paths with the respective cost are saved in out_cost and out_path, which are pointers to a memory segment which
// is of size (graph->V * sizeof(cl_uint))
void dijkstra_parallel(Graph* graph, unsigned source, unsigned device_num, cl_float* out_cost, cl_uint* out_path, unsigned long *time);

#endif // DIJKSTRA_PARALLEL_H_INCLUDED
