#ifndef DIJKSTRA_PARALLEL_H_INCLUDED
#define DIJKSTRA_PARALLEL_H_INCLUDED

#include <graph.h>
#include <stdbool.h>

// Calculates the shortest path from source node 'source' to all other nodes in Graph 'graph' in parallel using device 'device num'
// The shortest paths with the respective cost are saved in out_cost and out_path, both arrays of length graph->V
// Both implementations use atomics to syncronize parallel accesses to same neighbors.

void dijkstra_parallel(Graph* graph, unsigned source, unsigned device_num, cl_float* out_cost, cl_uint* out_path, unsigned long *time);

// This version uses Hostpointers to save the time writing to and reading from the buffers.
void dijkstra_parallel_cpu(Graph* graph, unsigned source, unsigned device_num, cl_float* out_cost, cl_uint* out_path, unsigned long *time);
#endif // DIJKSTRA_PARALLEL_H_INCLUDED
