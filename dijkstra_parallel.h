#ifndef DIJKSTRA_PARALLEL_H_INCLUDED
#define DIJKSTRA_PARALLEL_H_INCLUDED

#include <graph.h>

void dijkstra_parallel_gpu(Graph* graph, unsigned source);

float* dijkstra_serial(Graph* graph, unsigned source);

#endif // DIJKSTRA_PARALLEL_H_INCLUDED
