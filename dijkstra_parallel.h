#ifndef DIJKSTRA_PARALLEL_H_INCLUDED
#define DIJKSTRA_PARALLEL_H_INCLUDED

#include <graph.h>

unsigned long dijkstra_parallel(Graph* graph, unsigned source, unsigned device_num);

//float* dijkstra_serial(Graph* graph, unsigned source);

#endif // DIJKSTRA_PARALLEL_H_INCLUDED
