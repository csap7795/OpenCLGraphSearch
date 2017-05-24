#ifndef TEST_DIJKSTRA_H_INCLUDED
#define TEST_DIJKSTRA_H_INCLUDED

#include<stdbool.h>
#include<CL/cl.h>
#include<graph.h>

bool verify_dijkstra(Graph* graph, cl_float *out_cost_parallel, cl_uint* out_path_parallel,unsigned source);
void verify_dijkstra_parallel(Graph* graph, unsigned source);
void bellman_ford_serial(Graph* graph,cl_float* out_cost, cl_uint* out_path, unsigned source);
void dijkstra_serial(Graph* graph, cl_float* cost_array,cl_uint* path_array,unsigned source);
void benchmark_dijkstra(Graph* graph, unsigned source);
unsigned long measure_time_dijkstra(Graph* graph, unsigned source, unsigned device_id);
void dijkstra_serial(Graph* graph, cl_float* cost_array,cl_uint* path_array,unsigned source);
unsigned long measure_time_dijkstra_cpu(Graph* graph, unsigned source);




#endif // TEST_DIJKSTRA_H_INCLUDED
