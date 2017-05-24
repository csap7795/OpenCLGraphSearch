#ifndef TEST_BFS_H_INCLUDED
#define TEST_BFS_H_INCLUDED

#include<stdbool.h>
#include<CL/cl.h>
#include<graph.h>

void verify_bfs_workgroup(Graph* graph, unsigned source);
void verify_bfs_baseline(Graph* graph, unsigned source);
bool verify_bfs(Graph* graph, cl_uint *out_cost_parallel, cl_uint* out_path_parallel,unsigned source);
void benchmark_bfs(Graph* graph, unsigned source);
void bfs_serial(Graph* graph, cl_uint *cost, cl_uint *path, unsigned source);
unsigned long measure_time_bfs_baseline(Graph* graph, unsigned source, unsigned device_id);
unsigned long measure_time_bfs_workgroup(Graph* graph, unsigned source, unsigned device_id);

#endif // TEST_BFS_H_INCLUDED
