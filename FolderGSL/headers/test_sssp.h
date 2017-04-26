#ifndef TEST_SSSP_H_INCLUDED
#define TEST_SSSP_H_INCLUDED

#include<stdbool.h>
#include<CL/cl.h>
#include<graph.h>

bool verify_sssp(Graph* graph, cl_float *out_cost_parallel,unsigned source);
void verify_sssp_parallel(Graph* graph,unsigned source);
unsigned long measure_time_sssp(Graph* graph, unsigned source, unsigned device_id);
void benchmark_sssp(Graph* graph, unsigned source);


#endif // TEST_SSSP_H_INCLUDED
