#ifndef TEST_SSSP_H_INCLUDED
#define TEST_SSSP_H_INCLUDED

#include<stdbool.h>
#include<CL/cl.h>
#include<graph.h>

void benchmark_sssp(Graph* graph, unsigned source);

bool verify_sssp_opt(Graph* graph, cl_float *out_cost_parallel, cl_uint *out_path_parallel,unsigned source);
void verify_sssp_opt_parallel(Graph* graph,unsigned source);
void measure_time_sssp_opt(Graph* graph, unsigned source, unsigned device_id, unsigned long* total_time, unsigned long* precalc_time);

unsigned long measure_time_sssp_normal(Graph* graph, unsigned source, unsigned device_id);
bool verify_sssp_normal(Graph* graph, cl_float *out_cost_parallel, cl_uint *out_path_parallel,unsigned source);
void verify_sssp_normal_parallel(Graph* graph, unsigned source);



#endif // TEST_SSSP_H_INCLUDED
