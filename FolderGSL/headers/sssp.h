#ifndef SSSP_H_INCLUDED
#define SSSP_H_INCLUDED


#include <graph.h>
void sssp(Graph* graph,unsigned source,cl_float* out_cost,cl_uint* out_path, unsigned device_num, unsigned long *time, unsigned long *precalc_time);
#endif // SSSP_H_INCLUDED
