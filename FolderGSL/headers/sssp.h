#ifndef SSSP_H_INCLUDED
#define SSSP_H_INCLUDED


#include <graph.h>

void sssp(Graph* graph,unsigned source,cl_float* out_cost, unsigned device_num, unsigned long *time);



#endif // SSSP_H_INCLUDED
