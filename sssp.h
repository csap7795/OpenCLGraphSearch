#ifndef SSSP_H_INCLUDED
#define SSSP_H_INCLUDED


#include <graph.h>

unsigned long sssp(Graph* graph,cl_float* out_cost ,unsigned source,unsigned device_num );



#endif // SSSP_H_INCLUDED
