#ifndef SSSP_H_INCLUDED
#define SSSP_H_INCLUDED

#include <graph.h>
// Both version calculate the shortest path from source-node 'source', where out_cost and out_path
// are used for returning the costs and paths to other nodes. The size of both arrays is graph->V.
// If desired, the total time can be saved for both algorithms.


// This version makes use of the edge-vertice-message model with both optmization techniques, sorting and layout-remapping
// of the Messagebuffer. If desired the time for manipulating the messagebuffer for the optimization can be
// returned using 'precalc time' pointer.
void sssp_opt(Graph* graph,unsigned source,cl_float* out_cost,cl_uint* out_path, unsigned device_num, unsigned long *time, unsigned long *precalc_time);

// This version makes use of the edge-vertice-message model without using optimization techniques.
void sssp_normal(Graph* graph,unsigned source,cl_float* out_cost, cl_uint* out_path, unsigned device_num, unsigned long *time);


#endif // SSSP_H_INCLUDED
