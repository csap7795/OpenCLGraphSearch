#ifndef DIKSTRA_PATH_H_INCLUDED
#define DIKSTRA_PATH_H_INCLUDED

#include<graph.h>

void dijkstra_path_preprocess(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVertices, cl_uint* inEdges, cl_uint* offset);
void dijkstra_path(Graph* graph,unsigned source,cl_float* out_cost, cl_uint* out_path, unsigned device_num, unsigned long *time);

#endif // DIKSTRA_PATH_H_INCLUDED
