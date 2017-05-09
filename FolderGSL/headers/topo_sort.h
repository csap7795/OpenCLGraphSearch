#ifndef TOPO_SORT_H_INCLUDED
#define TOPO_SORT_H_INCLUDED
#include <graph.h>

void topological_order(Graph* graph, cl_uint* out_order_parallel,unsigned device_num,unsigned long *time);
void topo_sort_preprocess(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVertices, cl_uint* inEdges, cl_uint* offset, cl_uint *messageBufferSize);


#endif // TOPO_SORT_H_INCLUDED
