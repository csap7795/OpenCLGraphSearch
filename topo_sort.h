#ifndef TOPO_SORT_H_INCLUDED
#define TOPO_SORT_H_INCLUDED
#include <graph.h>


void topological_order(Graph* graph, cl_uint* out_order_parallel,unsigned device_num );

#endif // TOPO_SORT_H_INCLUDED
