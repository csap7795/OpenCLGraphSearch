#ifndef TOPO_SORT_H_INCLUDED
#define TOPO_SORT_H_INCLUDED
#include <graph.h>

// Calculates a topological order of Graph graph. If it exists each Vertex is assingned an
// unsigned integer value referring to it's position in the order ( Vertices with indegree 0 get 0).
// Thus out_order_parallel is a array of size graph->V.

// This algorithm makes use of the edge-vertice-message model with both optmization techniques, sorting and layout-remapping
// of the Messagebuffer. If desired the time for manipulating the messagebuffer for the optimization can be
// returned using 'precalc_time' pointer.
void topological_order_opt(Graph* graph, cl_uint* out_order_parallel,unsigned device_num, unsigned long *time,unsigned long *precalc_time);

// This version makes use of the edge-vertice-message model without using optimization techniques.
void topological_order_normal(Graph* graph, cl_uint* out_order_parallel,unsigned device_num, unsigned long *time);

#endif // TOPO_SORT_H_INCLUDED
