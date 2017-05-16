#ifndef DIKSTRA_PATH_H_INCLUDED
#define DIKSTRA_PATH_H_INCLUDED

#include<graph.h>


// Function which is calculating the shortest path from sourcenode source
// to all other nodes in Graph graph using the edge-vertex-message modell.
// Out_cost and out_path are used to save the results, they both have length graph->V.
// The algorithm is executed on device with number device_num and saves the time for the execution behind time pointer if requested
void dijkstra_path(Graph* graph,unsigned source,cl_float* out_cost, cl_uint* out_path, unsigned device_num, unsigned long *time);

// Function which is used by dijkstra_path, to calculate all necessary data for the edge-vertex-message modell
void dijkstra_path_preprocess(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVertices, cl_uint* inEdges, cl_uint* offset);


#endif // DIKSTRA_PATH_H_INCLUDED
