#ifndef BFS_PARALLEL_H_INCLUDED
#define BFS_PARALLEL_H_INCLUDED

#include <graph.h>

// Functions to calculate the hops from a source node to all other nodes via a breadth first search.
// out_cost and out_path are pointers for the calculated result which point to a memorysegment of size 'graph->V * sizeof(cl_uint)'
// if time is not a NULL pointer the time for calculating the algorithm is saved behind the pointer
// graph, source and device_num are the parameters, graph is a pointer to a Graph structure, source denotes the source node from
// which all hops are calculated. Device_num denotes the number of the device to work on. Devices are enumerated through all platforms.

// This approach works on local memory
void bfs_parallel_workgroup(Graph* graph, cl_uint* out_cost, cl_uint* out_path,unsigned source, unsigned device_num, unsigned long *time);

// This approach works on global memory
void bfs_parallel_baseline(Graph* graph, cl_uint* out_cost, cl_uint* out_path, unsigned source, unsigned device_num, unsigned long *time);

// Gives a statistic about a graph for each bfs iteration, containing the Number of Vertices in the Frontier,
// the number of all Edges traversed and the number of Unique Edges, i.e. how many different
// edges were traversed. Can be used to show unnecessary work done in low-diameter graphs.
void bfs_logical_frontier_plot(Graph* graph,unsigned source, unsigned device_num);



#endif // BFS_PARALLEL_H_INCLUDED
