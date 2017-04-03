#ifndef BFS_PARALLEL_H_INCLUDED
#define BFS_PARALLEL_H_INCLUDED

#include "graph.h"
#include <stdlib.h>
void bfs_parallel_workgroup(Graph* graph, unsigned source, size_t group_size, unsigned device_num);
void bfs_parallel_baseline(Graph* graph, unsigned source, unsigned device_num);

#endif // BFS_PARALLEL_H_INCLUDED
