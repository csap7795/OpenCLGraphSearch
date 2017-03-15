#ifndef BFS_PARALLEL_H_INCLUDED
#define BFS_PARALLEL_H_INCLUDED

#include "graph.h"
#include <stdlib.h>

void bfs_parallel_gpu_workgroup(Graph* graph, unsigned source,size_t group_size);
void bfs_parallel_gpu_baseline(Graph* graph, unsigned source);

#endif // BFS_PARALLEL_H_INCLUDED
