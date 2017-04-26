#ifndef GRAPH_TRANSPOSE_H_INCLUDED
#define GRAPH_TRANSPOSE_H_INCLUDED

#include <graph.h>
#include <stdlib.h>

unsigned long transpose_serial(Graph* graph, Graph* transposed);
unsigned long transpose_parallel(Graph* graph, Graph* transposed, size_t device);


#endif // GRAPH_TRANSPOSE_H_INCLUDED
