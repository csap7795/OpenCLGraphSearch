#ifndef GRAPH_TRANSPOSE_H_INCLUDED
#define GRAPH_TRANSPOSE_H_INCLUDED

#include <graph.h>
#include <stdlib.h>

// Transpose a Graph using a serial approach
// returns the transposed graph
// saves execution time if asked
Graph* transpose_serial(Graph* graph, unsigned long *time);

// Transposes a Graph in parallel on device 'device'
// returns the transposed graph
// saves execution time if asked
Graph* transpose_parallel(Graph* graph,size_t device,unsigned long *time);


#endif // GRAPH_TRANSPOSE_H_INCLUDED
