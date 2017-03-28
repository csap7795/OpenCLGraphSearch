#ifndef EDGE_VERTICE_MESSAGE_H_INCLUDED
#define EDGE_VERTICE_MESSAGE_H_INCLUDED

#include <graph.h>
#include <CL/cl.h>
#include <stdbool.h>

void calculateNumEdgesAndSourceVertices(Graph* graph, cl_uint* sourceVertices, cl_uint* numEdges);

void remapMassageBuffer(Graph* graph,cl_uint *messageWriteIndex, cl_uint *numEdgesSorted, cl_uint* offset, cl_uint *oldToNew, cl_uint *messageBufferSize);
void remapMassageBuffer_parallel(Graph* graph,cl_uint *messageWriteIndex, cl_uint *numEdgesSorted, cl_uint* offset, cl_uint *oldToNew, cl_uint *messageBufferSize);
void CalculateWriteIndices(Graph* graph, cl_uint *oldToNew, cl_uint *messageWriteIndex, cl_uint *offset, cl_uint* inEdgesSorted, cl_uint *messageBufferSize);

void messageBufferSort(Graph* graph, cl_uint* inEdgesSorted, cl_uint* oldToNew, cl_uint* offset);
void messageBufferSort_parallel(Graph* graph, cl_uint* inEdges, cl_uint* inEdgesSorted, cl_uint* oldToNew, cl_uint* offset);

void preprocessing(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVerticesSorted, cl_uint* numEdgesSorted, cl_uint* oldToNew, cl_uint* offset,cl_uint* messageBufferSize);

void preprocessing_parallel(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVerticesSorted,cl_uint* numEdgesSorted, cl_uint* oldToNew, cl_uint* offset,cl_uint* messageBufferSize,size_t device_num);
void preprocessing_parallel_cpu(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVerticesSorted,cl_uint* numEdgesSorted, cl_uint* oldToNew, cl_uint* offset,cl_uint* messageBufferSize,size_t device_num);

void sortSourceVertices(Graph* graph, cl_uint* sourceVertices, cl_uint* oldToNew,cl_uint* sourceVerticesSorted);

#endif // EDGE_VERTICE_MESSAGE_H_INCLUDED
