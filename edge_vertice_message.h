#ifndef EDGE_VERTICE_MESSAGE_H_INCLUDED
#define EDGE_VERTICE_MESSAGE_H_INCLUDED

#include <graph.h>
#include <CL/cl.h>
#include <stdbool.h>
void messageBufferSort(Graph* graph, cl_uint* inEdges, cl_uint* inEdgesSorted, cl_uint* oldToNew, cl_uint* offset,unsigned* messageBufferSize, bool enable);
void remapMassageBuffer(Graph* graph,cl_uint *messageWriteIndex, cl_uint *numEdgesSorted, cl_uint* offset, cl_uint *oldToNew, cl_uint *messageBufferSize, bool enable);
void preprocessing(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVertices,cl_uint* sourceVerticesSorted, cl_uint* numEdges, cl_uint* numEdgesSorted, cl_uint* oldToNew, cl_uint* offset,cl_uint* messageBufferSize, bool enable);
void sssp(Graph* graph, unsigned source, bool enable);

#endif // EDGE_VERTICE_MESSAGE_H_INCLUDED
