#ifndef EDGE_VERTICE_MESSAGE_H_INCLUDED
#define EDGE_VERTICE_MESSAGE_H_INCLUDED

#include <graph.h>
#include <CL/cl.h>

#define GROUP_NUM 32
#define BUCKET_NUM 1000

int preprocessing_parallel(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVerticesSorted,cl_uint* numEdgesSorted, cl_uint* oldToNew, cl_uint* newToOld, cl_uint* offset,cl_uint* messageBufferSize,cl_device_type type);

// Preprocesses all necessary data for using this framework on the GPU
void preprocessing_parallel_gpu(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVerticesSorted,cl_uint* numEdgesSorted, cl_uint* oldToNew,cl_uint* newToOld, cl_uint* offset,cl_uint* messageBufferSize,size_t device_num);

// Preprocess all necessary data on the cpu, actually means the datalayout of the messageBuffer is not changed, as it's only good for the GPU
void preprocessing_parallel_cpu(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVerticesSorted,cl_uint* numEdgesSorted, cl_uint* oldToNew, cl_uint* newToOld, cl_uint* offset,cl_uint* messageBufferSize,size_t device_num);

// Calculates the amount of incoming Edges for each vertex and saves the source Vertex of every edge in 'sourceVertices'
void calculateNumEdgesAndSourceVertices(Graph* graph, cl_uint* sourceVertices, cl_uint* numEdges);

// Sorts the inEdges array, an array saving the amount of incoming edges for each vertice for better perfomance due to better workload balance
void messageBufferSort_parallel(Graph* graph, cl_uint* inEdges, cl_uint* inEdgesSorted, cl_uint* oldToNew,cl_uint* newToOld, cl_uint* offset);

// remaps the messagebuffer, i.e. packs it in a new datalayout, which makes coalescing access possible
void remapMassageBuffer_parallel(Graph* graph,cl_uint *messageWriteIndex, cl_uint *numEdgesSorted, cl_uint* offset, cl_uint *oldToNew, cl_uint *messageBufferSize);

// The next 3 Functions calculate the indices indicating the write positions for the messageBuffer, if the messageBuffer was remapped
void parallelCalculationWriteIndices(Graph* graph, cl_uint* oldToNew, cl_uint* offset, cl_uint* messageWriteIndex, size_t buckets, cl_uint messageBufferSize);
void serialCalculationofWriteIndices(Graph* graph, cl_uint* oldToNew, cl_uint* offset, cl_uint* messageWriteIndex);
void CalculateWriteIndices(Graph* graph, cl_uint *oldToNew, cl_uint *messageWriteIndex, cl_uint *offset, cl_uint* inEdgesSorted, cl_uint *messageBufferSize);

// This function sorts the Source Vertices, i.e. saves the sourcevertices of each edge under their new alias
void sortSourceVertices(Graph* graph, cl_uint* sourceVertices, cl_uint* oldToNew,cl_uint* sourceVerticesSorted);

// Calculates all necessary data for the edge-vertice-message model serial without making optimizations like
// sorting the messagebuffer or remap it's data layout
void serial_without_optimization_preprocess(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVertices, cl_uint* inEdges, cl_uint* offset, cl_uint *messageBufferSize);

#endif // EDGE_VERTICE_MESSAGE_H_INCLUDED
