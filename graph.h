#ifndef GRAPH_H_INCLUDED
#define GRAPH_H_INCLUDED

#include <CL/cl.h>

// The Datastructure for saving Graphdata, i.e. 3 Arrays, one saving the amount of neighbors (vertices), one the actual neighbors (edges), and the last for saving the weight of the edge
// Should the Graph use OpenCl types, i.e. cl_uint??
struct Graph
{
    unsigned* vertices;
    unsigned V;

    unsigned* edges;
    unsigned E;

    float* weight;

};

typedef struct Graph Graph;


Graph* matrixToGraph(cl_float** matrix, int length);
//cl_float** getTestMatrix(unsigned vertices);
Graph* createNegativeCycleGraph(unsigned vertices);
//cl_float** getAdjMatrix(unsigned vertices,unsigned edges);
Graph* getEmptyGraph(unsigned vertices, unsigned edges);
Graph* getTreeGraphWeight(int level, int edges);
Graph* getTreeGraph(int level, int edges);
Graph* getRandomGraph (int verticeCount, int edges_per_vertex);
void freeGraph(Graph* graph);
void printGraph(Graph* graph);
void fillPathMatrix(cl_uint** matrix, unsigned length);

#endif // GRAPH_H_INCLUDED
