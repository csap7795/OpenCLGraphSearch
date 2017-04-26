#ifndef GRAPH_H_INCLUDED
#define GRAPH_H_INCLUDED

#include <CL/cl.h>
#include <stdbool.h>

// The Datastructure for saving Graphdata, i.e. 3 Arrays, one saving the amount of neighbors (vertices), one the actual neighbors (edges), and the last for saving the weight of the edge
// Should the Graph use OpenCl types, i.e. cl_uint??
struct Graph
{
    cl_uint* vertices;
    unsigned V;

    cl_uint* edges;
    unsigned E;

    cl_float* weight;

};

typedef struct Graph Graph;


void dfs(Graph* graph, unsigned* components, cl_uint v, unsigned c);
void connectGraph(Graph* graph);
void connectGraphbfs(Graph* graph);
void addEdge(Graph* graph, cl_uint src, cl_uint dest, cl_float weight);

Graph* createUnconnectedGraph();
Graph* matrixToGraph(cl_float** matrix, int length);
Graph* createNegativeCycleGraph(unsigned vertices);
Graph* getEmptyGraph(unsigned vertices, unsigned edges);
Graph* getTreeGraphWeight(int level, int edges);
Graph* getTreeGraph(int level, int edges);
Graph* getRandomGraph (unsigned verticeCount, unsigned edges_per_vertex);
unsigned getNormalDistributedValues(unsigned range);
void freeGraph(Graph* graph);
void printGraph(Graph* graph);
void fillPathMatrix(cl_uint** matrix, unsigned length);
Graph* readGraphFromFile(const char* filename);
void writeGraphToFile(const char *filename, Graph* graph);
bool graph_equal(Graph* g1, Graph* g2);


#endif // GRAPH_H_INCLUDED
