#ifndef GRAPH_H_INCLUDED
#define GRAPH_H_INCLUDED

#include <CL/cl.h>
#include <stdbool.h>

// The Datastructure for saving Graphdata, i.e. 3 Arrays, one saving the amount of neighbors (vertices),
// one the actual neighbors (edges), and the last for saving the weight of the edge

struct Graph
{
    cl_uint* vertices;
    unsigned V;

    cl_uint* edges;
    unsigned E;

    cl_float* weight;

};

typedef struct Graph Graph;

// Functions to connect a Graph.

// This function connects a graph using a depth-first-search
// As the depth-first-search is recursive, use it only on small graphs
void connectGraph(Graph* graph);

// The depth first search, assigning vertex v  a component with id c
// If vertex v is in component c, components[v] == c.
void dfs(Graph* graph, unsigned* components, cl_uint v, unsigned c);

// For connecting a graph, an edge will be added to the Graph between each component
void addEdge(Graph* graph, cl_uint src, cl_uint dest, cl_float weight);

// Connect a graph using a breadth first seach
void connectGraphbfs(Graph* graph);

// addEdges adds length edges with source vertex src[edge_id] and destination vertex[edge_id] and weight
// weight[id] to the graph, is used by connectGraphbfs
void addEdges(Graph* graph, cl_uint* src, cl_uint* dest, unsigned length, cl_float* weight);

// Function for creating Graphs

// Create a Graph which is surely not connected
Graph* createUnconnectedGraph();

// Creates a Graph where almost every node is connected to the last one
// -> This graph can be used forcing a RaceCondition if the atomic operations don*t work
Graph* getSemaphoreGraph(int verticeCount);

// Creates a graph which is equal to the one represented by a matrix
Graph* matrixToGraph(cl_float** matrix, int length);

// Creates a graph which has a negative cycle, can be used to show that bellman ford algorithm works
Graph* createNegativeCycleGraph(unsigned vertices);

// Creates and Empty Graph which has vertices Vertices and edges Edges
Graph* getEmptyGraph(unsigned vertices, unsigned edges);


// Creates a graph in Tree form which has level levels and edges^currentlevel nodes per stage
Graph* getTreeGraph(int level, int edges);

// Creates a graph in Tree form which has level levels and edges^currentlevel nodes per stage
// also adds weights to the edges
Graph* getTreeGraphWeight(int level, int edges);

// Generates a Random Graph in Tree form, where edgeperVertex denotes the maximum amount of edges per vertex
Graph* getRandomTreeGraph(int level, int edges, unsigned edgeperVertex);

// Generates a randomly shaped graph, each vertex has a random amount of neighbors
// between 0 and edges_per_vertex, distributed by a normal distribution
Graph* getRandomGraph (unsigned verticeCount, unsigned edges_per_vertex);
unsigned getNormalDistributedValues(unsigned range);

// Additional Functions for Graphs

// Frees the allocated space of a graph
void freeGraph(Graph* graph);

// prints a graph to the console
void printGraph(Graph* graph);

// Reads in the graph saved in the file pointed by filename
Graph* readGraphFromFile(const char* filename);

// Writes the graph to the file pointed by filename
void writeGraphToFile(const char *filename, Graph* graph);

// Checks if to graphs contain the same values in their arrays
bool graph_equal(Graph* g1, Graph* g2);

// Generates length different random neighbors between start_node and num_edges
// The numbers are saved in edges
void assignRandomNumbersNotTheSame(cl_uint *edges, cl_uint length, unsigned start_node, unsigned num_edges);


Graph* createGraphFromFile(const char* filename);

Graph* createAcyclicFromGraph(Graph* graph);
bool checkacyclic(Graph* graph);
Graph* removeCycles(Graph* graph);
Graph* readInTextData(const char* filename);


#endif // GRAPH_H_INCLUDED
