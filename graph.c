#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>


Graph* matrixToGraph(cl_float** matrix, int length)
{
    //Count Edges
    unsigned edgecount = 0;
    for(int i = 0; i< length;i++)
        for(int j = 0; j<length;j++)
            if(matrix[i][j] != CL_FLT_MAX && matrix[i][j] != 0)
                edgecount++;

    Graph* graph = getEmptyGraph(length,edgecount);

    //fill the graph
    edgecount = 0;
    graph->vertices[0] = 0;
    for(int i = 0; i<length-1;i++)
    {
        for(int j = 0; j<length;j++)
            if(matrix[i][j] != CL_FLT_MAX && matrix[i][j] != 0)
            {
                graph->edges[edgecount] = j;
                graph->weight[edgecount] = matrix[i][j];
                graph->vertices[i+1] = ++edgecount;

            }
    }
    // for the last row, only calculate the edges
    for(int j = 0; j<length;j++)
    {
        if(matrix[length-1][j] != CL_FLT_MAX)
        {
            graph->edges[edgecount] = j;
            graph->weight[edgecount] = matrix[length-1][j];
            edgecount++;
        }
    }

    return graph;

}

Graph* createNegativeCycleGraph(unsigned vertices)
{
    Graph* graph = getEmptyGraph(vertices,vertices);
    for(int i = 0; i<graph->V;i++)
    {
        graph->vertices[i] = i;
        graph->edges[i] = (i+1) % graph->V;
        graph->weight[i] = -1.0f;
    }
    return graph;
}

Graph* getEmptyGraph(unsigned vertices, unsigned edges)
{
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->E = edges;
    graph->V = vertices;
    graph->edges = (unsigned*)malloc(graph->E * sizeof(unsigned));
    graph->vertices = (unsigned*)malloc(graph->V * sizeof(unsigned));
    graph->weight = (float*) malloc(graph->E * sizeof(float));
    return graph;
}

Graph* getTreeGraphWeight(int level, int edges)
{
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = 0;

    for(int i = 0 ; i<level;i++)
    {
        graph->V += (unsigned)pow((float)edges,i);
    }

    graph->vertices = (unsigned*) malloc(sizeof(unsigned)*(graph->V));
    graph->E = graph->V-1;
    graph->edges = (unsigned*)malloc(sizeof(unsigned)*graph->E);
    graph->weight = (float*)malloc(sizeof(float)*graph->E);


    for(int i = 0; i<(graph->V)/edges;i++)
    {
       graph->vertices[i] = edges*i;
       graph->edges[i] = i+1;
       graph->weight[i] = 1.0f;
    }

    for(int i = (graph->V)/edges;i<graph->V;i++)
    {
        graph->vertices[i] = graph->E;
        if(i<graph->E){
            graph->edges[i] = i+1;
            graph->weight[i] = 1.0f;
        }
    }
    // Last Entry is number of edges
    //graph->vertices[graph->V] = graph->E;

    return graph;
}

Graph* getTreeGraph(int level, int edges)
{
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = 0;

    for(int i = 0 ; i<level;i++)
    {
        graph->V += (unsigned)pow((float)edges,i);
    }

    graph->vertices = (unsigned*) malloc(sizeof(unsigned)*(graph->V));
    graph->E = graph->V-1;
    graph->edges = (unsigned*)malloc(sizeof(unsigned)*graph->E);
    graph->weight = NULL;


    for(int i = 0; i<(graph->V)/edges;i++)
    {
       graph->vertices[i] = edges*i;
       graph->edges[i] = i+1;
    }

    for(int i = (graph->V)/edges;i<graph->V;i++)
    {
        graph->vertices[i] = graph->E;
        if(i<graph->E)
            graph->edges[i] = i+1;
    }
    // Last Entry is number of edges
    //graph->vertices[graph->V] = graph->E;

    return graph;
}

Graph* getRandomGraph (int verticeCount, int edges_per_vertex){

    srand(time(NULL));

    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->vertices = (unsigned*) malloc(sizeof(unsigned)*verticeCount);
    graph->V = verticeCount;

    unsigned edgeCount = 0;
    // generate random graph : 1.Step : Each Vertice gets at maximum 10 Edges

    for(int i = 0;i<verticeCount;i++)
    {
        graph->vertices[i] = edgeCount;
        edgeCount += (rand()%(edges_per_vertex)+1);
    }


    // create date for Edges and Weight, notice that two nodes can be connected by different edges
    graph->E = edgeCount;
    graph->edges = (unsigned*) malloc(sizeof(unsigned)*edgeCount);
    graph->weight = (float*) malloc(sizeof(float)*edgeCount);

    // 2.Step : Each Edge gets a Weight-Value between 0 and 1
    unsigned i = 0;
    for(i = 0; i<verticeCount-1;i++){
        for(int j = graph->vertices[i]; j<graph->vertices[i+1];j++)
        {
            int node = rand()%verticeCount;
            if(node != i){
                graph->edges[j] = node;
                graph->weight[j] = (float)((double)rand()/(double)RAND_MAX);
            }
            else
                j--;
        }
    }

    for(int j = graph->vertices[i];j<edgeCount;j++)
    {
        unsigned node = (unsigned)rand()%verticeCount;
            if(node != i){
                graph->edges[j] = node;
                graph->weight[j] = (float)((double)rand()/(double)RAND_MAX);
            }
            else
                j--;
    }

    return graph;

}

//Frees allocated Space for Graph Data
void freeGraph(Graph* graph)
{
    free(graph->vertices);
    free(graph->edges);
    free(graph->weight);
    free (graph);
}

//prints Graph to Console
void printGraph(Graph* graph)
{
    printf("VERTICES[%u]:\t",graph->V);

    for(int i = 0; i<graph->V; i++)
        printf("%u\t",graph->vertices[i]);

    printf("\n");

    printf("EDGES[%u]:\t",graph->E);

    for(int i = 0; i<graph->E; i++)
        printf("%u\t",graph->edges[i]);

    printf("\n");

    if(graph->weight != NULL)
    {
        printf("WEIGHT[%u]:\t",graph->E);

        for(int i = 0; i<graph->E; i++)
            printf("%.1f\t",graph->weight[i]);

        printf("\n");
    }

}



