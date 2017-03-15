#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

Graph* getRandomGraph (int verticeCount){

    srand(time(NULL));

    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->vertices = (unsigned*) malloc(sizeof(unsigned)*verticeCount);
    graph->V = verticeCount;

    unsigned edgeCount = 0;
    // generate random graph : 1.Step : Each Vertice gets at maximum 10 Edges

    for(int i = 0;i<verticeCount;i++)
    {
        graph->vertices[i] = edgeCount;
        edgeCount += rand()%(10);
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



