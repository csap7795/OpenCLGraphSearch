#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <graph.h>
#include <matrix.h>
#include <cl_utils.h>
#include <CL/cl.h>

#include <test_floyd_warshall.h>
#include <test_dijkstra.h>
#include <test_bfs.h>
#include <test_sssp.h>
#include <test_topo_order.h>
#include <test_dijkstra.h>


#define G_SIZE 512
#define SIZE 1024*1024*128
#define EPV 10

void createGraphData(unsigned vertice_count, unsigned epv)
{
    Graph* graph = getRandomGraph(vertice_count,epv);
    connectGraphbfs(graph);
    char vertice_char = ' ';
    char edge_char = ' ';

    float vertices = (float)graph->V;
    if(vertices >= 1000)
    {
        vertices /= 1000;
        vertice_char = 'k';
        if(vertices > 1000)
        {
            vertices/=1000;
            vertice_char = 'm';
        }

    }

    float edges = (float)graph->E;
    if(edges >= 1000)
    {
        edges /= 1000;
        edge_char = 'k';
        if(edges > 1000)
        {
            edges/=1000;
            edge_char = 'm';
        }

    }

    char tmp[1024];
    sprintf(tmp, "Graph/%.1f%cV-%.1f%cE.g",vertices,vertice_char,edges,edge_char);
    writeGraphToFile(tmp,graph);
    free(graph);
}


int main(int argc, char* argv[])
{

    Graph* graph = getSemaphoreGraph(10000);
    verify_dijkstra_parallel(graph,0);


    /*if(argc != 2)
    {   printf("To few Arguments\n");
        return 0;
    }

    const char* filename = argv[1];

    srand(time(NULL));
    Graph* graph = readGraphFromFile(filename);
    unsigned source = 0;
	printf("%u",graph->V);
    if(graph->V <=0)
    {
        cl_float** mat = GraphToMatrix(graph);
        benchmark_floyd_warshall(mat,graph->V,graph->E);
        freeFloatMatrix(mat,graph->V);
    }

    else
    {
        //benchmark_bfs(graph,source);
        //benchmark_dijkstra(graph,source);
        //verify_sssp_parallel(graph,source);
        verify_dijkstra_parallel(graph,source);
        //benchmark_sssp(graph,source);
        //benchmark_transpose(graph);
        //benchmark_topo(graph);
    }

    freeGraph(graph);*/


    return 0;
}


