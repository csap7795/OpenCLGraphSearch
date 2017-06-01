#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <graph.h>
#include <matrix.h>
#include <queue.h>
#include <CL/cl.h>
#include <bfs_parallel.h>

#include <Test_floyd_warshall.h>
#include <Test_dijkstra.h>
#include <Test_bfs.h>
#include <Test_sssp.h>
#include <Test_topo_order.h>
#include <Test_dijkstra.h>
#include <Test_transpose.h>

#define G_SIZE 512
#define SIZE 1024*1024*128
#define EPV 10

void createGraphData(Graph* graph)
{

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

}
void test_boundaries();
#define VERTICES 800000
int main(int argc, char* argv[])
{
    if(argc != 2)
    {   printf("To few Arguments\n");
        return 0;
    }

    const char* filename = argv[1];

    printf("Processing Graph %s\n",filename);

    //srand(time(NULL));
    Graph* graph = readGraphFromFile(filename);
    unsigned source = 0;

    printf("Diameter of Graph: %u\n",bfs_diameter(graph,source));

	//printf("%u",graph->V);
    if(graph->V <1000)
    {
        cl_float** mat = GraphToMatrix(graph);
        benchmark_floyd_warshall(mat,graph->V,graph->E);
        if(graph->V<=512)
        {
            verify_floyd_warshall_row(mat, graph->V);
            verify_floyd_warshall_column(mat, graph->V);
            verify_floyd_warshall_workgroup(mat, graph->V);
        }
        freeFloatMatrix(mat,graph->V);
    }
    else
    {
        benchmark_bfs(graph,source);
        benchmark_dijkstra(graph,source);
        benchmark_sssp(graph,source);
        benchmark_topo(graph);
        benchmark_transpose(graph);
    }

    freeGraph(graph);

    return 0;
}

