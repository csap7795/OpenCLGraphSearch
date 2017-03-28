#include <stdio.h>
#include <stdlib.h>
#include <graph.h>
#include <assert.h>
#include <dijkstra_parallel.h>
#include <sssp.h>
#include <time.h>
#include <cl_utils.h>

#define TEST_ITERATIONS 5

float test_sssp(Graph* graph,unsigned device_id)
{
    srand(time(NULL));

    unsigned long average_time = 0;
    for(int i = 0; i<TEST_ITERATIONS;i++)
    {
        unsigned source = rand() % graph->V;
        average_time += sssp(graph,source,device_id);
    }
    float divisor = TEST_ITERATIONS;
    return average_time/divisor;
}

float test_dijkstra(Graph* graph,unsigned device_id)
{
    srand(time(NULL));

    unsigned long average_time = 0;
    for(int i = 0; i<TEST_ITERATIONS;i++)
    {
        unsigned source = rand() % graph->V;
        average_time += dijkstra_parallel(graph,source,device_id);
    }
    float divisor = TEST_ITERATIONS;
    return average_time/divisor;
}


int main()
{
    unsigned device_count = cluCountDevices();

    unsigned edges = 10;

    for(int k = 100000;k<500000;k+=100000)
    {
        Graph* graph = getRandomGraph(k,edges);

        for(unsigned i = 0; i<device_count; i++)
        {
            printf("Optimized\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_sssp(graph,i) );

            printf("Baseline\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_dijkstra(graph,i) );
        }
    }

    printf("\n\n");

    for(int k = 1000000;k<5000000;k+=1000000)
    {
        Graph* graph = getRandomGraph(k,edges);

        for(unsigned i = 0; i<device_count; i++)
        {
            printf("Optimized\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_sssp(graph,i) );

            printf("Baseline\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_dijkstra(graph,i) );
        }
    }

    edges *= 10;
    printf("\n\n");

    for(int k = 10000;k<50000;k+=10000)
    {
        Graph* graph = getRandomGraph(k,edges);

        for(unsigned i = 0; i<device_count; i++)
        {
            printf("Optimized\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_sssp(graph,i) );

            printf("Baseline\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_dijkstra(graph,i) );
        }
    }
    printf("\n\n");

    for(int k = 100000;k<500000;k+=100000)
    {
        Graph* graph = getRandomGraph(k,edges);

        for(unsigned i = 0; i<device_count; i++)
        {
            printf("Optimized\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_sssp(graph,i) );

            printf("Baseline\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_dijkstra(graph,i) );
        }
    }

    printf("\n\n");

    edges*=10;

    for(int k = 1000;k<5000;k+=1000)
    {
        Graph* graph = getRandomGraph(k,edges);

        for(unsigned i = 0; i<device_count; i++)
        {
            printf("Optimized\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_sssp(graph,i) );

            printf("Baseline\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_dijkstra(graph,i) );
        }
    }

     printf("\n\n");

    for(int k = 10000;k<50000;k+=10000)
    {
        Graph* graph = getRandomGraph(k,edges);

        for(unsigned i = 0; i<device_count; i++)
        {
            printf("Optimized\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_sssp(graph,i) );

            printf("Baseline\t%d nodes\t %u edges\ton device %d\t: %.2f ms\n",k,edges,i,test_dijkstra(graph,i) );
        }
    }

    return 0;
}

