#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <dijkstra_serial.h>
#include <stdbool.h>
#include <time_ms.h>

/*float* dijkstra_serial(Graph* graph, unsigned source)
{
    graph->vertices = (unsigned*) realloc(graph->vertices,sizeof(unsigned) * (graph->V+1));
    graph->vertices[graph->V] = graph->E;

    bool* mask_array = (bool*)calloc(graph->V,sizeof(unsigned));
    mask_array[source] = true;

    float* cost_array = (float*)malloc(graph->V * sizeof(float));
    bool finished = false;

    for(int i = 0; i<graph->V;i++)
    {
        cost_array[i] = FLT_MAX;
    }
    cost_array[source] = 0.0f;

    //unsigned* neighbors = &graph->edges[source];
    unsigned num_neighbors;// = graph->vertices[source+1] - graph->vertices[source];
    unsigned current = source;
    unsigned long start_time = time_ms();
    while(!finished)
    {
        finished = true;
        float min = FLT_MAX;
        unsigned m = 0;
        num_neighbors = graph->vertices[current+1] - graph->vertices[current];
        unsigned* neighbors = &graph->edges[graph->vertices[current]];

        for(int i = 0; i<num_neighbors;i++)
        {
            unsigned neighbor = neighbors[i];
            float weight = cost_array[current] + graph->weight[graph->vertices[current] + i];
            // Update Cost for neighbors
            if(mask_array[neighbor] == false && weight < cost_array[neighbor])
            {
                cost_array[neighbor] = weight;
                finished = false;
            }
        }

        for(int i = 0; i<graph->V;i++)
        {
             // Save Minimum path for next main loop iteration
            if( mask_array[i] == false && min>cost_array[i])
            {
                min = cost_array[i];
                m = i;
                finished = false;
            }
        }

        current = m;
        mask_array[m] = true;
    }

    unsigned long total_time = time_ms() - start_time;
    printf("Time for source node %u serial Dijkstra: %lu\n",source,total_time);

    free(mask_array);
    return cost_array;

}*/
