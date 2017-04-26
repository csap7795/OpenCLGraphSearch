#include "bfs_serial.h"
#include <stdio.h>
#include <stdlib.h>
#include "queue.h"
#include "time_ms.h"
#include <limits.h>

/*unsigned* bfs_serial(Graph* graph,unsigned source)
{
    unsigned* levels = (unsigned*)calloc(graph->V,sizeof(unsigned));
    for(int i = 0; i<graph->V;i++)
        levels[i] = INT_MAX;

    levels[source] = 0;

    unsigned* vertices = (unsigned*)malloc((graph->V+1)*sizeof(unsigned));
    for(int i = 0; i< graph->V;i++)
        vertices[i] = graph->vertices[i];

    vertices[graph->V] = graph->E;

    unsigned num_neighbors;
    bool finished = false;
    unsigned current_level = 0;

    for(;!finished;current_level++)
    {
        finished = true;
        for(int i = 0; i<graph->V;i++)
        {
            if(levels[i] == current_level)
            {
                unsigned offset = vertices[i];
                num_neighbors = vertices[i+1] - vertices[i];
                for(int j = 0; j<num_neighbors;j++)
                {
                    if(levels[graph->edges[offset+j]] == INT_MAX)
                    {
                        finished = false;
                        levels[graph->edges[offset+j]] = current_level+1;
                    }
                }
            }

        }
    }

    free(vertices);

    return levels;

}*/

void bfs_serial_queue(Graph* graph, unsigned source)
{
    queue* bfs_queue = init_queue();
    queue_add(bfs_queue,source);

    bool* visited = (bool*) calloc(graph->V,sizeof(bool));
    visited[source] = true;

    unsigned long start_time = time_ms();
    while(!queue_is_empty(bfs_queue))
    {
        unsigned v = queue_get(bfs_queue);
        unsigned neighbors = 0;
        unsigned edge_index = graph->vertices[v];

        if(v != graph->V-1)
            neighbors = graph->vertices[v+1] - edge_index;

        else
            neighbors = graph->E - edge_index;

        for(unsigned i = 0; i<neighbors;i++)
        {
            unsigned index = edge_index + i;
            if(!visited[graph->edges[index]])
            {
                visited[graph->edges[index]] = true;
                queue_add(bfs_queue,graph->edges[index]);
            }
        }
    }
    free(visited);
    printf("Time for source node %u and serial execution : %lu\n",source,time_ms()-start_time);
}

