#ifndef BELLMAN_FORD_H_INCLUDED
#define BELLMAN_FORD_H_INCLUDED

#include <graph.h>
// Function to calculate the graph->V+1th iteration to determine if there exists negative cycles
// returns true if there's at least one negative cylce, otherwise false
// if there exists a negative cycle detected at node i, the i'th entry of negative_cycles is set to true
// in_cost is the cost array of any single source shortest path calculation like for example dijkstra
bool bellman_ford(Graph* graph, unsigned device_num, cl_float* in_cost, bool* negative_cycles);

// Function which calulates all negative cycles, and saves them as a list of a list of a node
// To use it declare a unsigned** variable/a unsigned * variable ( declare! memory will be allocated by this function )and commit the adresses as third/fourth argument
// The first one saves the acutal cycles, the second one the number of elements for each cycle
void createNegativeCycles(Graph* graph,unsigned device, unsigned ***cycles_out, unsigned **num_path_elements, cl_float *cost, cl_uint path);

// Function to free allocated space of cycles_out
void freeNegativeCycles(int length,unsigned **cycles_out);

#endif // BELLMAN_FORD_H_INCLUDED
