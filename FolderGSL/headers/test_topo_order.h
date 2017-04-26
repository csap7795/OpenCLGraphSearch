#ifndef TEST_TOPO_ORDER_H_INCLUDED
#define TEST_TOPO_ORDER_H_INCLUDED

#include<stdbool.h>
#include<CL/cl.h>
#include<graph.h>

void benchmark_topo(Graph* graph);
void serial_topo_order(Graph* graph, cl_uint* out_order_serial);
bool verify_topo_sort(Graph* graph, cl_uint *out_order_parallel);
void test_topo_sort(Graph* graph);
unsigned long measure_time_topo(Graph* graph, unsigned device_id);

#endif // TEST_TOPO_ORDER_H_INCLUDED
