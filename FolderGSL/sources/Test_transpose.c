#include <test_topo_order.h>
#include <graph_transpose.h>
#include <cl_utils.h>
#include <CL/cl.h>
#include <graph.h>
#include <time.h>
#include <time_ms.h>
#include <stdbool.h>
#include <benchmark_utils.h>

#define CSVFILENAME_TOPO "transpose.csv"
#define REPEATS 2

unsigned long measure_time_transpose(Graph* graph, unsigned device_id);
void benchmark_transpose(Graph* graph)
{
    // Create path to the kernel file
    char csv_file_transpose[1024];
    generate_path_name(CSVFILENAME_TOPO,csv_file_transpose);
    unsigned num_devices = cluCountDevices();

    //Create CSV File for documenting results
    initCsv(csv_file_transpose,num_devices);

    for(unsigned device = 0; device < num_devices;device++)
    {
        long unsigned time = 0;
        for(int i = 0; i<REPEATS;i++)
        {
           time += measure_time_transpose(graph,device);
        }

        time = time/REPEATS;

        writeToCsv(csv_file_transpose,graph->V,graph->E,device,time);
    }
}

unsigned long measure_time_transpose(Graph* graph, unsigned device_id)
{
    //create result variables
    Graph* out = getEmptyGraph(graph->V,graph->E);
    unsigned long time;

    transpose_parallel(graph,out,device_id,&time);

    freeGraph(out);
    return time;
}

/*void test_topo_sort(Graph* graph)
{
    //create result variables
    cl_uint* out_order_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);

    printf("%s\n","test_topo_sort");

    //Iterate over available devices and calculate the topological ordering
    for(unsigned i = 0; i<cluCountDevices();i++)
    {
        cl_device_id tmp = cluInitDevice(i,NULL,NULL);
        printf("%s\n",cluGetDeviceDescription(tmp,i));
        printf("Parallel and serial execution produce same results? ");
        topological_order(graph,out_order_parallel,i,NULL);
        printf("%s\n",verify_topo_sort(graph,out_order_parallel) ? "TRUE" : "FALSE");
    }

    free(out_order_parallel);
}

bool verify_topo_sort(Graph* graph, cl_uint *out_order_parallel)
{
        //create result variables
        cl_uint* out_order_serial = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
        //calculate the algorithm
        serial_topo_order(graph,out_order_serial);

        bool result =  cl_uint_arr_equal(out_order_parallel,out_order_serial,graph->V);

        free(out_order_serial);

        return result;

}*/



