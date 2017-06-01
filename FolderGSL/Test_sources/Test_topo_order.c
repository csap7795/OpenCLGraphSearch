#include <Test_topo_order.h>
#include <topo_sort.h>
#include <cl_utils.h>
#include <CL/cl.h>
#include <graph.h>
#include <time.h>
#include <stdbool.h>
#include <benchmark_utils.h>
#include <stdio.h>

#define CSVFILENAME_TOPO "topo.csv"
#define CSVFILENAME_TOPO_OPT "topo_opt.csv"
#define REPEATS 1

void benchmark_topo(Graph* graph)
{
    // Create path to the kernel file
    char csv_file_topo[1024];
    generate_path_name_csv(CSVFILENAME_TOPO,csv_file_topo);
    char csv_file_topo_opt[1024];
    generate_path_name_csv(CSVFILENAME_TOPO_OPT,csv_file_topo_opt);
    unsigned num_devices = cluCountDevices();

    //Create CSV File for documenting results
    initCsv(csv_file_topo,num_devices);
    initCsv(csv_file_topo_opt,num_devices);

    for(unsigned device = 0; device < num_devices;device++)
    {
        printf("Processing topological order for device : %u\n",device);
        long unsigned time = 0;
        long unsigned time_opt = 0;
        for(int i = 0; i<REPEATS;i++)
        {
           time += measure_time_topo_normal(graph,device);
           time_opt += measure_time_topo_opt(graph,device);
        }

        time = time/REPEATS;
        time_opt = time_opt/REPEATS;

        writeToCsv(csv_file_topo,graph->V,graph->E,device,time);
        writeToCsv(csv_file_topo_opt,graph->V,graph->E,device,time_opt);

        printf("Done!\n");
    }
}
unsigned long measure_time_topo_opt(Graph* graph, unsigned device_id)
{
    //create result variables
    cl_uint* out_order_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
    unsigned long time;

    topological_order_opt(graph,out_order_parallel,device_id,NULL,&time);

    free(out_order_parallel);
    return time;
}
unsigned long measure_time_topo_normal(Graph* graph, unsigned device_id)
{
    //create result variables
    cl_uint* out_order_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
    unsigned long time;

    topological_order_normal(graph,out_order_parallel,device_id,&time);

    free(out_order_parallel);
    return time;
}

void verify_topo_sort_normal_parallel(Graph* graph)
{
    //create result variables
    cl_uint* out_order_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);

    printf("\n%s\n","verify_topo_sort_normal");

    //Iterate over available devices and calculate the topological ordering
    for(unsigned i = 0; i<cluCountDevices();i++)
    {
       // cl_device_id tmp = cluInitDevice(i,NULL,NULL);
         printf("%s\n",cluDeviceTypeStringFromNum(i));
        printf("Parallel and serial execution produce same results? ");
        topological_order_normal(graph,out_order_parallel,i,NULL);
        printf("%s\n",verify_topo_sort(graph,out_order_parallel) ? "TRUE" : "FALSE");
    }

    free(out_order_parallel);
}

void verify_topo_sort_opt_parallel(Graph* graph)
{
    //create result variables
    cl_uint* out_order_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);

    printf("\n%s\n","verify_topo_sort_opt");

    //Iterate over available devices and calculate the topological ordering
    for(unsigned i = 0; i<cluCountDevices();i++)
    {
        //cl_device_id tmp = cluInitDevice(i,NULL,NULL);
         printf("%s\n",cluDeviceTypeStringFromNum(i));
        printf("Parallel and serial execution produce same results? ");
        topological_order_opt(graph,out_order_parallel,i,NULL,NULL);
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

}

void serial_topo_order(Graph* graph, cl_uint* out_order_serial)
{
    cl_uint* inEdges = (cl_uint*)calloc(graph->V,sizeof(cl_uint));
    int* active = (int*)calloc(graph->V,sizeof(int));
    int i;
    for(i = 0; i<graph->V-1;i++)
    {
        for(int j = graph->vertices[i];j<graph->vertices[i+1];j++)
        {
            inEdges[graph->edges[j]]++;
        }
    }
    //Last element
    for(int j = graph->vertices[i];j<graph->E;j++)
    {
            inEdges[graph->edges[j]]++;
    }

    for(i = 0; i<graph->V;i++)
    {
        out_order_serial[i] = CL_UINT_MAX;
        if(inEdges[i] == 0)
            active[i] = 0;
    }

    //Iterate over the graph as long as there are changes
    for(int level=0; level<graph->V; level++)
    {
        i = 0;
        for(i = 0; i<graph->V-1;i++)
        {
            if(active[i] == level && inEdges[i] == 0)
            {
                out_order_serial[i] = level;
                for(int j = graph->vertices[i];j<graph->vertices[i+1];j++)
                {
                    inEdges[graph->edges[j]]--;
                    active[graph->edges[j]] = level+1;
                }
            }
        }
        //Last element
        if(active[i] == level && inEdges[i]==0)
        {
            out_order_serial[i] = level;
            for(int j = graph->vertices[i];j<graph->E;j++)
            {
                    inEdges[graph->edges[j]]--;
                    active[j] = level+1;
            }
       }
    }

    free(inEdges);
    free(active);
}



