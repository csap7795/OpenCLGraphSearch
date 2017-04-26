#include <test_sssp.h>
#include <sssp.h>
#include <cl_utils.h>
#include <CL/cl.h>
#include <graph.h>
#include <time.h>
#include <time_ms.h>
#include <stdbool.h>
#include <benchmark_utils.h>

#define REPEATS 10
#define CSVFILENAME_SSSP "sssp.csv"

void benchmark_sssp(Graph* graph, unsigned source)
{
    // Create path to the kernel file
    char csv_file_sssp[1024];
    generate_path_name(CSVFILENAME_SSSP,csv_file_sssp);
    unsigned num_devices = cluCountDevices();

    //Create CSV File for documenting results
    initCsv(csv_file_sssp,num_devices);

    for(unsigned device = 0; device < num_devices;device++)
    {
        long unsigned time = 0;
        for(int i = 0; i<REPEATS;i++)
        {
           time += measure_time_sssp(graph,source,device);
        }

        time = time/REPEATS;

        writeToCsv(csv_file_sssp,graph->V,graph->E,device,time);
    }
}

unsigned long measure_time_sssp(Graph* graph, unsigned source, unsigned device_id)
{
    //create result variables
    cl_float* out_cost_parallel = (cl_float*)malloc(sizeof(cl_float) * graph->V);
    unsigned long time;

    sssp(graph,source,out_cost_parallel,device_id,&time);

    free(out_cost_parallel);
    return time;
}

void verify_sssp_parallel(Graph* graph,unsigned source)
{
    //create result variables
    cl_float* out_cost_parallel = (cl_float*)malloc(sizeof(cl_float) * graph->V);

    printf("%s\n","test_sssp");

    //Iterate over available devices and calculate the topological ordering
    for(unsigned i = 0; i<cluCountDevices();i++)
    {
        cl_device_id tmp = cluInitDevice(i,NULL,NULL);
        printf("%s\n",cluGetDeviceDescription(tmp,i));
        sssp(graph,source,out_cost_parallel,i,NULL);
        printf("Parallel and serial execution produce same results? ");
        printf("%s\n",verify_sssp(graph,out_cost_parallel,source) ? "TRUE" : "FALSE");
    }

    free(out_cost_parallel);
}

bool verify_sssp(Graph* graph, cl_float *out_cost_parallel,unsigned source)
{
        //create result variables
        cl_float* out_cost_serial = (cl_float*)malloc(sizeof(cl_float) * graph->V);
        cl_uint* out_path_serial = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
        //calculate the algorithm
        dijkstra_serial(graph,out_cost_serial,out_path_serial,source);

        bool result =  cl_float_arr_equal(out_cost_parallel,out_cost_serial,graph->V);

        free(out_cost_serial);
        free(out_path_serial);

        return result;

}

