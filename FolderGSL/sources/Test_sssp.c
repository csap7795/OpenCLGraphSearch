#include <test_sssp.h>
#include <sssp.h>
#include <cl_utils.h>
#include <CL/cl.h>
#include <graph.h>
#include <time.h>
#include <stdbool.h>
#include <benchmark_utils.h>
#include <dikstra_path.h>

#define REPEATS 10
#define CSVFILENAME_SSSP "sssp.csv"
#define CSVFILENAME_PRECALC "pre_sssp.csv"
#define CSVFILENAME_DIJKSTRA "dijkstra_path.csv"

void benchmark_sssp(Graph* graph, unsigned source)
{
    // Create path to the kernel file
    char csv_file_sssp[1024];
    char csv_file_dijkstra_path[1024];
    char csv_file_precalc[1024];
    generate_path_name_csv(CSVFILENAME_SSSP,csv_file_sssp);
    generate_path_name_csv(CSVFILENAME_PRECALC,csv_file_precalc);
    generate_path_name_csv(CSVFILENAME_DIJKSTRA,csv_file_dijkstra_path);
    unsigned num_devices = cluCountDevices();

    //Create CSV File for documenting results
    initCsv(csv_file_sssp,num_devices);
    initCsv(csv_file_precalc,num_devices);
    initCsv(csv_file_dijkstra_path,num_devices);

    for(unsigned device = 0; device < num_devices;device++)
    {
        long unsigned time_sssp = 0;
        long unsigned time_pre_sssp = 0;
        long unsigned total_time = 0;
        long unsigned precalc_time = 0;
        long unsigned time_dijkstra = 0;

        for(int i = 0; i<REPEATS;i++)
        {
           measure_time_sssp(graph,source,device,&total_time,&precalc_time);
           time_sssp += total_time;
           time_pre_sssp += precalc_time;
           time_dijkstra += measure_time_dijkstra_path(graph,source,device);
        }

        time_sssp = time_sssp/REPEATS;
        time_pre_sssp = time_pre_sssp/REPEATS;
        time_dijkstra = time_dijkstra/REPEATS;

        writeToCsv(csv_file_sssp,graph->V,graph->E,device,time_sssp);
        writeToCsv(csv_file_precalc,graph->V,graph->E,device,time_pre_sssp);
        writeToCsv(csv_file_dijkstra_path,graph->V,graph->E,device,time_dijkstra);
    }
}

void measure_time_sssp(Graph* graph, unsigned source, unsigned device_id, unsigned long* total_time, unsigned long* precalc_time)
{
    //create result variables
    cl_float* out_cost_parallel = (cl_float*)malloc(sizeof(cl_float) * graph->V);
    cl_uint* out_path_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);

    sssp(graph,source,out_cost_parallel,out_path_parallel,device_id,total_time,precalc_time);

    free(out_path_parallel);
    free(out_cost_parallel);
}

unsigned long measure_time_dijkstra_path(Graph* graph, unsigned source, unsigned device_id)
{
    //create result variables
    cl_float* out_cost_parallel = (cl_float*)malloc(sizeof(cl_float) * graph->V);
    cl_uint* out_path_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
    unsigned long time;

    dijkstra_path(graph,source,out_cost_parallel,out_path_parallel,device_id,&time);

    free(out_cost_parallel);
    free(out_path_parallel);
    return time;
}

void verify_dijkstra_path_parallel(Graph* graph, unsigned source)
{
    //create result variables
    cl_float* out_cost_parallel = (cl_float*)malloc(sizeof(cl_float) * graph->V);
    cl_uint* out_path_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);

    //Iterate over available devices and calculate the topological ordering
    for(unsigned i = 0; i<cluCountDevices();i++)
    {
        cl_device_id tmp = cluInitDevice(i,NULL,NULL);
        printf("%s\n",cluGetDeviceDescription(tmp,i));
        bool result = true;
        for(int j = 0; j<REPEATS;j++){
            dijkstra_path(graph,source,out_cost_parallel,out_path_parallel,i,NULL);
            result &= verify_dijkstra_path(graph,out_cost_parallel,out_path_parallel,source);
        }
        printf("Parallel and serial execution produce same results?\t");
        printf("%s\n", result ? "TRUE" : "FALSE");
    }

    free(out_cost_parallel);
}

void verify_sssp_parallel(Graph* graph,unsigned source)
{
    //create result variables
    cl_float* out_cost_parallel = (cl_float*)malloc(sizeof(cl_float) * graph->V);
    cl_uint* out_path_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);

    //Iterate over available devices and calculate the topological ordering
    for(unsigned i = 0; i<cluCountDevices();i++)
    {
        cl_device_id tmp = cluInitDevice(i,NULL,NULL);
        printf("%s\n",cluGetDeviceDescription(tmp,i));
        bool result = true;
        for(int j = 0; j<REPEATS;j++){
            sssp(graph,source,out_cost_parallel,out_path_parallel,i,NULL,NULL);
            result &= verify_sssp(graph,out_cost_parallel,out_path_parallel,source);
        }
        printf("Parallel and serial execution produce same results? ");
        printf("%s\n", result ? "TRUE" : "FALSE");
    }

    free(out_path_parallel);
    free(out_cost_parallel);
}

bool verify_sssp(Graph* graph, cl_float *out_cost_parallel, cl_uint *out_path_parallel,unsigned source)
{
        //create result variables
        cl_float* out_cost_serial = (cl_float*)malloc(sizeof(cl_float) * graph->V);
        cl_uint* out_path_serial = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
        //calculate the algorithm
        dijkstra_serial(graph,out_cost_serial,out_path_serial,source);

        bool result =  cl_float_arr_equal(out_cost_parallel,out_cost_serial,graph->V);
        result &=  cl_uint_arr_equal(out_path_parallel,out_path_serial,graph->V);

        free(out_cost_serial);
        free(out_path_serial);

        return result;

}

bool verify_dijkstra_path(Graph* graph, cl_float *out_cost_parallel, cl_uint *out_path_parallel,unsigned source)
{
        //create result variables
        cl_float* out_cost_serial = (cl_float*)malloc(sizeof(cl_float) * graph->V);
        cl_uint* out_path_serial = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
        //calculate the algorithm
        dijkstra_serial(graph,out_cost_serial,out_path_serial,source);

        bool result =  cl_float_arr_equal(out_cost_parallel,out_cost_serial,graph->V);
        result &=  cl_uint_arr_equal(out_path_parallel,out_path_serial,graph->V);

        free(out_cost_serial);
        free(out_path_serial);

        return result;

}

