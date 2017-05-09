#include<test_bfs.h>
#include <bfs_parallel.h>
#include <cl_utils.h>
#include <CL/cl.h>
#include <graph.h>
#include <time.h>
#include <stdbool.h>
#include <benchmark_utils.h>


#define REPEATS 10

#define CSVFILENAME_WORKGROUP "bfs_workgroup.csv"
#define CSVFILENAME_BASELINE "bfs_baseline.csv"
void benchmark_bfs(Graph* graph, unsigned source)
{

    // Create path to the kernel file
    char csv_file_workgroup[1024];
    char csv_file_baseline[1024];
    generate_path_name(CSVFILENAME_BASELINE,csv_file_baseline);
    generate_path_name(CSVFILENAME_WORKGROUP,csv_file_workgroup);

    unsigned num_devices = cluCountDevices();
    //Create CSVFILES for documenting results
    initCsv(csv_file_workgroup,num_devices);
    initCsv(csv_file_baseline,num_devices);

    for(unsigned device = 1; device < num_devices;device++)
    {
        long unsigned time_baseline = 0;
        long unsigned time_workgroup = 0;
        for(int i = 0; i<REPEATS;i++)
        {
           time_baseline += measure_time_bfs_baseline(graph,source,device);
           time_workgroup += measure_time_bfs_workgroup(graph,source,device);
        }

        time_baseline = time_baseline/REPEATS;
        time_workgroup = time_workgroup/REPEATS;

        writeToCsv(csv_file_workgroup,graph->V,graph->E,device,time_workgroup);
        writeToCsv(csv_file_baseline,graph->V,graph->E,device,time_baseline);
    }

}

unsigned long measure_time_bfs_baseline(Graph* graph, unsigned source, unsigned device_id)
{
    //create result variables
    cl_uint* out_cost_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
    cl_uint* out_path_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
    unsigned long time;

    bfs_parallel_baseline(graph,out_cost_parallel,out_path_parallel,source,device_id,&time);

    free(out_cost_parallel);
    free(out_path_parallel);
    return time;
}

unsigned long measure_time_bfs_workgroup(Graph* graph, unsigned source, unsigned device_id)
{
    //create result variables
    cl_uint* out_cost_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
    cl_uint* out_path_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
    unsigned long time;

    bfs_parallel_workgroup(graph,out_cost_parallel,out_path_parallel,source,device_id,&time);

    free(out_cost_parallel);
    free(out_path_parallel);
    return time;
}

void verify_bfs_baseline(Graph* graph, unsigned source)
{
    //create result variables
    cl_uint* out_cost_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
    cl_uint* out_path_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);

    printf("%s\n","test_bfs_baseline");

    //Iterate over available devices and calculate APSP, verify the result
    for(int i = 0; i<cluCountDevices();i++)
    {
        cl_device_id tmp = cluInitDevice(i,NULL,NULL);
        printf("%s\n",cluGetDeviceDescription(tmp,i));
        bfs_parallel_baseline(graph,out_cost_parallel,out_path_parallel,source,i,NULL);
        printf("Parallel and serial execution produce same results? ");
        printf("%s\n",verify_bfs(graph,out_cost_parallel,out_path_parallel,source) ? "TRUE" : "FALSE");
    }

    free(out_cost_parallel);
    free(out_path_parallel);
}

void verify_bfs_workgroup(Graph* graph, unsigned source)
{
    //create result variables
    cl_uint* out_cost_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
    cl_uint* out_path_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);

    printf("%s\n","test_bfs_workgroup");

    //Iterate over available devices and calculate APSP, verify the result
    for(int i = 0; i<cluCountDevices();i++)
    {
        cl_device_id tmp = cluInitDevice(i,NULL,NULL);
        printf("%s\n",cluGetDeviceDescription(tmp,i));
        bfs_parallel_workgroup(graph,out_cost_parallel,out_path_parallel,source,i,NULL);
        printf("Parallel and serial execution produce same results? ");
        printf("%s\n",verify_bfs(graph,out_cost_parallel,out_path_parallel,source) ? "TRUE" : "FALSE");
    }

    free(out_cost_parallel);
    free(out_path_parallel);

}


bool verify_bfs(Graph* graph, cl_uint *out_cost_parallel, cl_uint* out_path_parallel,unsigned source)
{
        //create result variables
        cl_uint* out_cost_serial = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
        cl_uint* out_path_serial = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
        //calculate the algorithm
        bfs_serial(graph,out_cost_serial,out_path_serial,source);

        bool result =  cl_uint_arr_equal(out_cost_parallel,out_cost_serial,graph->V);

        free(out_cost_serial);
        free(out_path_serial);

        return result;

}

/*Serial Breadth first search*/
void bfs_serial(Graph* graph, cl_uint *cost, cl_uint *path, unsigned source)
{
    for(int i = 0; i<graph->V;i++)
    {
        cost[i] = CL_UINT_MAX;
        path[i] = CL_UINT_MAX;
    }

    cost[source] = 0;
    path[source] = source;

    cl_uint* vertices = (unsigned*)malloc((graph->V+1)*sizeof(cl_uint));
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
            if(cost[i] == current_level)
            {
                unsigned offset = vertices[i];
                num_neighbors = vertices[i+1] - vertices[i];
                for(int j = 0; j<num_neighbors;j++)
                {
                    if(cost[graph->edges[offset+j]] == CL_UINT_MAX)
                    {
                        finished = false;
                        cost[graph->edges[offset+j]] = current_level+1;
                        path[graph->edges[offset+j]] = i;
                    }
                }
            }

        }
    }
    free(vertices);
}

