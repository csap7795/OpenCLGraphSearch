#include <Test_dijkstra.h>
#include <bfs_parallel.h>
#include <cl_utils.h>
#include <CL/cl.h>
#include <graph.h>
#include <time.h>
#include <stdbool.h>
#include <dijkstra_parallel.h>
#include <benchmark_utils.h>

#define CSVFILENAME_DIJKSTRA "dijkstra.csv"
#define REPEATS 10

void benchmark_dijkstra(Graph* graph, unsigned source)
{
    // Create path to the kernel file
    char csv_file_dijkstra[1024];
    generate_path_name_csv(CSVFILENAME_DIJKSTRA,csv_file_dijkstra);
    unsigned num_devices = cluCountDevices();

    //Create CSV File for documenting results
    initCsv(csv_file_dijkstra,num_devices);

    for(unsigned device = 0; device < num_devices;device++)
    {

        printf("Processing dijkstra for device : %u\n",device);
        long unsigned time = 0;
        for(int i = 0; i<REPEATS;i++)
        {
           time += measure_time_dijkstra(graph,source,device);
        }

        time = time/REPEATS;

        writeToCsv(csv_file_dijkstra,graph->V,graph->E,device,time);
        printf("Done!\n");
    }
}

unsigned long measure_time_dijkstra_cpu(Graph* graph, unsigned source)
{
    //create result variables
    cl_float* out_cost_parallel = (cl_float*)malloc(sizeof(cl_float) * graph->V);
    cl_uint* out_path_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
    unsigned long time;

    dijkstra_parallel_cpu(graph,source,1,out_cost_parallel,out_path_parallel,&time);

    free(out_cost_parallel);
    free(out_path_parallel);
    return time;
}

unsigned long measure_time_dijkstra(Graph* graph, unsigned source, unsigned device_id)
{
    //create result variables
    cl_float* out_cost_parallel = (cl_float*)malloc(sizeof(cl_float) * graph->V);
    cl_uint* out_path_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
    unsigned long time;

    dijkstra_parallel(graph,source,device_id,out_cost_parallel,out_path_parallel,&time);

    free(out_cost_parallel);
    free(out_path_parallel);
    return time;
}

void verify_dijkstra_parallel(Graph* graph, unsigned source)
{
    //create result variables
    cl_float* out_cost_parallel = (cl_float*)malloc(sizeof(cl_float) * graph->V);
    cl_uint* out_path_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);

    printf("\n%s\n","verify_dijkstra");

    //Iterate over available devices and calculate SSSP, verify the result
    for(int i = 0; i<cluCountDevices();i++)
    {
        //cl_device_id tmp = cluInitDevice(i,NULL,NULL);
         printf("%s\n",cluDeviceTypeStringFromNum(i));
        dijkstra_parallel(graph,source,i,out_cost_parallel,out_path_parallel,NULL);
        printf("Parallel and serial execution produce same results? ");
        printf("%s\n",verify_dijkstra(graph,out_cost_parallel,out_path_parallel,source) ? "TRUE" : "FALSE");
    }


    free(out_cost_parallel);
    free(out_path_parallel);

}


bool verify_dijkstra(Graph* graph, cl_float *out_cost_parallel, cl_uint* out_path_parallel,unsigned source)
{
        //create result variables
        cl_float* out_cost_serial = (cl_float*)malloc(sizeof(cl_float) * graph->V);
        cl_uint* out_path_serial = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);
        //calculate the algorithm
        sssp_normal(graph,source,out_cost_serial,out_path_serial,1,NULL);
        //dijkstra_serial(graph,out_cost_serial,out_path_serial,source);
        bool result = cl_float_arr_equal(out_cost_parallel,out_cost_serial,graph->V);
        //result &=  cl_uint_arr_equal(out_path_parallel,out_path_serial,graph->V);

        free(out_cost_serial);
        free(out_path_serial);

        return result;

}

//Serial Dijkstra
void dijkstra_serial(Graph* graph, cl_float* cost_array,cl_uint* path_array,unsigned source)
{
    graph->vertices = (unsigned*) realloc(graph->vertices,sizeof(unsigned) * (graph->V+1));
    graph->vertices[graph->V] = graph->E;

    bool* mask_array = (bool*)calloc(graph->V,sizeof(unsigned));
    mask_array[source] = true;

    bool finished = false;

    for(int i = 0; i<graph->V;i++)
    {
        cost_array[i] = CL_FLT_MAX;
        path_array[i] = CL_UINT_MAX;
    }
    cost_array[source] = 0.0f;
    path_array[source] = source;

    //unsigned* neighbors = &graph->edges[source];
    unsigned num_neighbors;// = graph->vertices[source+1] - graph->vertices[source];
    unsigned current = source;
    while(!finished)
    {
        finished = true;
        float min = CL_FLT_MAX;
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
                path_array[neighbor] = current;
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

    free(mask_array);

}
void bellman_ford_serial(Graph* graph,cl_float* out_cost, cl_uint* out_path, unsigned source)
{
    for(int i = 0; i<graph->V;i++)
    {
        out_cost[i] = CL_FLT_MAX;
        out_path[i] = CL_UINT_MAX;
    }
    out_cost[source] = 0.0f;
    out_path[source] = source;

    for(int i = 0; i<graph->V;i++)
    {
        for(int j = 0; j<graph->V-1;j++)
        {
            for(cl_uint k = graph->vertices[j]; k < graph->vertices[j+1]; k++)
            {
                cl_uint neighbor = graph->edges[k];
                cl_float distance = graph->weight[k] + out_cost[j];
                if(out_cost[neighbor] > distance)
                {
                    out_cost[neighbor] = distance;
                    out_path[neighbor] = j;
                }
            }

        }
        //Last element
            for(cl_uint k = graph->vertices[graph->V-1]; k < graph->E; k++)
            {
                cl_uint neighbor = graph->edges[k];
                cl_float distance = graph->weight[k] + out_cost[graph->V-1];
                if(out_cost[neighbor] > distance)
                {
                    out_cost[neighbor] = distance;
                    out_path[neighbor] = graph->V-1;
                }
            }
    }

    for(int j = 0; j<graph->V-1;j++)
    {
            for(cl_uint k = graph->vertices[j]; k < graph->vertices[j+1]; k++)
            {
                cl_uint neighbor = graph->edges[k];
                cl_float distance = graph->weight[k] + out_cost[j];
                if(out_cost[neighbor] > distance)
                {
                    printf("Es gibt einen Zyklus negativen Gewichts\n");
                    return;
                }
            }

    }
    for(cl_uint k = graph->vertices[graph->V-1]; k < graph->E; k++)
    {
        cl_uint neighbor = graph->edges[k];
        cl_float distance = graph->weight[k] + out_cost[graph->V-1];
        if(out_cost[neighbor] > distance)
        {
        printf("Es gibt einen Zyklus negativen Gewichts\n");
        return;
        }
    }


}

