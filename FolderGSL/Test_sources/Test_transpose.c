#include <Test_transpose.h>
#include <graph_transpose.h>
#include <cl_utils.h>
#include <CL/cl.h>
#include <graph.h>
#include <time.h>
#include <stdbool.h>
#include <benchmark_utils.h>

#define CSVFILENAME_TOPO "transpose.csv"
#define REPEATS 10

unsigned long measure_time_transpose(Graph* graph, unsigned device_id);

void benchmark_transpose(Graph* graph)
{
    // Create path to the kernel file
    char csv_file_transpose[1024];
    generate_path_name_csv(CSVFILENAME_TOPO,csv_file_transpose);
    unsigned num_devices = cluCountDevices();

    //Create CSV File for documenting results
    initCsv(csv_file_transpose,num_devices);

    for(unsigned device = 0; device < num_devices;device++)
    {
        printf("Processing transpose for device : %u\n",device);
        long unsigned time = 0;
        for(int i = 0; i<REPEATS;i++)
        {
           time += measure_time_transpose(graph,device);
        }

        time = time/REPEATS;

        writeToCsv(csv_file_transpose,graph->V,graph->E,device,time);

        printf("Done!\n");
    }
}

unsigned long measure_time_transpose(Graph* graph, unsigned device_id)
{
    //create result variables
    Graph* out = getEmptyGraph(graph->V,graph->E);
    long unsigned time;

    transpose_parallel(graph,device_id,&time);

    freeGraph(out);
    return time;
}

void verify_transpose_parallel(Graph* graph)
{
    //create result variables
    cl_uint* out_order_parallel = (cl_uint*)malloc(sizeof(cl_uint) * graph->V);

    printf("\n%s\n","verify_transpose");

    //Iterate over available devices and calculate the transpose of the graph
    for(unsigned i = 0; i<cluCountDevices();i++)
    {
         printf("%s\n",cluDeviceTypeStringFromNum(i));
        printf("Parallel and serial execution produce same results? %s\n",verify_transpose(graph,i) ? "TRUE" : "FALSE");
    }

    free(out_order_parallel);
}

bool verify_transpose(Graph* graph,unsigned device)
{
    unsigned long time_p;
    unsigned long time_s;
    Graph* graph_t_p = transpose_parallel(graph,device,&time_p);
    Graph* graph_t_s = transpose_serial(graph,&time_s);

    printf("Time parallel transpose: %lu\t Time serial transpose: %lu\n", time_p, time_s);

    bool ret_val = true;

    for(int i = 0; i<graph->V;i++)
    {
        if(graph_t_p->vertices[i+1] != graph_t_s->vertices[i+1] || graph_t_p->vertices[i] != graph_t_s->vertices[i] )
        {
            ret_val = false;
            break;
        }

        unsigned num_neighbors_p = graph_t_p->vertices[i+1] - graph_t_p->vertices[i];
        cl_uint* neighbors_p = (cl_uint*)malloc(sizeof(cl_uint)*num_neighbors_p);
        cl_uint* neighbors_s = (cl_uint*)malloc(sizeof(cl_uint)*num_neighbors_p);
        int index = 0;
        for(int j = graph_t_p->vertices[i]; j<graph_t_p->vertices[i+1];j++)
        {
            neighbors_p[index] = graph_t_p->edges[j];
            neighbors_s[index++] = graph_t_s->edges[j];
        }
        ret_val = have_same_neighbors(neighbors_p,neighbors_s,num_neighbors_p);
        free(neighbors_p);
        free(neighbors_s);
        if(!ret_val)
        {
            break;
        }
    }

    freeGraph(graph_t_p);
    freeGraph(graph_t_s);
    return ret_val;
}

// Checks if all elements in arr1 are contained in arr2 and vice versa,
// i.e. checks if the arrays contain the same values
bool have_same_neighbors(cl_uint *arr1, cl_uint *arr2, unsigned length)
{
    bool ret_val1;
    bool ret_val2;
    for(int i = 0; i<length; i++)
    {

        ret_val1 = false;
        ret_val2 = false;

        for(int j = 0; j<length;j++)
        {
            if(arr1[i] == arr2[j])
            {
                ret_val1 = true;
                break;
            }
        }
        for(int j = 0; j<length;j++)
        {
            if(arr1[j] == arr2[i])
            {
                ret_val2 = true;
                break;
            }
        }
        if(!(ret_val1 && ret_val2))
            return false;

    }

    return true;
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



