#include <dijkstra_parallel.h>
#include <cl_utils.h>

#include <stdio.h>
#include <CL/cl.h>
#include <stdbool.h>
#include <limits.h>
#include <float.h>
#include <unistd.h>
#include <libgen.h>

static cl_program program;
static cl_context context;
static cl_command_queue command_queue;
static cl_device_id device;

static void build_program(size_t device_num)
{
    device = cluInitDevice(device_num,&context,&command_queue);
    //printf("%s\n\n",cluGetDeviceDescription(device,device_num));

    char* filename = "/bellman_ford.cl";
    char cfp[1024];

    /* Create path to the kernel file*/
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s%s",dirname(dirname(cfp)),"/kernels",filename);
    program = cluBuildProgramFromFile(context,device,kernel_file,NULL);
}

//Is able to detect negative cycles in the graph under the condition that they are connected to the source node
bool bellman_ford(Graph* graph, unsigned device_num, cl_float* in_cost, bool* negative_cycles)
{
    /*First, build the program*/
    build_program(device_num);

    cl_int err;

    // Create Memory Buffers for Graph Data
    cl_mem vertice_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (graph->V+1), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating verticebuffer");

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    cl_mem weight_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating weight_buffer");

    // Create Memory Buffers for Cost Objects
    cl_mem cost_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_float) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating cost_buffer");

    // Create Buffers for the results
    cl_mem negative_cycle_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_short) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating verticebuffer");

    cl_mem detected_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating verticebuffer");

    // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), graph->vertices , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, graph->V * sizeof(cl_uint), sizeof(cl_uint), &graph->E , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, weight_buffer, CL_TRUE,0, graph->E * sizeof(cl_uint), graph->weight,0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue,cost_buffer, CL_FALSE,0,graph->V*sizeof(cl_float), in_cost,0,NULL,NULL);
    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    // Create the kernel
    cl_kernel negativeCycle_kernel = clCreateKernel(program,"negativeCycle",&err);
    CLU_ERRCHECK(err,"Failed to create negative cycle kernel from program");

    // Set the kernel Arguments
    cluSetKernelArguments(negativeCycle_kernel,6,sizeof(cl_mem),&vertice_buffer,sizeof(cl_mem),&edge_buffer,sizeof(cl_mem),&weight_buffer,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),&negative_cycle_buffer,sizeof(cl_mem),&detected_buffer);

    cl_int detected = 0;
    err = clEnqueueWriteBuffer(command_queue, detected_buffer, CL_FALSE, 0, sizeof(cl_int), &detected , 0, NULL, NULL);
    CLU_ERRCHECK(err,"Failed copying detected_flag to detected_buffers");

    //Execute the kernel
    size_t globalSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, negativeCycle_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    //read back results
    err = clEnqueueReadBuffer(command_queue,detected_buffer,CL_TRUE,0,sizeof(cl_int),&detected,0,NULL,NULL);

    // If the kernel detected negative cycles, read back where the negative cycles are located, else write false into check_cycles
    if(detected == 1 && negative_cycles != NULL)
    {
        cl_short *tmp = (cl_short*)malloc(sizeof(cl_short)*graph->V);
        err = clEnqueueReadBuffer(command_queue,negative_cycle_buffer,CL_TRUE,0,sizeof(cl_short)*graph->V,tmp,0,NULL,NULL);
        for(int i = 0; i<graph->V;i++)
            negative_cycles[i] = tmp[i];
        free(tmp);
    }

    // Clean up
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    err |= clReleaseKernel(negativeCycle_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(vertice_buffer);
    err |= clReleaseMemObject(edge_buffer);
    err |= clReleaseMemObject(weight_buffer);
    err |= clReleaseMemObject(cost_buffer);
    err |= clReleaseMemObject(negative_cycle_buffer);
    err |= clReleaseMemObject(detected_buffer);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err, "Failed during finalizing OpenCL");

    return detected;
}

// returns the number of negative cycles detected and the actual cycle paths are saved in cycles_out
int createNegativeCycles(Graph* graph,unsigned device, unsigned ***cycles_out,unsigned **num_path_elements, cl_float *cost, cl_uint* path)
{
    bool* neg_cycles = (bool*)calloc(graph->V,sizeof(bool));
    bellman_ford(graph,device,cost,neg_cycles);
    size_t count = 0;
    for(int i = 0; i<graph->V;i++)
        if(neg_cycles[i])
            count++;

    *cycles_out = (unsigned**)malloc(sizeof(unsigned*) * count);
    *num_path_elements = (unsigned*)malloc(sizeof(unsigned) * count);
    count = 0;

    for(int i = 0; i<graph->V;i++)
    {
        if(!neg_cycles[i])
            continue;

       size_t pathlength = 1;
        cl_uint predecessor = path[i];
        while(predecessor != i)
        {
            predecessor = path[predecessor];
            pathlength++;
        }
        (*cycles_out)[count] = (unsigned*)malloc(sizeof(unsigned) * pathlength);
        predecessor = path[i];
        for(int k = 0; k<pathlength;k++)
        {
            (*cycles_out)[count][k] = predecessor;
            predecessor = path[predecessor];
        }
        (*num_path_elements)[count] = pathlength;
        count++;
    }

    free(neg_cycles);
    return count+1;
}

void freeNegativeCycles(int length,unsigned **cycles_out)
{
    for(int i = 0; i<length;i++)
        free(cycles_out[i]);

    free(cycles_out);
}
