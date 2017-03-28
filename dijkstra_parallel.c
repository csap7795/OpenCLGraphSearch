#include "dijkstra_parallel.h"
#include <stdio.h>
#include <CL/cl.h>
#include <stdbool.h>
#include <limits.h>
#include <float.h>
#include <dijkstra_serial.h>
#include <cl_utils.h>
#include <time_ms.h>
#include <unistd.h>
#include <libgen.h>

#define CL_DEVICE 1

unsigned long dijkstra_parallel(Graph* graph, unsigned source, unsigned device_num)
{

    unsigned long start_time = time_ms();
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device = cluInitDevice(device_num,&context,&command_queue);

    cl_int err;

    //printf("%s\n",cluGetDeviceDescription(device,i));

    // Create Memory Buffers for Graph Data
    cl_mem vertice_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (graph->V+1), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating verticebuffer");

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    cl_mem weight_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating weight_buffer");

    // Create Memory Buffers for Helper Objects
    cl_mem mask_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_bool) * graph->V, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating mask_buffer");

    cl_mem cost_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating cost_buffer");

    cl_mem update_cost_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating update_cost_buffer");

    cl_mem semaphore_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_int) * graph->V,NULL,&err);
    CLU_ERRCHECK(err,"Failed creating semaphore_buffer");

    cl_mem finished_flag = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_bool), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating finished_flag");

    // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), graph->vertices , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, graph->V * sizeof(cl_uint), sizeof(cl_uint), &graph->E , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, weight_buffer, CL_TRUE,0, graph->E * sizeof(cl_uint), graph->weight,0, NULL, NULL);
    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    //Build Program and create Kernels
    char* filename = "/dijkstra_kernel.cl";
    char cfp[1024];
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s",dirname(cfp),filename);
    filename = kernel_file;
    cl_program program = cluBuildProgramFromFile(context,device,kernel_file,NULL);

    cl_kernel init_kernel = clCreateKernel(program,"initializeBuffers",&err);
    CLU_ERRCHECK(err,"Failed to create initializing_buffers kernel from program");

    cl_kernel dijkstra1_kernel = clCreateKernel(program,"dijkstra1",&err);
    CLU_ERRCHECK(err,"Failed to create bfs baseline kernel from program");

    cl_kernel dijkstra2_kernel = clCreateKernel(program,"dijkstra2",&err);
    CLU_ERRCHECK(err,"Failed to create bfs baseline kernel from program");

    //Set KernelArguments
    cluSetKernelArguments(init_kernel,5,sizeof(cl_mem),&mask_buffer,sizeof(cl_mem),&cost_buffer,sizeof(cl_mem),&update_cost_buffer,sizeof(cl_mem),&semaphore_buffer,sizeof(cl_uint),&source);
    cluSetKernelArguments(dijkstra1_kernel,7,sizeof(cl_mem),(void*)&vertice_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&weight_buffer,sizeof(cl_mem),(void*)&mask_buffer,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),(void*)&update_cost_buffer,sizeof(cl_mem),(void*)&semaphore_buffer);
    cluSetKernelArguments(dijkstra2_kernel,4,sizeof(cl_mem),(void*)&mask_buffer,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),(void*)&update_cost_buffer,sizeof(cl_mem),(void*)&finished_flag);


    // Execute the OpenCL kernel for initializing Buffers
    size_t globalSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, init_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    // start looping both main kernels
    bool finished;

    while(true)
    {
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, dijkstra1_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "Failed to enqueue Dijkstra1 kernel");
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, dijkstra2_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "Failed to enqueue Dijkstra2 kernel");
        err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_bool),&finished,0,NULL,NULL);

        if(finished)
            break;

        finished = true;
        err = clEnqueueWriteBuffer(command_queue, finished_flag, CL_TRUE, 0, sizeof(cl_bool), &finished , 0, NULL, NULL);

    }

    //printf("Time for source node %u parallel Dijkstra: %lu\n",source,total_time);

    /*float* cost_array = dijkstra_serial(graph,source);
    float* cost_parallel = (float*) malloc(sizeof(float) * graph->V);

    err = clEnqueueReadBuffer(command_queue,cost_buffer,CL_TRUE,0,sizeof(cl_float) * graph->V,cost_parallel,0,NULL,NULL);

    for(int i = 0;i<graph->V;i++)
    {
        if(cost_array[i] != cost_parallel[i])
        {
            printf("Wrong Results an Stelle %d: Serial says %f and parallel says %f\n",i,cost_array[i],cost_parallel[i] );
            break;
        }
    }

    free(cost_array);
    free(cost_parallel);*/

    // Clean up
    err = clFlush(command_queue);
    err = clFinish(command_queue);
    err = clReleaseKernel(init_kernel);
    err = clReleaseKernel(dijkstra1_kernel);
    err = clReleaseKernel(dijkstra2_kernel);
    err = clReleaseProgram(program);
    err = clReleaseMemObject(vertice_buffer);
    err = clReleaseMemObject(edge_buffer);
    err = clReleaseMemObject(weight_buffer);
    err = clReleaseMemObject(cost_buffer);
    err = clReleaseMemObject(update_cost_buffer);
    err = clReleaseMemObject(mask_buffer);
    err = clReleaseMemObject(semaphore_buffer);
    err = clReleaseMemObject(finished_flag);
    err = clReleaseCommandQueue(command_queue);
    err = clReleaseContext(context);

    unsigned long total_time = time_ms() - start_time;
    return total_time;
}

