#include "bfs_parallel.h"
#include <stdio.h>
#include "cl_utils.h"
#include <CL/cl.h>
#include <stdbool.h>
#include <limits.h>
#include "time_ms.h"

#define KERNEL_FILENAME "/home/chris/Dokumente/OpenCL/CodeBlocks Projekte/GraphSearchLibrary/bfs_kernel.cl"
#define CL_DEVICE 1

#define CHUNK_SIZE 512
#define W_SZ 4

unsigned round_up_globalSize(unsigned V)
{
    if(V%CHUNK_SIZE == 0)
        return V;

    else return CHUNK_SIZE*((V/CHUNK_SIZE)+1);
}

void bfs_parallel_gpu_workgroup(Graph* graph, unsigned source, size_t group_size)
{
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device = cluInitDevice(CL_DEVICE,&context,&command_queue);

    cl_int err;
    cl_uint current_level = 0;

    printf("%s\n",cluGetDeviceDescription(device,CL_DEVICE));

    // Round up globalSize to get a multiple of CHUNK_SIZE
    unsigned total = round_up_globalSize(graph->V);
    cl_uint addbuffer[CHUNK_SIZE] ={ [0 ... CHUNK_SIZE-1] = graph->E};

    cl_mem vertice_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (total+1), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating verticebuffer");

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    cl_mem level_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * total, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating level_buffer");

    cl_mem finished_flag = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_bool), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating finished_flag");

     // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), graph->vertices , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, graph->V * sizeof(cl_uint), sizeof(cl_uint) * (total-graph->V+1), addbuffer, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_TRUE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    // Build Program and create Kernels
    char tmp[1024];
    sprintf(tmp, "-DW_SZ=%u -DCHUNK_SIZE=%i",(unsigned)group_size, CHUNK_SIZE);
    cl_program program = cluBuildProgramFromFile(context,device,KERNEL_FILENAME,tmp);

    cl_kernel init_kernel = clCreateKernel(program,"initialize_bfs_kernel",&err);
    CLU_ERRCHECK(err,"Failed to create initializing bfs kernel from program");

    cl_kernel bfs_workgroup_kernel = clCreateKernel(program,"workgroup_bfs_kernel",&err);
    CLU_ERRCHECK(err,"Failed to create bfs baseline kernel from program");

    // Set Kernel Arguments
    cluSetKernelArguments(init_kernel,3,sizeof(cl_mem),(void*)&level_buffer,sizeof(cl_mem),(void*)&finished_flag,sizeof(cl_uint),&source);
    cluSetKernelArguments(bfs_workgroup_kernel,5,sizeof(cl_mem),(void*)&vertice_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&level_buffer,sizeof(cl_mem),(void*)&finished_flag,sizeof(cl_uint),&current_level);

    // Execute Kernels
    size_t globalSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, init_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "Failed to enqueue 2D kernel");

    //start looping the main kernel
    globalSize = total/CHUNK_SIZE * group_size;
    size_t localSize = group_size;
    bool finished;
    cl_uint* levels = (cl_uint*)malloc(sizeof(cl_uint)*graph->V);

    unsigned long start_time = time_ms();
    while(true)
    {
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, bfs_workgroup_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL), "Failed to enqueue 2D kernel");
        err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_bool),&finished,0,NULL,NULL);

        if(finished)
            break;

        current_level++;
        finished = true;
        err = clEnqueueWriteBuffer(command_queue, finished_flag, CL_TRUE, 0, sizeof(cl_bool), &finished , 0, NULL, NULL);
        clSetKernelArg(bfs_workgroup_kernel,4,sizeof(cl_uint),&current_level);

    }
    unsigned long total_time = time_ms()-start_time;
    err = clEnqueueReadBuffer(command_queue,level_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->V,levels,0,NULL,NULL);
    unsigned max = 0;
    for(int i = 0; i<graph->V;i++)
    {
        if(levels[i]>max && levels[i] != INT_MAX)
            max = levels[i];
    }
    printf("GroupSize : %u\n",(unsigned) group_size);
    printf("Time for source node %u workgroupapproach with Diameter %u: %lu\n\n",source,max,total_time);
    free(levels);

    //finalize
    err = clFinish(command_queue);
    err |= clReleaseKernel(init_kernel);
    err |= clReleaseKernel(bfs_workgroup_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(vertice_buffer);
    err |= clReleaseMemObject(edge_buffer);
    err |= clReleaseMemObject(level_buffer);
    err |= clReleaseMemObject(finished_flag);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err,"Failed finalizing OpenCL")



}

void bfs_parallel_gpu_baseline(Graph* graph, unsigned source)
{
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device = cluInitDevice(CL_DEVICE,&context,&command_queue);

    cl_int err;
    cl_uint current_level = 0;

    printf("%s\n",cluGetDeviceDescription(device,CL_DEVICE));

    // Create Memory Buffers for Graph Data
    cl_mem vertice_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (graph->V+1), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating verticebuffer");

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    cl_mem level_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->V, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating level_buffer");

    cl_mem finished_flag = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_bool), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating finished_flag");


    // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), graph->vertices , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, graph->V * sizeof(cl_uint), sizeof(cl_uint), &graph->E , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_TRUE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    // Build Program and create Kernels
    char tmp[1024];
    sprintf(tmp, "-DW_SZ=%i -DCHUNK_SIZE=%i",W_SZ, CHUNK_SIZE);
    const char* kernel_file = "/home/chris/Dokumente/OpenCL/CodeBlocks Projekte/GraphSearchLibrary/bfs_kernel.cl";
    cl_program program = cluBuildProgramFromFile(context,device,kernel_file,tmp);

    cl_kernel init_kernel = clCreateKernel(program,"initialize_bfs_kernel",&err);
    CLU_ERRCHECK(err,"Failed to create initializing bfs kernel from program");

    cl_kernel bfs_kernel = clCreateKernel(program,"bfs_kernel",&err);
    CLU_ERRCHECK(err,"Failed to create bfs baseline kernel from program");

    // Set Kernel Arguments
    cluSetKernelArguments(init_kernel,3,sizeof(cl_mem),(void*)&level_buffer,sizeof(cl_mem),(void*)&finished_flag,sizeof(cl_uint),&source);
    cluSetKernelArguments(bfs_kernel,5,sizeof(cl_mem),(void*)&vertice_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&level_buffer,sizeof(cl_mem),(void*)&finished_flag,sizeof(cl_uint),&current_level);

    // Execute Kernels
    size_t globalSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, init_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "Failed to enqueue 2D kernel");

    //start looping the main kernel
    bool finished;
    cl_uint* levels = (cl_uint*)malloc(sizeof(cl_uint)*graph->V);

    unsigned long start_time = time_ms();
    while(true)
    {
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, bfs_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "Failed to enqueue 2D kernel");
        err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_bool),&finished,0,NULL,NULL);

        if(finished)
            break;

        current_level++;
        finished = true;
        err = clEnqueueWriteBuffer(command_queue, finished_flag, CL_TRUE, 0, sizeof(cl_bool), &finished , 0, NULL, NULL);
        clSetKernelArg(bfs_kernel,4,sizeof(cl_uint),&current_level);

    }
    unsigned long total_time = time_ms() - start_time;
    err = clEnqueueReadBuffer(command_queue,level_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->V,levels,0,NULL,NULL);
    int max = 0;
    for(int i = 0; i<graph->V;i++)
    {
        if(levels[i]>max && levels[i] != INT_MAX)
            max = levels[i];
    }

    printf("Time for source node %u baseline approach and Diameter %d: %lu\n",source,max,total_time);
    free(levels);

    //finalize
    err = clFinish(command_queue);
    err |= clReleaseKernel(init_kernel);
    err |= clReleaseKernel(bfs_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(vertice_buffer);
    err |= clReleaseMemObject(edge_buffer);
    err |= clReleaseMemObject(level_buffer);
    err |= clReleaseMemObject(finished_flag);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err,"Failed finalizing OpenCL")
}

