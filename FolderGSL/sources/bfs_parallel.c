#include <bfs_parallel.h>

#include <CL/cl.h>
#include <cl_utils.h>
#include <limits.h>
#include <benchmark_utils.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libgen.h>
#include <alloca.h>

// Best for GPU
#define CHUNK_SIZE_GPU 256
#define W_SZ_GPU 32

// Best for CPU
#define CHUNK_SIZE_CPU 512
#define W_SZ_CPU 2

static cl_program program;
static cl_context context;
static cl_command_queue command_queue;
static cl_device_id device;

static void build_program(size_t device_num, cl_device_type device_type)
{
    // Associate the device specified by device_num with a context and a command_queue
    device = cluInitDevice(device_num,&context,&command_queue);
    //printf("%s\n\n",cluGetDeviceDescription(device,device_num));
    int workSize, chunkSize;

    // If device_type is -1, worksize and chunksize are not necessary for the kernel_execution but for the compiling
    // , so they are just set to one
    if(device_type == -1)
    {
        workSize = 1;
        chunkSize = 1;
    }
    // Set worksize and chunksize depending on the type of the device
    else
    {
        switch(device_type){
            case CL_DEVICE_TYPE_GPU :   workSize = W_SZ_GPU; chunkSize = CHUNK_SIZE_GPU;break;
            case CL_DEVICE_TYPE_CPU :   workSize = W_SZ_CPU; chunkSize = CHUNK_SIZE_CPU;break;
            default                 :   workSize = 1; chunkSize = 1;break;
        }
    }

    char tmp[1024];
    sprintf(tmp, "-DW_SZ=%d -DCHUNK_SIZE=%d",workSize, chunkSize);

    char* filename = "/bfs_kernel.cl";
    char cfp[1024];

    /* Create path to the kernel file*/
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s%s",dirname(dirname(cfp)),"/kernels",filename);

    program = cluBuildProgramFromFile(context,device,kernel_file,tmp);
}

/* Outperforms baseline approach 3 times with groupsize 64 vertices 100000 and edges per vertex 1000*/
void bfs_parallel_workgroup(Graph* graph, cl_uint* out_cost, cl_uint* out_path,unsigned source, unsigned device_num, unsigned long *time)
{

    cl_device_type device_type;
    clGetDeviceInfo(device,CL_DEVICE_TYPE,sizeof(cl_device_type),&device_type,NULL);
    build_program(device_num, device_type);

    size_t localSize, globalSize, total;
    unsigned chunkSize;

    //Measure time
    unsigned long start_time = time_ms();

    // Set variables depending on the device type
    switch(device_type){
        case CL_DEVICE_TYPE_GPU :   localSize = W_SZ_GPU;
                                    total = round_up_globalSize(graph->V,CHUNK_SIZE_GPU);
                                    globalSize = total/CHUNK_SIZE_GPU * localSize;
                                    chunkSize = CHUNK_SIZE_GPU;
                                    break;
        case CL_DEVICE_TYPE_CPU :   localSize = W_SZ_CPU;
                                    total = round_up_globalSize(graph->V,CHUNK_SIZE_CPU);
                                    globalSize = total/CHUNK_SIZE_GPU * localSize;
                                    chunkSize = CHUNK_SIZE_CPU;
                                    break;
        default            :   perror("Device Type is neither CPU nor GPU\n");	exit(-1);
    }

    // Create an Array for filling the vertice_buffer if it's size was round up to a multiple of chunkSize
    cl_uint* addbuffer = (cl_uint*)alloca(sizeof(cl_uint) * chunkSize);
    for(int i = 0; i<chunkSize;i++)
        addbuffer[i] = graph->E;

    cl_int err;
    cl_uint current_level = 0;

    // Create Memory Buffers for Graph Data
    cl_mem vertice_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (total+1), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating verticebuffer");

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    // Create Buffers for the results
    cl_mem level_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * total, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating level_buffer");

    cl_mem path_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * total, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating level_buffer");

    // Create a Buffer indicating if there's no active vertex and so the execution can be stopped
    cl_mem finished_flag = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_int), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating finished_flag");

     // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), graph->vertices , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, graph->V * sizeof(cl_uint), sizeof(cl_uint) * (total-graph->V+1), addbuffer, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    /*Create Kernels*/
    cl_kernel init_kernel = clCreateKernel(program,"initialize_bfs_kernel",&err);
    CLU_ERRCHECK(err,"Failed to create initializing bfs kernel from program");

    cl_kernel bfs_workgroup_kernel = clCreateKernel(program,"workgroup_bfs_kernel",&err);
    CLU_ERRCHECK(err,"Failed to create bfs baseline kernel from program");

    // Set Kernel Arguments
    cluSetKernelArguments(init_kernel,4,sizeof(cl_mem),(void*)&level_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(cl_mem),(void*)&finished_flag,sizeof(cl_uint),&source);
    cluSetKernelArguments(bfs_workgroup_kernel,6,sizeof(cl_mem),(void*)&vertice_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&level_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(cl_mem),(void*)&finished_flag,sizeof(cl_uint),&current_level);

    // Execute Kernel for initialization
    size_t init_globalSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, init_kernel, 1, NULL, &init_globalSize, NULL, 0, NULL, NULL), "Failed to enqueue 2D kernel");

    //start looping the main kernel
    cl_int finished;
    for(int i=0; i<graph->V;i++)
    {
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, bfs_workgroup_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL), "Failed to enqueue 2D kernel");
        err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_int),&finished,0,NULL,NULL);

        if(finished == 1)
            break;

        current_level++;
        finished = 1;
        err = clEnqueueWriteBuffer(command_queue, finished_flag, CL_FALSE, 0, sizeof(cl_int), &finished , 0, NULL, NULL);
        clSetKernelArg(bfs_workgroup_kernel,5,sizeof(cl_uint),&current_level);

    }

    // Read back the results
    err = clEnqueueReadBuffer(command_queue,level_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->V,out_cost,0,NULL,NULL);
    err |= clEnqueueReadBuffer(command_queue,path_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->V,out_path,0,NULL,NULL); CLU_ERRCHECK(err,"Error reading back results");
    CLU_ERRCHECK(err,"Error reading back results");

    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    CLU_ERRCHECK(err,"Error finishing command_queue");

    // Save time if asked
    if(time != NULL)
        *time = time_ms()-start_time;

    err = clReleaseKernel(init_kernel);
    err |= clReleaseKernel(bfs_workgroup_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(vertice_buffer);
    err |= clReleaseMemObject(edge_buffer);
    err |= clReleaseMemObject(level_buffer);
    err |= clReleaseMemObject(path_buffer);
    err |= clReleaseMemObject(finished_flag);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err,"Failed finalizing OpenCL");

}

void bfs_parallel_baseline(Graph* graph, cl_uint* out_cost, cl_uint* out_path, unsigned source, unsigned device_num, unsigned long *time)
{
    build_program(device_num,-1);

    //Measure time
    unsigned long start_time = time_ms();

    cl_int err;
    cl_uint current_level = 0;

    // Create Memory Buffers for Graph Data
    cl_mem vertice_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (graph->V+1), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating vertice_buffer");

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edge_buffer");

    // Create Memory Buffers for reading back results
    cl_mem level_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->V, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating level_buffer");

    cl_mem path_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * graph->V, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating path_buffer");

    // Create a Buffer indicating if there's no active vertex and so the execution can be stopped
    cl_mem finished_flag = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_int), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating finished_flag");


    // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), graph->vertices , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, graph->V * sizeof(cl_uint), sizeof(cl_uint), &graph->E , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    // Create Kernels
    cl_kernel init_kernel = clCreateKernel(program,"initialize_bfs_kernel",&err);
    CLU_ERRCHECK(err,"Failed to create initialize_bfs_kernel from program");

    cl_kernel bfs_kernel = clCreateKernel(program,"baseline_kernel",&err);
    CLU_ERRCHECK(err,"Failed to create baseline_kernel from program");

    // Set Kernel Arguments
    cluSetKernelArguments(init_kernel,4,sizeof(cl_mem),(void*)&level_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(cl_mem),(void*)&finished_flag,sizeof(cl_uint),&source);
    cluSetKernelArguments(bfs_kernel,6,sizeof(cl_mem),(void*)&vertice_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&level_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(cl_mem),(void*)&finished_flag,sizeof(cl_uint),&current_level);

    // Execute Kernels
    size_t globalSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, init_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "Failed to enqueue 2D kernel");

    //start looping the main kernel
    cl_int finished;
    int i;
    for(i=0; i<graph->V;i++)
    {
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, bfs_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "Failed to enqueue 2D kernel");
        err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_int),&finished,0,NULL,NULL);

        if(finished)
            break;

        current_level++;
        finished = 1;
        err = clEnqueueWriteBuffer(command_queue, finished_flag, CL_FALSE, 0, sizeof(cl_int), &finished , 0, NULL, NULL);
        clSetKernelArg(bfs_kernel,5,sizeof(cl_uint),&current_level);

    }
    //printf("Stages: %d\n",i);

    // Read out data from the buffers
    err = clEnqueueReadBuffer(command_queue,level_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->V,out_cost,0,NULL,NULL);
    err |= clEnqueueReadBuffer(command_queue,path_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->V,out_path,0,NULL,NULL);
    CLU_ERRCHECK(err,"Error reading back results");

    // Wait until all queued commands finish
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    CLU_ERRCHECK(err,"Error finishing command_queue");

    // Save time if asked
    if(time != NULL)
        *time = time_ms()-start_time;

    //finalize
    err = clReleaseKernel(init_kernel);
    err |= clReleaseKernel(bfs_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(vertice_buffer);
    err |= clReleaseMemObject(edge_buffer);
    err |= clReleaseMemObject(level_buffer);
    err |= clReleaseMemObject(path_buffer);
    err |= clReleaseMemObject(finished_flag);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err,"Failed finalizing OpenCL");
}

void bfs_logical_frontier_plot(Graph* graph,unsigned source, unsigned device_num)
{
    build_program(device_num,-1);

    cl_int err;
    cl_uint current_level = 0;

    // Create Memory Buffers for Graph Data
    cl_mem vertice_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (graph->V+1), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating vertice_buffer");

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edge_buffer");

    // Create Memory Buffers for reading back results
    cl_mem frontier_vertex_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_int), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating level_buffer");

    cl_mem frontier_edge_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_int) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating path_buffer");

    // Create Memory Buffers for reading back results
    cl_mem level_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->V, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating level_buffer");

    // Create a Buffer indicating if there's no active vertex and so the execution can be stopped
    cl_mem finished_flag = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_int), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating finished_flag");

    // Copy Graph Data to their respective memory buffers
    cl_int* zeroes = (cl_int*)calloc(graph->E,sizeof(cl_int));
    cl_uint* uintmax = (cl_uint*)malloc(graph->V*sizeof(cl_uint));
    for(int i = 0 ; i<graph->V;i++)
        uintmax[i] = CL_UINT_MAX;

    uintmax[source] = 0;
    err = clEnqueueWriteBuffer(command_queue, level_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), uintmax , 0, NULL, NULL);

    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, 0, (graph->V+1) * sizeof(cl_uint), graph->vertices, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, frontier_vertex_buffer, CL_FALSE, 0, sizeof(cl_int), zeroes , 0, NULL, NULL);err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, frontier_edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_int), zeroes , 0, NULL, NULL);

    CLU_ERRCHECK(err,"Failed copying graph data to buffers");err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);


    cl_kernel frontier_kernel = clCreateKernel(program,"frontier_plot",&err);
    CLU_ERRCHECK(err,"Failed to create baseline_kernel from program");

    // Set Kernel Arguments
    cluSetKernelArguments(frontier_kernel,7,sizeof(cl_mem),(void*)&vertice_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&frontier_vertex_buffer,sizeof(cl_mem),(void*)&frontier_edge_buffer,sizeof(cl_mem),(void*)&finished_flag,sizeof(cl_uint),&current_level,sizeof(cl_mem),&level_buffer);

    // Execute Kernels
    size_t globalSize = graph->V;
    //CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, frontier_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "Failed to enqueue 2D kernel");

    //start looping the main kernel
    cl_int finished;
    cl_int vertex_frontier;
    cl_int* edge_frontier = (cl_int*)malloc(sizeof(cl_int) * graph->E);
    int unique_edges=0;
    int all_edges=0;
    int i;
    for(i=0; i<graph->V;i++)
    {
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, frontier_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "Failed to enqueue 2D kernel");
        err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_int),&finished,0,NULL,NULL);
        err = clEnqueueReadBuffer(command_queue,frontier_vertex_buffer,CL_TRUE,0,sizeof(cl_int),&vertex_frontier,0,NULL,NULL);
        err = clEnqueueReadBuffer(command_queue,frontier_edge_buffer,CL_TRUE,0,sizeof(cl_int) * graph->E,edge_frontier,0,NULL,NULL);
        for(int k = 0; k<graph->E;k++)
        {
            if(edge_frontier[k] > 0){
                unique_edges++;
                all_edges += edge_frontier[k];}
        }

        printf("Step:%d\t\tFV:%d\t\tUE:%d\t\tAE:%d\n",current_level,vertex_frontier,unique_edges,all_edges);

        if(finished)
            break;

        current_level++;
        finished = 1;
        err = clEnqueueWriteBuffer(command_queue, finished_flag, CL_TRUE, 0, sizeof(cl_int), &finished , 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue, frontier_vertex_buffer, CL_TRUE, 0, sizeof(cl_int), zeroes , 0, NULL, NULL);err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue, frontier_edge_buffer, CL_TRUE, 0, graph->E * sizeof(cl_int), zeroes , 0, NULL, NULL);
        unique_edges = 0;
        all_edges = 0;
        clSetKernelArg(frontier_kernel,5,sizeof(cl_uint),&current_level);


    }

    // Wait until all queued commands finish
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    CLU_ERRCHECK(err,"Error finishing command_queue");

    free(zeroes);
    free(edge_frontier);
    free(uintmax);

    //finalize
    err = clReleaseKernel(frontier_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(vertice_buffer);
    err |= clReleaseMemObject(edge_buffer);
    err |= clReleaseMemObject(frontier_edge_buffer);
    err |= clReleaseMemObject(frontier_vertex_buffer);
    err |= clReleaseMemObject(level_buffer);
    err |= clReleaseMemObject(finished_flag);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err,"Failed finalizing OpenCL");
}


