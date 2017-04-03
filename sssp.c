#include <sssp.h>
#include <CL/cl.h>
#include <stdio.h>
#include <cl_utils.h>
#include <time_ms.h>
#include <edge_vertice_message.h>
#include <float.h>
#include <dijkstra_serial.h>
#include <unistd.h>
#include <libgen.h>

#define GROUP_NUM 32

static cl_program program;
static cl_context context;
static cl_command_queue command_queue;
static cl_device_id device;

static void build_kernel(size_t device_num)
{
    device = cluInitDevice(device_num,&context,&command_queue);
    //printf("%s\n\n",cluGetDeviceDescription(device,device_num));
    cl_device_type device_type;
    clGetDeviceInfo(device,CL_DEVICE_TYPE,sizeof(cl_device_type),&device_type,NULL);
    int group_num;
    (device_type == CL_DEVICE_TYPE_GPU) ? (group_num = GROUP_NUM) : (group_num = 1);

    char* filename = "/sssp.cl";
    char tmp[1024];
    sprintf(tmp, "-DGROUP_NUM=%d",group_num);
    char cfp[1024];
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s",dirname(cfp),filename);
    filename = kernel_file;
    printf("%s\n",kernel_file);

    program = cluBuildProgramFromFile(context,device,kernel_file,tmp);
}

unsigned long sssp(Graph* graph, unsigned source,unsigned device_num )
{

    unsigned long start_time = time_ms();

    build_kernel(device_num);

    cl_device_type device_type;
    clGetDeviceInfo(device,CL_DEVICE_TYPE,sizeof(cl_device_type),&device_type,NULL);

  	//Allocate Data
    cl_uint* sourceVerticesSorted = (cl_uint*)malloc(graph->E * sizeof(cl_uint));
    cl_uint* messageWriteIndex = (cl_uint*) malloc(graph->E * sizeof(cl_uint));
    cl_uint* numEdgesSorted = (cl_uint*) calloc(graph->V, sizeof(cl_uint));
    cl_uint* offset = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* oldToNew = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint messageBuffersize;
    cl_int err;

    //Preprocess Indices and initialize VertexBuffer, Calculate the number of incoming Edges for each vertex and set the sourceVertex of each edge

    if(device_type == CL_DEVICE_TYPE_GPU)
        preprocessing_parallel(graph,messageWriteIndex,sourceVerticesSorted,numEdgesSorted,oldToNew,offset,&messageBuffersize,1);
    else
        preprocessing_parallel_cpu(graph,messageWriteIndex,sourceVerticesSorted,numEdgesSorted,oldToNew,offset,&messageBuffersize,1);

    //unsigned long total_time = time_ms() - start_time;
    //printf("Time for Preprocessing : %lu\n",total_time);


    cl_float* messageBuffer = (cl_float*) malloc(messageBuffersize * sizeof(cl_float));
    for(int i = 0; i<messageBuffersize;i++){
        messageBuffer[i] = FLT_MAX;
    }

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    cl_mem weight_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating weight_buffer");

    // Create Memory Buffers for Helper Objects
    cl_mem message_buffer= clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * messageBuffersize, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating message_buffer");

    cl_mem messageWriteIndex_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating messageWriteIndex_buffer");

    cl_mem sourceVertices_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating sourceVertices_buffer");

    cl_mem  numEdges_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->V, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating numEdges_buffer");

    cl_mem cost_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating cost_buffer");

    cl_mem  active_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_bool) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating active_buffer");

    cl_mem  offset_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating active_buffer");

    cl_mem finished_flag = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_bool), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating finished_flag");

    // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, weight_buffer, CL_FALSE,0, graph->E * sizeof(cl_uint), graph->weight,0, NULL, NULL);

    err = clEnqueueWriteBuffer(command_queue, message_buffer, CL_FALSE, 0, messageBuffersize * sizeof(cl_float), messageBuffer , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, messageWriteIndex_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), messageWriteIndex , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, sourceVertices_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), sourceVerticesSorted , 0, NULL, NULL);

    err = clEnqueueWriteBuffer(command_queue, numEdges_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), numEdgesSorted , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, offset_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint), offset , 0, NULL, NULL);

    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    cl_kernel init_kernel = clCreateKernel(program,"initialize",&err);
    CLU_ERRCHECK(err,"Failed to create initializing kernel from program");

    cl_kernel edge_kernel = clCreateKernel(program,"edgeCompute",&err);
    CLU_ERRCHECK(err,"Failed to create EdgeCompute kernel from program");

    cl_kernel vertex_kernel = clCreateKernel(program,"vertexCompute",&err);
    CLU_ERRCHECK(err,"Failed to create vertexCompute kernel from program");

    //Set KernelArguments
    cluSetKernelArguments(init_kernel,3,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),(void*)&active_buffer, sizeof(unsigned), &oldToNew[source]);
    cluSetKernelArguments(edge_kernel,7,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&sourceVertices_buffer,sizeof(cl_mem),(void*)&messageWriteIndex_buffer,sizeof(cl_mem),(void*)&message_buffer,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),(void*)&weight_buffer,sizeof(cl_mem),(void*)&active_buffer,sizeof(cl_mem));
    cluSetKernelArguments(vertex_kernel,6,sizeof(cl_mem),(void*)&offset_buffer,sizeof(cl_mem),(void*)&message_buffer,sizeof(cl_mem),(void*)&numEdges_buffer,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),(void*)&active_buffer,sizeof(cl_mem),(void*)&finished_flag);

    // Execute the OpenCL kernel for initializing Buffers
    size_t globalEdgeSize = graph->E;
    size_t globalVertexSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, init_kernel, 1, NULL, &globalVertexSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    // start looping both main kernels
    bool finished;

    while(true)
    {
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, edge_kernel, 1, NULL, &globalEdgeSize, NULL, 0, NULL, NULL), "Failed to enqueue Dijkstra1 kernel");
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, vertex_kernel, 1, NULL, &globalVertexSize, NULL, 0, NULL, NULL), "Failed to enqueue Dijkstra2 kernel");
        err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_bool),&finished,0,NULL,NULL);

        if(finished)
            break;

        finished = true;
        err = clEnqueueWriteBuffer(command_queue, finished_flag, CL_TRUE, 0, sizeof(cl_bool), &finished , 0, NULL, NULL);

    }

    //printf("Time for Calculating edgeVerticeMessage : %lu\n",total_time);

    float* cost_parallel = (float*) malloc(sizeof(float) * graph->V);
    float* cost_parallel_ordered = (float*) malloc(sizeof(float) * graph->V);
    err = clEnqueueReadBuffer(command_queue,cost_buffer,CL_TRUE,0,sizeof(cl_float) * graph->V,cost_parallel,0,NULL,NULL);

    for(int i = 0; i<graph->V;i++)
    {
        cost_parallel_ordered[i] = cost_parallel[oldToNew[i]];
    }

    //Just to check if the results are equal
    /*float* cost_serial = dijkstra_serial(graph,source);

    for(int i = 0;i<graph->V;i++)
    {
        if(cost_serial[i] != cost_parallel_ordered[i])
        {
            printf("Wrong Results an Stelle %d: Serial says %f and parallel says %f\n",i,cost_serial[i],cost_parallel_ordered[i] );
            break;
        }
    }


    free(cost_serial);*/

    free(cost_parallel);
    free(cost_parallel_ordered);


    //Clean up
    free(sourceVerticesSorted);
    free(messageBuffer);
    free(messageWriteIndex);
    free(offset);
    free(oldToNew);

    err = clFlush(command_queue);
    err = clFinish(command_queue);
    err = clReleaseKernel(init_kernel);
    err = clReleaseKernel(edge_kernel);
    err = clReleaseKernel(vertex_kernel);
    err = clReleaseProgram(program);
    err = clReleaseMemObject(edge_buffer);
    err = clReleaseMemObject(weight_buffer);
    err = clReleaseMemObject(message_buffer);
    err = clReleaseMemObject(messageWriteIndex_buffer);
    err = clReleaseMemObject(numEdges_buffer);
    err = clReleaseMemObject(sourceVertices_buffer);
    err = clReleaseMemObject(cost_buffer);
    err = clReleaseMemObject(active_buffer);
    err = clReleaseMemObject(offset_buffer);
    err = clReleaseMemObject(finished_flag);
    err = clReleaseCommandQueue(command_queue);
    err = clReleaseContext(context);

    unsigned long total_time = time_ms() - start_time;
    return total_time;
}

