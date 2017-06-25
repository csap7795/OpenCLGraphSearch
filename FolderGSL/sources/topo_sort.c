#include <topo_sort.h>
#include <edge_vertice_message.h>
#include <cl_utils.h>
#include <unistd.h>
#include <libgen.h>
#include <stdio.h>



static cl_program program;
static cl_context context;
static cl_command_queue command_queue;
static cl_device_id device;

static void build_kernel(size_t device_num,int group_num)
{
    device = cluInitDevice(device_num,&context,&command_queue);
    //printf("%s\n\n",cluGetDeviceDescription(device,device_num));

    char* filename = "/topo_sort.cl";
    char tmp[1024];
    sprintf(tmp, "-DGROUP_NUM=%d",group_num);
    char cfp[1024];
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s%s",dirname(dirname(cfp)),"/kernels",filename);
    program = cluBuildProgramFromFile(context,device,kernel_file,tmp);
}

void topological_order_normal(Graph* graph, cl_uint* out_order_parallel,unsigned device_num, unsigned long *time)
{

    int group_num = 1;
    build_kernel(device_num,group_num);

    //start measuring time
    unsigned long start_time = time_ms();

    //Allocate Data
    cl_uint* sourceVerticesSorted = (cl_uint*)malloc(graph->E * sizeof(cl_uint));
    cl_uint* messageWriteIndex = (cl_uint*) malloc(graph->E * sizeof(cl_uint));
    cl_uint* numEdgesSorted = (cl_uint*) calloc(graph->V, sizeof(cl_uint));
    cl_uint* offset = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* oldToNew = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* newToOld = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint messageBuffersize;
    cl_int err;

    //Preprocess Indices and initialize VertexBuffer, Calculate the number of incoming Edges for each vertex and set the sourceVertex of each edge
    serial_without_optimization_preprocess(graph,messageWriteIndex,sourceVerticesSorted,numEdgesSorted,offset,&messageBuffersize);

    cl_bool *messageBuffer = (cl_bool*)calloc(messageBuffersize, sizeof(cl_bool));

    // Create Memory Buffers
    cl_mem inEdges_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating inEdges_buffer");

    cl_mem message_buffer= clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_bool) * messageBuffersize, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating message_buffer");

    cl_mem messageWriteIndex_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating messageWriteIndex_buffer");

    cl_mem sourceVertices_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating sourceVertices_buffer");

    cl_mem  offset_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating offset_buffer");

    cl_mem order_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating order_buffer");

    cl_mem  active_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_bool) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating active_buffer");

    cl_mem finished_flag = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_bool), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating finished_flag");

    // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, message_buffer, CL_FALSE, 0, messageBuffersize * sizeof(cl_bool), messageBuffer , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, messageWriteIndex_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), messageWriteIndex , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, sourceVertices_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), sourceVerticesSorted , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, inEdges_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint), numEdgesSorted , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, offset_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint), offset, 0, NULL, NULL);

    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    // Create the kernels
    cl_kernel init_kernel = clCreateKernel(program,"initialize",&err);
    CLU_ERRCHECK(err,"Failed to create initializing kernel from program");

    cl_kernel edge_kernel = clCreateKernel(program,"edgeCompute",&err);
    CLU_ERRCHECK(err,"Failed to create EdgeCompute kernel from program");

    cl_kernel vertex_kernel = clCreateKernel(program,"vertexCompute",&err);
    CLU_ERRCHECK(err,"Failed to create vertexCompute kernel from program");


    bool finished = true;
    cl_uint current_order = 0;
    //Set KernelArguments
    cluSetKernelArguments(init_kernel,4,sizeof(cl_mem),(void*)&inEdges_buffer,sizeof(cl_mem),(void*)&order_buffer,sizeof(cl_mem),(void*)&active_buffer,sizeof(cl_mem),(void*)&finished_flag);
    cluSetKernelArguments(edge_kernel,5,sizeof(cl_mem),(void*)&inEdges_buffer,sizeof(cl_mem),(void*)&sourceVertices_buffer,sizeof(cl_mem),(void*)&messageWriteIndex_buffer,sizeof(cl_mem),(void*)&message_buffer,sizeof(cl_mem),(void*)&active_buffer);
    cluSetKernelArguments(vertex_kernel,7,sizeof(cl_mem),(void*)&offset_buffer,sizeof(cl_mem),(void*)&message_buffer,sizeof(cl_mem),(void*)&inEdges_buffer,sizeof(cl_mem),(void*)&order_buffer,sizeof(cl_mem),(void*)&active_buffer,sizeof(cl_mem),(void*)&finished_flag, sizeof(cl_uint),&current_order);

    // Execute the OpenCL kernel for initializing Buffers
    size_t globalVertexSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, init_kernel, 1, NULL, &globalVertexSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");
    err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_bool),&finished,0,NULL,NULL);

    // start looping both main kernels
    size_t globalEdgeSize = graph->E;
    while(!finished)
    {

        current_order++;
        clSetKernelArg(vertex_kernel,6,sizeof(cl_uint),&current_order);
        finished = true;

        err = clEnqueueWriteBuffer(command_queue, finished_flag, CL_TRUE, 0, sizeof(cl_bool), &finished , 0, NULL, NULL);
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, edge_kernel, 1, NULL, &globalEdgeSize, NULL, 0, NULL, NULL), "Failed to enqueue Edge_kernel");
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, vertex_kernel, 1, NULL, &globalVertexSize, NULL, 0, NULL, NULL), "Failed to enqueue Vertex_kernel");
        err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_bool),&finished,0,NULL,NULL);

    }

    err = clEnqueueReadBuffer(command_queue,order_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->V,out_order_parallel,0,NULL,NULL);
    CLU_ERRCHECK(err,"Error reading back results in topological_order_normal");


    // Wait for all commands in command_queue to finish.
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    CLU_ERRCHECK(err,"Error finishing command_queue");

    // Save time if asked
    if(time != NULL)
        *time = time_ms()-start_time;

    // Free allocated data
    free(sourceVerticesSorted);
    free(messageBuffer);
    free(messageWriteIndex);
    free(offset);
    free(oldToNew);
    free(newToOld);

    //Clean up
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    err |= clReleaseKernel(init_kernel);
    err |= clReleaseKernel(edge_kernel);
    err |= clReleaseKernel(vertex_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(message_buffer);
    err |= clReleaseMemObject(messageWriteIndex_buffer);
    err |= clReleaseMemObject(inEdges_buffer);
    err |= clReleaseMemObject(sourceVertices_buffer);
    err |= clReleaseMemObject(order_buffer);
    err |= clReleaseMemObject(active_buffer);
    err |= clReleaseMemObject(offset_buffer);
    err |= clReleaseMemObject(finished_flag);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err, "Failed during finalizing OpenCL");
}

void topological_order_opt(Graph* graph, cl_uint* out_order_parallel,unsigned device_num, unsigned long *time,unsigned long *precalc_time)
{
    // start time calculation for preprocessing
    unsigned long start_time = time_ms();
    // Determine the type of the device
    int group_num;
    //Allocate necessary Data
    cl_uint* sourceVerticesSorted = (cl_uint*)malloc(graph->E * sizeof(cl_uint));
    cl_uint* messageWriteIndex = (cl_uint*) malloc(graph->E * sizeof(cl_uint));
    cl_uint* numEdgesSorted = (cl_uint*) calloc(graph->V, sizeof(cl_uint));
    cl_uint* offset = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* oldToNew = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* newToOld = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint messageBuffersize;
    cl_int err;

    group_num = preprocessing_parallel(graph,messageWriteIndex,sourceVerticesSorted,numEdgesSorted,oldToNew,newToOld,offset,&messageBuffersize,device_num);

    if(group_num == 0){
        perror("Device seems to be neither GPU or CPU");
        exit(-1);
    }

    if(precalc_time != NULL)
        *precalc_time = time_ms()-start_time;

    build_kernel(device_num,group_num);

    cl_bool *messageBuffer = (cl_bool*)calloc(messageBuffersize, sizeof(cl_bool));

    // Create Memory Buffers
    cl_mem inEdges_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating inEdges_buffer");

    cl_mem message_buffer= clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_bool) * messageBuffersize, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating message_buffer");

    cl_mem messageWriteIndex_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating messageWriteIndex_buffer");

    cl_mem sourceVertices_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating sourceVertices_buffer");

    cl_mem  offset_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating offset_buffer");

    cl_mem order_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating order_buffer");

    cl_mem  active_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_bool) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating active_buffer");

    cl_mem finished_flag = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_bool), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating finished_flag");

    // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, message_buffer, CL_FALSE, 0, messageBuffersize * sizeof(cl_bool), messageBuffer , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, messageWriteIndex_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), messageWriteIndex , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, sourceVertices_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), sourceVerticesSorted , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, inEdges_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint), numEdgesSorted , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, offset_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint), offset, 0, NULL, NULL);

    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    // Create the kernels
    cl_kernel init_kernel = clCreateKernel(program,"initialize",&err);
    CLU_ERRCHECK(err,"Failed to create initializing kernel from program");

    cl_kernel edge_kernel = clCreateKernel(program,"edgeCompute",&err);
    CLU_ERRCHECK(err,"Failed to create EdgeCompute kernel from program");

    cl_kernel vertex_kernel = clCreateKernel(program,"vertexCompute",&err);
    CLU_ERRCHECK(err,"Failed to create vertexCompute kernel from program");


    bool finished = true;
    cl_uint current_order = 0;
    //Set KernelArguments
    cluSetKernelArguments(init_kernel,4,sizeof(cl_mem),(void*)&inEdges_buffer,sizeof(cl_mem),(void*)&order_buffer,sizeof(cl_mem),(void*)&active_buffer,sizeof(cl_mem),(void*)&finished_flag);
    cluSetKernelArguments(edge_kernel,5,sizeof(cl_mem),(void*)&inEdges_buffer,sizeof(cl_mem),(void*)&sourceVertices_buffer,sizeof(cl_mem),(void*)&messageWriteIndex_buffer,sizeof(cl_mem),(void*)&message_buffer,sizeof(cl_mem),(void*)&active_buffer);
    cluSetKernelArguments(vertex_kernel,7,sizeof(cl_mem),(void*)&offset_buffer,sizeof(cl_mem),(void*)&message_buffer,sizeof(cl_mem),(void*)&inEdges_buffer,sizeof(cl_mem),(void*)&order_buffer,sizeof(cl_mem),(void*)&active_buffer,sizeof(cl_mem),(void*)&finished_flag, sizeof(cl_uint),&current_order);

    // Execute the OpenCL kernel for initializing Buffers
    size_t globalVertexSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, init_kernel, 1, NULL, &globalVertexSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");
    err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_bool),&finished,0,NULL,NULL);

    // start looping both main kernels
    size_t globalEdgeSize = graph->E;
    while(!finished)
    {

        current_order++;
        clSetKernelArg(vertex_kernel,6,sizeof(cl_uint),&current_order);
        finished = true;

        err = clEnqueueWriteBuffer(command_queue, finished_flag, CL_TRUE, 0, sizeof(cl_bool), &finished , 0, NULL, NULL);
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, edge_kernel, 1, NULL, &globalEdgeSize, NULL, 0, NULL, NULL), "Failed to enqueue Edge kernel");
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, vertex_kernel, 1, NULL, &globalVertexSize, NULL, 0, NULL, NULL), "Failed to enqueue Vertex kernel");
        err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_bool),&finished,0,NULL,NULL);

    }

    cl_uint* order_parallel = (cl_uint*) malloc(sizeof(cl_uint) * graph->V);
    err = clEnqueueReadBuffer(command_queue,order_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->V,order_parallel,0,NULL,NULL);

    // Abgleich auf Unsigned max um zu checken ob Topologische Ordnung Existiert.
    for(int i = 0; i<graph->V;i++)
    {
        out_order_parallel[i] = order_parallel[oldToNew[i]];
    }
    free(order_parallel);

    // Wait for all commands in command_queue to finish.
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    CLU_ERRCHECK(err,"Error finishing command_queue");

    // Save time if asked
    if(time != NULL)
        *time = time_ms()-start_time;

    // Free allocated data
    free(sourceVerticesSorted);
    free(messageBuffer);
    free(messageWriteIndex);
    free(offset);
    free(oldToNew);
    free(newToOld);

    //Clean up
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    err |= clReleaseKernel(init_kernel);
    err |= clReleaseKernel(edge_kernel);
    err |= clReleaseKernel(vertex_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(message_buffer);
    err |= clReleaseMemObject(messageWriteIndex_buffer);
    err |= clReleaseMemObject(inEdges_buffer);
    err |= clReleaseMemObject(sourceVertices_buffer);
    err |= clReleaseMemObject(order_buffer);
    err |= clReleaseMemObject(active_buffer);
    err |= clReleaseMemObject(offset_buffer);
    err |= clReleaseMemObject(finished_flag);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err, "Failed during finalizing OpenCL");
}



