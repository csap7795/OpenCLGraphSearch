#include <topo_sort.h>
#include <edge_vertice_message.h>
#include <time_ms.h>
#include <cl_utils.h>

#define GROUP_NUM 32
#define PREPROCESS_ENABLE 1

static cl_program program;
static cl_context context;
static cl_command_queue command_queue;
static cl_device_id device;

static void build_kernel(size_t device_num)
{
    device = cluInitDevice(device_num,&context,&command_queue);
    //printf("%s\n\n",cluGetDeviceDescription(device,device_num));
    char tmp[1024];
    int group_num = PREPROCESS_ENABLE ? GROUP_NUM : 1;
    sprintf(tmp, "-DGROUP_NUM=%d",group_num);
    const char* kernel_file = "/home/chris/Dokumente/OpenCL/CodeBlocks Projekte/GraphSearchLibrary/topo_sort.cl";
    program = cluBuildProgramFromFile(context,device,kernel_file,tmp);
}

void topological_order(Graph* graph, cl_uint* out_order_parallel,unsigned device_num )
{

    build_kernel(device_num);
    //Allocate Data
    cl_uint* sourceVerticesSorted = (cl_uint*)malloc(graph->E * sizeof(cl_uint));
    cl_uint* messageWriteIndex = (cl_uint*) malloc(graph->E * sizeof(cl_uint));
    cl_uint* numEdgesSorted = (cl_uint*) calloc(graph->V, sizeof(cl_uint));
    cl_uint* offset = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* oldToNew = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint messageBuffersize;
    cl_int err;

    //Preprocess Indices and initialize VertexBuffer, Calculate the number of incoming Edges for each vertex and set the sourceVertex of each edge

    unsigned long start_time = time_ms();
    if(PREPROCESS_ENABLE)
        preprocessing_parallel(graph,messageWriteIndex,sourceVerticesSorted,numEdgesSorted,oldToNew,offset,&messageBuffersize,1);
    else
        preprocessing(graph,messageWriteIndex,sourceVerticesSorted,numEdgesSorted,oldToNew,offset,&messageBuffersize);

    unsigned long total_time = time_ms() - start_time;
    //printf("Time for Preprocessing : %lu\n",total_time);


    cl_bool *messageBuffer = (cl_bool*)calloc(messageBuffersize, sizeof(cl_bool));


    cl_mem inEdges_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    // Create Memory Buffers for Helper Objects
    cl_mem message_buffer= clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_bool) * messageBuffersize, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating message_buffer");

    cl_mem messageWriteIndex_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating messageWriteIndex_buffer");

    cl_mem sourceVertices_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating sourceVertices_buffer");

    cl_mem  offset_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating active_buffer");

    cl_mem order_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating cost_buffer");

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

    cl_kernel init_kernel = clCreateKernel(program,"initialize",&err);
    CLU_ERRCHECK(err,"Failed to create initializing kernel from program");

    cl_kernel edge_kernel = clCreateKernel(program,"edgeCompute",&err);
    CLU_ERRCHECK(err,"Failed to create EdgeCompute kernel from program");

    cl_kernel vertex_kernel = clCreateKernel(program,"vertexCompute",&err);
    CLU_ERRCHECK(err,"Failed to create vertexCompute kernel from program");

    //Set KernelArguments
    bool finished = true;
    cl_uint current_order = 0;

    cluSetKernelArguments(init_kernel,4,sizeof(cl_mem),(void*)&inEdges_buffer,sizeof(cl_mem),(void*)&order_buffer,sizeof(cl_mem),(void*)&active_buffer,sizeof(cl_mem),(void*)&finished_flag);
    cluSetKernelArguments(edge_kernel,5,sizeof(cl_mem),(void*)&inEdges_buffer,sizeof(cl_mem),(void*)&sourceVertices_buffer,sizeof(cl_mem),(void*)&messageWriteIndex_buffer,sizeof(cl_mem),(void*)&message_buffer,sizeof(cl_mem),(void*)&active_buffer);
    cluSetKernelArguments(vertex_kernel,7,sizeof(cl_mem),(void*)&offset_buffer,sizeof(cl_mem),(void*)&message_buffer,sizeof(cl_mem),(void*)&inEdges_buffer,sizeof(cl_mem),(void*)&order_buffer,sizeof(cl_mem),(void*)&active_buffer,sizeof(cl_mem),(void*)&finished_flag, sizeof(cl_uint),&current_order);

    // Execute the OpenCL kernel for initializing Buffers
    size_t globalEdgeSize = graph->E;
    size_t globalVertexSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, init_kernel, 1, NULL, &globalVertexSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");
    err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_bool),&finished,0,NULL,NULL);

    // start looping both main kernels
    start_time = time_ms();
    while(!finished)
    {

        current_order++;
        clSetKernelArg(vertex_kernel,6,sizeof(cl_uint),&current_order);
        finished = true;

        err = clEnqueueWriteBuffer(command_queue, finished_flag, CL_TRUE, 0, sizeof(cl_bool), &finished , 0, NULL, NULL);
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, edge_kernel, 1, NULL, &globalEdgeSize, NULL, 0, NULL, NULL), "Failed to enqueue Dijkstra1 kernel");
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, vertex_kernel, 1, NULL, &globalVertexSize, NULL, 0, NULL, NULL), "Failed to enqueue Dijkstra2 kernel");
        err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_bool),&finished,0,NULL,NULL);

    }
    total_time = time_ms() - start_time;

    //printf("Time for Calculating edgeVerticeMessage : %lu\n",total_time);

    cl_uint* order_parallel = (cl_uint*) malloc(sizeof(cl_uint) * graph->V);
    err = clEnqueueReadBuffer(command_queue,order_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->V,order_parallel,0,NULL,NULL);

    // Abgleich auf Unsigned max um zu checken ob Topologische Ordnung Existiert.
    for(int i = 0; i<graph->V;i++)
    {
        out_order_parallel[i] = order_parallel[oldToNew[i]];
    }

    free(order_parallel);

    //Clean up
    free(sourceVerticesSorted);
    free(messageBuffer);
    free(messageWriteIndex);
    free(offset);
    free(oldToNew);

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
}


