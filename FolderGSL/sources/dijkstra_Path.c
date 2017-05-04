#include <dikstra_path.h>
#include <stdio.h>
#include <cl_utils.h>
#include <time_ms.h>
#include <edge_vertice_message.h>
#include <float.h>
#include <unistd.h>
#include <libgen.h>

static cl_program program;
static cl_context context;
static cl_command_queue command_queue;
static cl_device_id device;

static void build_kernel(size_t device_num)
{
    device = cluInitDevice(device_num,&context,&command_queue);

    char* filename = "/dijkstra_path.cl";
    char cfp[1024];
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s%s",dirname(dirname(cfp)),"/kernels",filename);
    program = cluBuildProgramFromFile(context,device,kernel_file,NULL);
}

void dijkstra_path(Graph* graph,unsigned source,cl_float* out_cost, cl_uint* out_path, unsigned device_num, unsigned long *time)
{
    build_kernel(device_num);

    unsigned long start_time = time_ms();

  	//Allocate Data
    cl_uint* sourceVertices = (cl_uint*)malloc(graph->E * sizeof(cl_uint));
    cl_uint* messageWriteIndex = (cl_uint*) malloc(graph->E * sizeof(cl_uint));
    cl_uint* inEdges = (cl_uint*) calloc(graph->V, sizeof(cl_uint));
    cl_uint* offset = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_int err;

    dijkstra_path_preprocess(graph,messageWriteIndex,sourceVertices,inEdges,offset);

    cl_float* messageBuffer_cost = (cl_float*) malloc(graph->E * sizeof(cl_float));
    cl_uint* messageBuffer_path = (cl_uint*) malloc(graph->E * sizeof(cl_uint));
    for(int i = 0; i<graph->E;i++){
        messageBuffer_cost[i] = CL_FLT_MAX;
    }

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    cl_mem weight_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating weight_buffer");

    // Create Memory Buffers for Helper Objects
    cl_mem message_buffer_cost= clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating message_buffer");

    cl_mem message_buffer_path= clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating message_buffer");

    cl_mem messageWriteIndex_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating messageWriteIndex_buffer");

    cl_mem sourceVertices_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating sourceVertices_buffer");

    cl_mem inEdges_buffer= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->V, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating inEdges_buffer");

    cl_mem cost_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating cost_buffer");

    cl_mem path_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating path_buffer");

    cl_mem  active_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_bool) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating active_buffer");

    cl_mem  offset_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating active_buffer");

    cl_mem finished_flag = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_bool), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating finished_flag");

    // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, weight_buffer, CL_FALSE,0, graph->E * sizeof(cl_uint), graph->weight,0, NULL, NULL);

    err = clEnqueueWriteBuffer(command_queue, message_buffer_cost, CL_FALSE, 0, graph->E * sizeof(cl_float), messageBuffer_cost , 0, NULL, NULL);
    //err = clEnqueueWriteBuffer(command_queue, message_buffer_path, CL_FALSE, 0, graph->E * sizeof(cl_uint), messageBuffer_path , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, messageWriteIndex_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), messageWriteIndex , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, sourceVertices_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), sourceVertices , 0, NULL, NULL);

    err = clEnqueueWriteBuffer(command_queue, inEdges_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), inEdges , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, offset_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint), offset , 0, NULL, NULL);

    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    cl_kernel init_kernel = clCreateKernel(program,"initialize",&err);
    CLU_ERRCHECK(err,"Failed to create initializing kernel from program");

    cl_kernel edge_kernel = clCreateKernel(program,"edgeCompute",&err);
    CLU_ERRCHECK(err,"Failed to create EdgeCompute kernel from program");

    cl_kernel vertex_kernel = clCreateKernel(program,"vertexCompute",&err);
    CLU_ERRCHECK(err,"Failed to create vertexCompute kernel from program");

    //Set KernelArguments
    cluSetKernelArguments(init_kernel,4,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(cl_mem),(void*)&active_buffer,sizeof(unsigned),&source);
    cluSetKernelArguments(edge_kernel,8,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&sourceVertices_buffer,sizeof(cl_mem),(void*)&messageWriteIndex_buffer,sizeof(cl_mem),(void*)&message_buffer_cost,sizeof(cl_mem),(void*)&message_buffer_path,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),(void*)&weight_buffer,sizeof(cl_mem),(void*)&active_buffer,sizeof(cl_mem));
    cluSetKernelArguments(vertex_kernel,8,sizeof(cl_mem),(void*)&offset_buffer,sizeof(cl_mem),(void*)&message_buffer_cost,sizeof(cl_mem),(void*)&message_buffer_path,sizeof(cl_mem),(void*)&inEdges_buffer,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(cl_mem),(void*)&active_buffer,sizeof(cl_mem),(void*)&finished_flag);

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

    err = clEnqueueReadBuffer(command_queue,cost_buffer,CL_FALSE,0,sizeof(cl_float) * graph->V,out_cost,0,NULL,NULL);
    err = clEnqueueReadBuffer(command_queue,path_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->V,out_path,0,NULL,NULL);

    //Clean up
    free(sourceVertices);
    free(messageBuffer_cost);
    free(messageBuffer_path);
    free(messageWriteIndex);
    free(offset);
    free(inEdges);

    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    err |= clReleaseKernel(init_kernel);
    err |= clReleaseKernel(edge_kernel);
    err |= clReleaseKernel(vertex_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(edge_buffer);
    err |= clReleaseMemObject(weight_buffer);
    err |= clReleaseMemObject(message_buffer_cost);
    err |= clReleaseMemObject(message_buffer_path);
    err |= clReleaseMemObject(messageWriteIndex_buffer);
    err |= clReleaseMemObject(inEdges_buffer);
    err |= clReleaseMemObject(sourceVertices_buffer);
    err |= clReleaseMemObject(cost_buffer);
    err |= clReleaseMemObject(path_buffer);
    err |= clReleaseMemObject(active_buffer);
    err |= clReleaseMemObject(offset_buffer);
    err |= clReleaseMemObject(finished_flag);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    if(time != NULL)
        *time = time_ms()-start_time;

}


void dijkstra_path_preprocess(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVertices, cl_uint* inEdges, cl_uint* offset)
{
        /* Fill sourceVertice array and numEdges which is the number of every incoming edges for every vertice*/
        for(int i = 0; i<graph->V;i++)
        {
            for(int j = graph->vertices[i]; j<graph->vertices[i+1];j++)
            {
                sourceVertices[j] = i;
                inEdges[graph->edges[j]]++;
            }
        }

        //Calculate offsets, scan over inEdges
        offset[0] = 0;
        for(int i=0; i<graph->V-1;i++)
            offset[i+1] = offset[i] + inEdges[i];


        // Calculate the Write Indices for the Edges
        unsigned* helper = (unsigned*)calloc(graph->V,sizeof(unsigned));
        for(int i = 0; i<graph->E;i++)
        {
            cl_uint dest = graph->edges[i];
            messageWriteIndex[i] = offset[dest] + helper[dest];
            helper[dest]++;
        }
        free(helper);
}
