#include <graph_transpose.h>
#include <graph.h>
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include <cl_utils.h>
#include <unistd.h>
#include <libgen.h>


static cl_program program;
static cl_context context;
static cl_command_queue command_queue;
static cl_device_id device;

static void build_kernel(size_t device_num)
{
    device = cluInitDevice(device_num,&context,&command_queue);
    //printf("%s\n\n",cluGetDeviceDescription(device,device_num));
    char* filename = "/transpose.cl";
    char cfp[1024];
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s%s",dirname(dirname(cfp)),"/kernels",filename);
    filename = kernel_file;
    program = cluBuildProgramFromFile(context,device,kernel_file,NULL);
}

Graph* transpose_serial(Graph *graph, unsigned long *time)
{

    // measure time
    unsigned long start_time = time_ms();

    Graph* transposed = getEmptyGraph(graph->V, graph->E);

    // allocate necessary data, inEdges[i] saves number of incoming edges into source node i
    // helper is used for
    unsigned *inEdges = (unsigned*)calloc(graph->V,sizeof(unsigned));
    unsigned *helper = (unsigned*)calloc(graph->V,sizeof(unsigned));

    //Calculate number of InEdges
    for(int i = 0; i<graph->E;i++)
    {
        inEdges[graph->edges[i]]++;
    }

    // Save the new offsets in the transposed graph
    transposed->vertices[0] = 0;
    for(int i = 1; i<=graph->V;i++)
    {
        transposed->vertices[i] = transposed->vertices[i-1] + inEdges[i-1];
    }

    // Iterate over all edges, determine source node and save the current edge in the transposed graph on the right position
    for(int i = 0; i<graph->V;i++)
    {
        for(int j = graph->vertices[i]; j<graph->vertices[i+1]; j++)
        {
            unsigned source = graph->edges[j];
            unsigned write_index = transposed->vertices[source] + helper[source];
            helper[source]++;
            transposed->edges[write_index] = i;
            transposed->weight[write_index] = graph->weight[j];
        }
    }

    // free allocated data
    free(inEdges);
    free(helper);

    // save execution time if asked
    if(time != NULL)
        *time = time_ms()-start_time;

    return transposed;
}

Graph* transpose_parallel(Graph* graph,size_t device, unsigned long *time)
{
    // build the kernel
    build_kernel(device);

    // start measuring time
    unsigned long start_time = time_ms();

    // create the Graph where the transposal of graph is saved
    Graph* transposed = getEmptyGraph(graph->V, graph->E);

    //allocate space for saving the incoming edges of each source node
    cl_uint* inEdges = (cl_uint*) calloc(transposed->V, sizeof(cl_uint));

    cl_int err;

    //Create Buffers for inEdges Kernel
    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    cl_mem inEdges_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    //Create Buffers for Transpose Kernel
    cl_mem vertice_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (graph->V+1), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating vertice buffer");

    cl_mem weight_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(float) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating weight buffer");

    cl_mem offset_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating offset buffer");

    //Create Buffers for inEdges Kernel
    cl_mem new_edge_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    cl_mem new_weight_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_float) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, 0, (graph->V+1) * sizeof(cl_uint), graph->vertices , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, weight_buffer, CL_FALSE,0, graph->E * sizeof(cl_float), graph->weight,0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, inEdges_buffer, CL_FALSE,0, graph->V * sizeof(cl_uint), inEdges,0, NULL, NULL);
    CLU_ERRCHECK(err,"Failed copying graphdata to buffers");

    // Create kernels
    cl_kernel inEdges_kernel = clCreateKernel(program,"calcInEdges",&err);
    CLU_ERRCHECK(err,"Failed to create inEdges_kernel from program");

    cl_kernel transpose_kernel = clCreateKernel(program,"transpose",&err);
    CLU_ERRCHECK(err,"Failed to create transpose_kernel from program");

    cluSetKernelArguments(inEdges_kernel,2,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&inEdges_buffer);
    cluSetKernelArguments(transpose_kernel,6,sizeof(cl_mem),(void*)&vertice_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&weight_buffer,sizeof(cl_mem),(void*)&offset_buffer, sizeof(cl_mem),(void*)&new_edge_buffer,sizeof(cl_mem),(void*)&new_weight_buffer);

    // enqueue kernel to calculate the incoming edges and wait for the result
    size_t globalEdgeSize = graph->E;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, inEdges_kernel, 1, NULL, &globalEdgeSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");
    err = clEnqueueReadBuffer(command_queue,inEdges_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->V,inEdges,0,NULL,NULL);
    CLU_ERRCHECK(err,"Error reading back results of inEdges_kernel");

    //Calculate new VerticeArray
    transposed->vertices[0] = 0;
    for(int i = 1; i<transposed->V;i++)
    {
        transposed->vertices[i] = transposed->vertices[i-1] + inEdges[i-1];
    }
    transposed->vertices[graph->V] = graph->E;

    // Copy Graph Data to their respective memory buffers
    err |= clEnqueueWriteBuffer(command_queue, offset_buffer, CL_FALSE,0, graph->V * sizeof(cl_uint), transposed->vertices,0, NULL, NULL);
    CLU_ERRCHECK(err,"Failed copying transpose->vertices to offset_buffer");

    //Set Kernel Arguments
    //cluSetKernelArguments(transpose_kernel,6,sizeof(cl_mem),(void*)&vertice_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&weight_buffer,sizeof(cl_mem),(void*)&offset_buffer, sizeof(cl_mem),(void*)&new_edge_buffer,sizeof(cl_mem),(void*)&new_weight_buffer);

    // enqueue kernel to calculate the transposal of the graph
    size_t globalVertexSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, transpose_kernel, 1, NULL, &globalVertexSize, NULL, 0, NULL, NULL),"Error executing transpose_Kernel");

    // Read back results
    err = clEnqueueReadBuffer(command_queue,new_edge_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->E,transposed->edges,0,NULL,NULL);
    err |= clEnqueueReadBuffer(command_queue,new_weight_buffer,CL_TRUE,0,sizeof(cl_float) * graph->E,transposed->weight,0,NULL,NULL);
    CLU_ERRCHECK(err,"Error reading back results of transpose_kernel");

    // Wait for Command queue to finish
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    CLU_ERRCHECK(err,"Error finishing command_queue");

    // Save time if asked
    if(time != NULL)
        *time =  time_ms() - start_time;

    // Free allocated space
    free(inEdges);

    // Clean up
    err = clReleaseKernel(inEdges_kernel);
    err |= clReleaseKernel(transpose_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(inEdges_buffer);
    err |= clReleaseMemObject(vertice_buffer);
    err |= clReleaseMemObject(edge_buffer);
    err |= clReleaseMemObject(weight_buffer);
    err |= clReleaseMemObject(offset_buffer);
    err |= clReleaseMemObject(new_edge_buffer);
    err |= clReleaseMemObject(new_weight_buffer);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err, "Failed during finalizing OpenCL");

    return transposed;
}
