#include <graph_transpose.h>
#include <graph.h>
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include <cl_utils.h>
#include <time_ms.h>

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

unsigned long transpose_serial(Graph* graph, Graph* transposed)
{

    unsigned long start_time = time_ms();
    //Calculate number of InEdges
    unsigned *inEdges = (unsigned*) calloc(graph->V,sizeof(unsigned));
    unsigned *outEdges = (unsigned*) calloc(graph->V,sizeof(unsigned));
    unsigned *helper = (unsigned*) calloc(graph->V,sizeof(unsigned));

    for(int i = 0; i<graph->E;i++)
    {
        inEdges[graph->edges[i]]++;
    }

    unsigned long total = time_ms() - start_time;

    transposed->vertices[0] = 0;
    for(int i = 1; i<graph->V;i++)
    {
        outEdges[i-1] = graph->vertices[i] - graph->vertices[i-1];
        transposed->vertices[i] = transposed->vertices[i-1] + inEdges[i-1];
    }

    outEdges[graph->V-1] = graph->E - graph->vertices[graph->V-1];

    for(int i = 0; i<graph->V;i++)
    {
        for(int j = 0; j<outEdges[i]; j++)
        {
            unsigned index = graph->vertices[i] + j;
            unsigned source = graph->edges[index];
            float weight = graph->weight[index];
            unsigned write_index = transposed->vertices[source] + helper[source];
            helper[source]++;
            transposed->edges[write_index] = i;
            transposed->weight[write_index] = weight;
        }
    }

    free(inEdges);
    free(outEdges);
    free(helper);

    return total;
}

unsigned long transpose_parallel(Graph* graph, Graph* transposed, size_t device)
{

    build_kernel(device);

    unsigned long start_time = time_ms();

    cl_int err;
    cl_uint* inEdges = (cl_uint*) calloc(transposed->V, sizeof(cl_uint));

    //Create Buffers for inEdges Kernel
    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    cl_mem inEdges_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, inEdges_buffer, CL_TRUE,0, graph->V * sizeof(cl_uint), inEdges,0, NULL, NULL);
    CLU_ERRCHECK(err,"Failed copying data to buffers");

    cl_kernel inEdges_kernel = clCreateKernel(program,"calcInEdges",&err);
    CLU_ERRCHECK(err,"Failed to create inEdges kernel from program");

    cluSetKernelArguments(inEdges_kernel,2,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&inEdges_buffer);

    size_t globalEdgeSize = graph->E;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, inEdges_kernel, 1, NULL, &globalEdgeSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    err = clEnqueueReadBuffer(command_queue,inEdges_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->V,inEdges,0,NULL,NULL);

    err = clReleaseMemObject(inEdges_buffer);


    unsigned long total =  time_ms() - start_time;

    /*for(int i = 0; i<graph->V;i++)
    {
        printf("%u\t",inEdges[i]);
    }
    printf("\n");
    free(inEdges);*/

    //Calculate new VerticeArray can be parallelized with prefix sum
    transposed->vertices[0] = 0;
    for(int i = 1; i<transposed->V;i++)
    {
        transposed->vertices[i] = transposed->vertices[i-1] + inEdges[i-1];
    }

    free(inEdges);

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
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), graph->vertices , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, graph->V * sizeof(cl_uint),sizeof(cl_uint),&graph->E , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, weight_buffer, CL_FALSE,0, graph->E * sizeof(cl_float), graph->weight,0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, offset_buffer, CL_TRUE,0, graph->V * sizeof(cl_uint), transposed->vertices,0, NULL, NULL);
    CLU_ERRCHECK(err,"Failed copying data to buffers");

    //Create Kernel
    cl_kernel transpose_kernel = clCreateKernel(program,"transpose",&err);
    CLU_ERRCHECK(err,"Failed to create inEdges kernel from program");

    //Set Kernel Arguments
    cluSetKernelArguments(transpose_kernel,6,sizeof(cl_mem),(void*)&vertice_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&weight_buffer,sizeof(cl_mem),(void*)&offset_buffer, sizeof(cl_mem),(void*)&new_edge_buffer,sizeof(cl_mem),(void*)&new_weight_buffer);

    size_t globalVertexSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, transpose_kernel, 1, NULL, &globalVertexSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    err = clEnqueueReadBuffer(command_queue,new_edge_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->E,transposed->edges,0,NULL,NULL);
    err = clEnqueueReadBuffer(command_queue,new_weight_buffer,CL_TRUE,0,sizeof(cl_float) * graph->E,transposed->weight,0,NULL,NULL);

    err = clFlush(command_queue);
    err = clFinish(command_queue);
    err = clReleaseKernel(inEdges_kernel);
    err = clReleaseKernel(transpose_kernel);
    err = clReleaseProgram(program);
    err = clReleaseMemObject(vertice_buffer);
    err = clReleaseMemObject(edge_buffer);
    err = clReleaseMemObject(weight_buffer);
    err = clReleaseMemObject(offset_buffer);
    err = clReleaseMemObject(new_edge_buffer);
    err = clReleaseMemObject(new_weight_buffer);
    err = clReleaseCommandQueue(command_queue);
    err = clReleaseContext(context);
    return total;
}
