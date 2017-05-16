#include <stdlib.h>
#include <stdio.h>
#include <edge_vertice_message.h>
#include <stdbool.h>
#include <float.h>
#include <cl_utils.h>
#include <benchmark_utils.h>
#include <dijkstra_serial.h>
#include <limits.h>
#include <unistd.h>
#include <libgen.h>

#define GROUP_NUM 32
#define BUCKET_NUM 1000

static cl_program program;
static cl_context context;
static cl_command_queue command_queue;
static cl_device_id device;

static void build_kernel(size_t device_num, int group_num)
{
    device = cluInitDevice(device_num,&context,&command_queue);
    //printf("%s\n\n",cluGetDeviceDescription(device,device_num));
    char tmp[1024];
    sprintf(tmp, "-DGROUP_NUM=%d",group_num);
    char* filename = "/EVM.cl";
    char cfp[1024];
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s%s",dirname(dirname(cfp)),"/kernels",filename);
    program = cluBuildProgramFromFile(context,device,kernel_file,tmp);
}

void preprocessing_parallel_cpu(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVerticesSorted,cl_uint* numEdgesSorted, cl_uint* oldToNew, cl_uint* newToOld, cl_uint* offset,cl_uint* messageBufferSize,size_t device_num)
{
    // Build the kernel
    build_kernel(device_num,1);

    //Allocate necessary data
    cl_uint* numEdges = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* sourceVertices = (cl_uint*) malloc(graph->E * sizeof(cl_uint));

    // calculate sourceVertices & numEdges
    calculateNumEdgesAndSourceVertices(graph,sourceVertices,numEdges);

    // sort MessageBuffer
    messageBufferSort_parallel(graph,numEdges,numEdgesSorted,oldToNew,newToOld,offset);

    //calculate the write positions for each edge in the messagebuffer
    CalculateWriteIndices(graph,oldToNew,messageWriteIndex,offset,numEdgesSorted, messageBufferSize);

    // save new aliases
    sortSourceVertices(graph,sourceVertices,oldToNew,sourceVerticesSorted);

    //Free Resources
    free(numEdges);
    free(sourceVertices);

    //Clean Up
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

}


void preprocessing_parallel_gpu(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVerticesSorted,cl_uint* numEdgesSorted, cl_uint* oldToNew,cl_uint* newToOld, cl_uint* offset,cl_uint* messageBufferSize,size_t device_num)
{
    // Build the kernel
    unsigned long start_time = time_ms();
    build_kernel(device_num,GROUP_NUM);
    printf("Function: Building Kernel...\t%lu\n",time_ms()-start_time);

    //Allocate necessary data
    cl_uint* numEdges = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* sourceVertices = (cl_uint*) malloc(graph->E * sizeof(cl_uint));

    // calculate sourceVertices & numEdges
    start_time = time_ms();
    calculateNumEdgesAndSourceVertices(graph,sourceVertices,numEdges);
    printf("Function: calculateNumEdges...\t%lu\n",time_ms()-start_time);

    // sort MessageBuffer
    start_time = time_ms();
    messageBufferSort_parallel(graph,numEdges,numEdgesSorted,oldToNew,newToOld,offset);
    printf("Function: sortmessagebuffer...\t%lu\n",time_ms()-start_time);

    //remap MessageBuffer
    start_time = time_ms();
    remapMassageBuffer_parallel(graph,messageWriteIndex,numEdgesSorted,offset,oldToNew,messageBufferSize);
    printf("Function: remapmassagebuffer_parallel...\t%lu\n",time_ms()-start_time);

    // save new aliases
    start_time = time_ms();
    sortSourceVertices(graph,sourceVertices,oldToNew,sourceVerticesSorted);
    printf("Function: sortsourcevertices...\t%lu\n",time_ms()-start_time);

    //Free Resources
    free(numEdges);
    free(sourceVertices);

    //Clean Up
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}

void calculateNumEdgesAndSourceVertices(Graph* graph, cl_uint* sourceVertices, cl_uint* numEdges)
{
    cl_int err;

    // Create the Buffers
    cl_mem vertex_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (graph->V+1), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating vertex_buffer");

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edge_buffer");

    cl_mem sourceVertex_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertex_buffer");

    cl_mem numEdges_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint)*graph->V,NULL,&err);
    CLU_ERRCHECK(err,"Failed creating numEdges_buffer");

    // Write data into the buffers
    cl_uint* zeroes = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    err = clEnqueueWriteBuffer(command_queue, vertex_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), graph->vertices , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, vertex_buffer, CL_FALSE, graph->V * sizeof(cl_uint), sizeof(cl_uint), &graph->E , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, numEdges_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint), zeroes , 0, NULL, NULL);
    free(zeroes);

    // Build the kernel for calculating the number of incoming Edges for each Vertex and the SourceVertex for each Edge
    cl_kernel edges_source_kernel = clCreateKernel(program,"inEdgesAndSourceVerticeCalculation",&err);
    CLU_ERRCHECK(err,"Failed to create initializing kernel from program");

    // Set the Kernel Arguments
    cluSetKernelArguments(edges_source_kernel,4,sizeof(cl_mem),(void*)&vertex_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&sourceVertex_buffer,sizeof(cl_mem),(void*)&numEdges_buffer);

    // Execute the OpenCL kernels
    size_t globalSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, edges_source_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    // Read out calculated data
    err = clEnqueueReadBuffer(command_queue,sourceVertex_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->E,sourceVertices,0,NULL,NULL);
    err = clEnqueueReadBuffer(command_queue,numEdges_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->V,numEdges,0,NULL,NULL);

    //Finalize
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    err |= clReleaseKernel(edges_source_kernel);
    err |= clReleaseMemObject(vertex_buffer);
    err |= clReleaseMemObject(edge_buffer);
    err |= clReleaseMemObject(sourceVertex_buffer);
    err |= clReleaseMemObject(numEdges_buffer);

    CLU_ERRCHECK(err, "Failed during finalizing OpenCL in function calculateNumEdgesAndSourceVertices()");
}

void messageBufferSort_parallel(Graph* graph, cl_uint* inEdges, cl_uint* inEdgesSorted, cl_uint* oldToNew,cl_uint* newToOld, cl_uint* offset)
{

    cl_int err;
    unsigned length = graph->V;

    // Calculate the Maximum and the Minimum Value of the incoming Edges of all Vertices
    cl_uint max = 0;
    cl_uint min = INT_MAX;

    for(int i = 0; i<length;i++)
    {
        if(inEdges[i] > max)
            max = inEdges[i];

        if(inEdges[i] < min)
            min = inEdges[i];
    }

    // Create buffers
    cl_mem input_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating inputbuffer");

    cl_mem input_sorted_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating input_sorted buffer");

    cl_mem old_to_new_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating old_to_new_buffer");

    cl_mem new_to_old_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating new_to_old_buffer");

    cl_mem offset_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating offset_buffer");

    cl_mem bucket_count_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating bucket_count_buffer");

    cl_mem bucket_index_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating bucket_index_buffer");

    // Copy Data to their respective memory buffers
    cl_uint* zeroes = (cl_uint*) calloc(length,sizeof(cl_uint));
    err = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0, length * sizeof(cl_uint), inEdges , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, bucket_count_buffer, CL_TRUE, 0, length * sizeof(cl_uint), zeroes , 0, NULL, NULL);
    free(zeroes);
    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    // Create the kernels
    cl_kernel assign_kernel = clCreateKernel(program,"assign_bucket",&err);
    CLU_ERRCHECK(err,"Failed to create initializing kernel from program");

    cl_kernel sort_kernel = clCreateKernel(program,"appr_sort",&err);
    CLU_ERRCHECK(err,"Failed to create EdgeCompute kernel from program");

     //Set KernelArguments
    cl_uint bucket_num = BUCKET_NUM;
    cluSetKernelArguments(assign_kernel,7,sizeof(cl_mem),(void*)&input_buffer,sizeof(cl_uint),(void*)&max,sizeof(cl_uint),(void*)&min,sizeof(cl_mem),(void*)&offset_buffer,sizeof(cl_uint),(void*)&bucket_num,sizeof(cl_mem),(void*)&bucket_count_buffer,sizeof(cl_mem),(void*)&bucket_index_buffer);
    cluSetKernelArguments(sort_kernel,7,sizeof(cl_mem),(void*)&input_buffer,sizeof(cl_mem),(void*)&input_sorted_buffer,sizeof(cl_mem),(void*)&offset_buffer,sizeof(cl_mem),(void*)&bucket_count_buffer,sizeof(cl_mem),(void*)&bucket_index_buffer,sizeof(cl_mem),(void*)&old_to_new_buffer,sizeof(cl_mem),(void*)&new_to_old_buffer);

    // Execute the OpenCL kernels
    size_t globalSize = length;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, assign_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    //Perform Prefix Scan on Count
    cl_uint* bucket_count = (cl_uint*) malloc(sizeof(cl_uint) * length);
    cl_uint* scanned_count = (cl_uint*) malloc(sizeof(cl_uint) * length);

    // Read back the results of the bucket_count_buffer, which indicates the total number of elements in the bucket
    err = clEnqueueReadBuffer(command_queue,bucket_count_buffer,CL_TRUE,0,sizeof(cl_uint) * length,bucket_count,0,NULL,NULL);

    //Scan over bucket count to know the offsets for each vertice in the new messageBuffer
    scanned_count[0] = 0;
    for(int i = 1; i<length;i++)
    {
        scanned_count[i] = scanned_count[i-1] + bucket_count[i-1];
    }
    // Write the offset in the respective buffer
    err = clEnqueueWriteBuffer(command_queue,bucket_count_buffer,CL_TRUE,0,sizeof(cl_uint) * length, scanned_count,0,NULL,NULL);

    // Enqueue Kernel which sorts the vertices by the number of incoming edges, i.e. put every vertex in the right bucket on it's position
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, sort_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing apprsortKernel");
    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    // Read out the sorted Arrays
    err = clEnqueueReadBuffer(command_queue,input_sorted_buffer,CL_FALSE,0,sizeof(cl_uint) * length,inEdgesSorted,0,NULL,NULL);
    err |= clEnqueueReadBuffer(command_queue,old_to_new_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->V,oldToNew,0,NULL,NULL);
    err |= clEnqueueReadBuffer(command_queue,new_to_old_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->V,newToOld,0,NULL,NULL);
    CLU_ERRCHECK(err,"Failed reading back graph data from buffers");

    //Free allocated data
    free(bucket_count);
    free(scanned_count);

    //Finalize
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    err |= clReleaseKernel(assign_kernel);
    err |= clReleaseKernel(sort_kernel);
    err |= clReleaseMemObject(input_buffer);
    err |= clReleaseMemObject(input_sorted_buffer);
    err |= clReleaseMemObject(old_to_new_buffer);
    err |= clReleaseMemObject(new_to_old_buffer);
    err |= clReleaseMemObject(offset_buffer);
    err |= clReleaseMemObject(bucket_count_buffer);
    err |= clReleaseMemObject(bucket_index_buffer);

    CLU_ERRCHECK(err, "Failed during finalizing OpenCL in function messageBufferSort_parallel()");

}

void remapMassageBuffer_parallel(Graph* graph,cl_uint *messageWriteIndex, cl_uint *numEdgesSorted, cl_uint* offset, cl_uint *oldToNew, cl_uint *messageBufferSize)
{
    //Calculate maxima array (Each entry saves the maximum value of the Group multiplied by GROUP_NUM

    cl_int err;

    // Make globalSize a multiple of GROUP_NUM
    size_t globalSize = round_up_globalSize(graph->V,GROUP_NUM);
    size_t buckets = globalSize / GROUP_NUM;

    // Allocate the data for maxima array
    unsigned* maxima = (unsigned*)malloc(sizeof(unsigned) * buckets);

    // Create the buffers
    cl_mem numEdges_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (globalSize), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating numEdgesbuffer");

    cl_mem maxima_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * buckets, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating maxima_buffer");

    // Copy Data to their respective memorybuffers
    cl_uint* zeroes = (cl_uint*) calloc(GROUP_NUM,sizeof(cl_uint));
    err = clEnqueueWriteBuffer(command_queue, numEdges_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint), numEdgesSorted , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, numEdges_buffer, CL_TRUE, graph->V * sizeof(cl_uint), (globalSize - graph->V)*sizeof(cl_uint) , zeroes, 0, NULL, NULL);
    free(zeroes);

    // Create the Kernel
    cl_kernel maxima_kernel = clCreateKernel(program,"maxima",&err);
    CLU_ERRCHECK(err,"Failed to create maxima kernel from program");

    // Set the Kernel Arguments
    cluSetKernelArguments(maxima_kernel,2,sizeof(cl_mem),(void*)&numEdges_buffer,sizeof(cl_mem),(void*)&maxima_buffer);

    // Execute the OpenCL kernels
    size_t localSize = GROUP_NUM;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, maxima_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL),"Error executing InitializeKernel");

    // Read out the data for the maxima Array
    err = clEnqueueReadBuffer(command_queue,maxima_buffer,CL_TRUE,0,sizeof(cl_uint) * buckets,maxima,0,NULL,NULL);

    /*err = clFlush(command_queue);
    err = clFinish(command_queue);*/

    //Prefix Scan over maxima
    offset[0] = 0;
    for(int i = 0; i<buckets-1; i++)
       offset[i+1] = offset[i] + maxima[i];

    // Save the size of the messageBuffer
    *messageBufferSize = offset[buckets-1] + maxima[buckets-1];

    // Free allocated Data
    free(maxima);

    // Calculate the WriteIndices, i.e. the location in the messageBuffer an edge writes to or the destination vertex reads from
    // Serial Execution seems to be faster
    unsigned long start_time = time_ms();
    serialCalculationofWriteIndices(graph,oldToNew,offset,messageWriteIndex);
    printf("Function: serialCalculationofWriteIndices...\t%lu\n",time_ms()-start_time);

    //Use this if you have a powerfull GPU
//    start_time = time_ms();
//    parallelCalculationofWriteIndices(graph,oldToNew,offset,messageWriteIndex,buckets,*messageBufferSize);
//    printf("Function: parallelCalculationofWriteIndices...\t%lu\n",time_ms()-start_time);

    //Clean up
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    err |= clReleaseKernel(maxima_kernel);
    err |= clReleaseMemObject(numEdges_buffer);
    err |= clReleaseMemObject(maxima_buffer);

    CLU_ERRCHECK(err, "Failed during finalizing OpenCL in function remapMessageBuffer_parallel()");
}

void parallelCalculationWriteIndices(Graph* graph, cl_uint* oldToNew, cl_uint* offset, cl_uint* messageWriteIndex, size_t buckets, cl_uint messageBufferSize)
{
    // Calculate the Write Indices for the Edges
    cl_int err;

    // Create the Buffers
    cl_mem edges_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating numEdgesbuffer");

    cl_mem oldToNew_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating oldToNew_buffer");

    cl_mem offset_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * buckets, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating offset_buffer");

    cl_mem helper_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating helper_buffer");

    cl_mem writeIndices_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * messageBufferSize, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating writeIndices_buffer");

    // Copy data to their respective memory buffers
    cl_uint* zeroes = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    err = clEnqueueWriteBuffer(command_queue, edges_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, oldToNew_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), oldToNew, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, offset_buffer, CL_FALSE, 0, buckets * sizeof(cl_uint), offset, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, helper_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint) , zeroes, 0, NULL, NULL);
    free(zeroes);

    // Create the kernel
    cl_kernel calcWriteIndices_kernel = clCreateKernel(program,"calculateWriteIndices",&err);
    CLU_ERRCHECK(err,"Failed to create calculateWriteIndices kernel from program");

    // Set the kernel Arguments
    cluSetKernelArguments(calcWriteIndices_kernel,5,sizeof(cl_mem),(void*)&edges_buffer,sizeof(cl_mem),(void*)&oldToNew_buffer,sizeof(cl_mem),(void*)&offset_buffer,sizeof(cl_mem),(void*)&helper_buffer,sizeof(cl_mem),(void*)&writeIndices_buffer);

    size_t globalSize = graph->E;
    cl_event kernel_event;
    cl_ulong time_start, time_end;
    long unsigned total_time = 0;

    // Execute the OpenCL kernels
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, calcWriteIndices_kernel, 1, NULL, &globalSize, NULL, 0, NULL, &kernel_event),"Error executing calculateWriteIndices Kernel");
    err = clEnqueueReadBuffer(command_queue,writeIndices_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->E,messageWriteIndex,0,NULL,NULL);

    // Reads out times the kernel was starting and finishing execution
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    // Calculates the time it took for the kernel and prints it to the console
    total_time += time_end - time_start;
    printf("Function: calcWriteIndicesKernel\t%lu ms\n",total_time/1000000);

    // Finalize
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    err |= clReleaseKernel(calcWriteIndices_kernel);
    err |= clReleaseMemObject(edges_buffer);
    err |= clReleaseMemObject(oldToNew_buffer);
    err |= clReleaseMemObject(offset_buffer);
    err |= clReleaseMemObject(helper_buffer);
    err |= clReleaseMemObject(writeIndices_buffer);

    CLU_ERRCHECK(err, "Failed during finalizing OpenCL in function ParallelExecutionofWriteIndices()");
}

void serialCalculationofWriteIndices(Graph* graph, cl_uint* oldToNew, cl_uint* offset, cl_uint* messageWriteIndex)
{
    // Create a helper object which saves the amount of all previously incoming edges for a given destination vertex
    unsigned* helper = (unsigned*)calloc(graph->V,sizeof(unsigned));

    // Loop over all edges, and calculate the writePostions
    for(int i = 0; i<graph->E;i++)
    {
        cl_uint old_dest = graph->edges[i];
        cl_uint new_dest = oldToNew[old_dest];
        cl_uint group_id = new_dest/GROUP_NUM;
        cl_uint inner_id = new_dest%GROUP_NUM;
        messageWriteIndex[i] = offset[group_id] + inner_id + GROUP_NUM*(helper[old_dest]++);//atomic_inc(&helper[old_source])*GROUP_NUM);
    }

    //Free allocated data
    free(helper);
}



void sortSourceVertices(Graph* graph, cl_uint* sourceVertices, cl_uint* oldToNew,cl_uint* sourceVerticesSorted)
{
    cl_int err;

    // Create buffers
    cl_mem sourceVertex_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    cl_mem oldToNew_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    cl_mem sorted_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertexSorted_buffer");

    // Write data to the respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, sourceVertex_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), sourceVertices, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, oldToNew_buffer, CL_TRUE, 0,graph->V * sizeof(cl_uint),oldToNew, 0, NULL, NULL);

    // create the kernel
    cl_kernel sort_kernel = clCreateKernel(program,"sort_source_vertex",&err);
    CLU_ERRCHECK(err,"Failed to create sort_source_vertex kernel from program");

    // set the kernel arguments
    cluSetKernelArguments(sort_kernel,3,sizeof(cl_mem),(void*)&sourceVertex_buffer,sizeof(cl_mem),(void*)&oldToNew_buffer,sizeof(cl_mem),(void*)&sorted_buffer);

    // Execute the OpenCL kernel
    size_t globalSize = graph->E;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, sort_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    // Read out the now sorted array of the SourceVertices of each edge
    err = clEnqueueReadBuffer(command_queue,sorted_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->E,sourceVerticesSorted,0,NULL,NULL);

    //Finalize
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    err |= clReleaseKernel(sort_kernel);
    err |= clReleaseMemObject(sourceVertex_buffer);
    err |= clReleaseMemObject(oldToNew_buffer);
    err |= clReleaseMemObject(sorted_buffer);

    CLU_ERRCHECK(err, "Failed during finalizing OpenCL in function sortSourceVertices()");

}


void CalculateWriteIndices(Graph* graph, cl_uint *oldToNew, cl_uint *messageWriteIndex, cl_uint *offset, cl_uint* inEdgesSorted, cl_uint *messageBufferSize)
{
    // If ther's no remapping, the size of the messageBuffer is equal to the number of edges in the graph
    *messageBufferSize = graph->E;

    // Calculate the offsets
    for(int i = 0; i<graph->V;i++)
        offset[i+1] = offset[i] + inEdgesSorted[i];

    // Calculate the Write Indices for the Edges
    unsigned* helper = (unsigned*)calloc(graph->V,sizeof(unsigned));
    for(int i = 0; i<graph->E;i++)
    {
        cl_uint old_source = graph->edges[i];
        cl_uint new_source = oldToNew[old_source];
        messageWriteIndex[i] = offset[new_source] + helper[old_source];
        helper[old_source]++;
    }
    free(helper);
}
