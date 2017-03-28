#include <stdlib.h>
#include <stdio.h>
#include <edge_vertice_message.h>
#include <stdbool.h>
#include <float.h>
#include <cl_utils.h>
#include <time_ms.h>
#include <dijkstra_serial.h>
#include <limits.h>
#include <unistd.h>
#include <libgen.h>

#define CL_DEVICE 1
#define GROUP_NUM 32
#define BUCKET_NUM 100
#define PREPROCESS_ENABLE 0

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
    sprintf(kernel_file,"%s%s",dirname(cfp),filename);
    filename = kernel_file;

    program = cluBuildProgramFromFile(context,device,kernel_file,tmp);
}

void preprocessing_parallel_cpu(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVerticesSorted,cl_uint* numEdgesSorted, cl_uint* oldToNew, cl_uint* offset,cl_uint* messageBufferSize,size_t device_num)
{
    build_kernel(device_num,1);

    //Allocate necessary data
    cl_uint* numEdges = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* sourceVertices = (cl_uint*) malloc(graph->E * sizeof(cl_uint));

    // calculate sourceVertices & numEdges
    calculateNumEdgesAndSourceVertices(graph,sourceVertices,numEdges);

    // sort MessageBuffer
    messageBufferSort_parallel(graph,numEdges,numEdgesSorted,oldToNew,offset);

    //remap MessageBuffer
    CalculateWriteIndices(graph,oldToNew,messageWriteIndex,offset,numEdgesSorted, messageBufferSize);
    //remapMassageBuffer_parallel(graph,messageWriteIndex,numEdgesSorted,offset,oldToNew,messageBufferSize);

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


void preprocessing_parallel(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVerticesSorted,cl_uint* numEdgesSorted, cl_uint* oldToNew, cl_uint* offset,cl_uint* messageBufferSize,size_t device_num)
{
    build_kernel(device_num,GROUP_NUM);
    //Allocate necessary data
    cl_uint* numEdges = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* sourceVertices = (cl_uint*) malloc(graph->E * sizeof(cl_uint));

    // calculate sourceVertices & numEdges
    calculateNumEdgesAndSourceVertices(graph,sourceVertices,numEdges);

    // sort MessageBuffer
    messageBufferSort_parallel(graph,numEdges,numEdgesSorted,oldToNew,offset);

    //remap MessageBuffer
    remapMassageBuffer_parallel(graph,messageWriteIndex,numEdgesSorted,offset,oldToNew,messageBufferSize);

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

void calculateNumEdgesAndSourceVertices(Graph* graph, cl_uint* sourceVertices, cl_uint* numEdges)
{
    cl_int err;

    cl_mem vertex_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (graph->V+1), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    cl_mem sourceVertex_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    cl_mem numEdges_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint)*graph->V,NULL,&err);
    CLU_ERRCHECK(err,"Failed creating numEdges_buffer");

    cl_uint* zeroes = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    err = clEnqueueWriteBuffer(command_queue, vertex_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), graph->vertices , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, vertex_buffer, CL_FALSE, graph->V * sizeof(cl_uint), sizeof(cl_uint), &graph->E , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, numEdges_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint), zeroes , 0, NULL, NULL);
    free(zeroes);

    cl_kernel preprocess_kernel = clCreateKernel(program,"preprocess",&err);
    CLU_ERRCHECK(err,"Failed to create initializing kernel from program");

    cluSetKernelArguments(preprocess_kernel,4,sizeof(cl_mem),(void*)&vertex_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&sourceVertex_buffer,sizeof(cl_mem),(void*)&numEdges_buffer);

    // Execute the OpenCL kernels
    size_t globalSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, preprocess_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    err = clEnqueueReadBuffer(command_queue,sourceVertex_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->E,sourceVertices,0,NULL,NULL);
    err = clEnqueueReadBuffer(command_queue,numEdges_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->V,numEdges,0,NULL,NULL);

    //Finalize
    err = clFlush(command_queue);
    err = clFinish(command_queue);
    err = clReleaseKernel(preprocess_kernel);
    err = clReleaseMemObject(vertex_buffer);
    err = clReleaseMemObject(edge_buffer);
    err = clReleaseMemObject(sourceVertex_buffer);
    err = clReleaseMemObject(numEdges_buffer);
}

void messageBufferSort_parallel(Graph* graph, cl_uint* inEdges, cl_uint* inEdgesSorted, cl_uint* oldToNew, cl_uint* offset)
{

    unsigned length = graph->V;
    cl_uint max = 0;
    cl_uint min = INT_MAX;
    cl_uint bucket_num = BUCKET_NUM;

    /* CAN BE PARALLELIZED AS REDUCTION */
    for(int i = 0; i<length;i++)
    {
        if(inEdges[i] > max)
            max = inEdges[i];

        if(inEdges[i] < min)
            min = inEdges[i];
    }

    cl_int err;

    cl_mem input_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating inputbuffer");

    cl_mem input_sorted_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating inputbuffer");

    cl_mem old_to_new_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating inputbuffer");

    cl_mem offset_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating offsetbuffer");

    cl_mem bucket_count_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating offsetbuffer");

    cl_mem bucket_index_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating offsetbuffer");

    // Copy Data to their respective memory buffers
    cl_uint* zeroes = (cl_uint*) calloc(length,sizeof(cl_uint));
    err = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0, length * sizeof(cl_uint), inEdges , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, bucket_count_buffer, CL_TRUE, 0, length * sizeof(cl_uint), zeroes , 0, NULL, NULL);
    free(zeroes);

    cl_kernel assign_kernel = clCreateKernel(program,"assign_bucket",&err);
    CLU_ERRCHECK(err,"Failed to create initializing kernel from program");

    cl_kernel sort_kernel = clCreateKernel(program,"appr_sort",&err);
    CLU_ERRCHECK(err,"Failed to create EdgeCompute kernel from program");

     //Set KernelArguments
    cluSetKernelArguments(assign_kernel,7,sizeof(cl_mem),(void*)&input_buffer,sizeof(cl_uint),(void*)&max,sizeof(cl_uint),(void*)&min,sizeof(cl_mem),(void*)&offset_buffer,sizeof(cl_uint),(void*)&bucket_num,sizeof(cl_mem),(void*)&bucket_count_buffer,sizeof(cl_mem),(void*)&bucket_index_buffer);
    cluSetKernelArguments(sort_kernel,6,sizeof(cl_mem),(void*)&input_buffer,sizeof(cl_mem),(void*)&input_sorted_buffer,sizeof(cl_mem),(void*)&offset_buffer,sizeof(cl_mem),(void*)&bucket_count_buffer,sizeof(cl_mem),(void*)&bucket_index_buffer,sizeof(cl_mem),(void*)&old_to_new_buffer);

    // Execute the OpenCL kernels
    size_t globalSize = length;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, assign_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    //Perform Prefix Scan on Count
    cl_uint* bucket_count = (cl_uint*) malloc(sizeof(cl_uint) * length);
    cl_uint* scanned_count = (cl_uint*) malloc(sizeof(cl_uint) * length);

    err = clEnqueueReadBuffer(command_queue,bucket_count_buffer,CL_TRUE,0,sizeof(cl_uint) * length,bucket_count,0,NULL,NULL);

    /*CAN BE PARALLELIZED AS PREFIX SCAN*/
    scanned_count[0] = 0;
    for(int i = 1; i<length;i++)
    {
        scanned_count[i] = scanned_count[i-1] + bucket_count[i-1];
    }
    err = clEnqueueWriteBuffer(command_queue,bucket_count_buffer,CL_TRUE,0,sizeof(cl_uint) * length, scanned_count,0,NULL,NULL);

    free(bucket_count);
    free(scanned_count);


    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, sort_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing apprsortKernel");

    err = clEnqueueReadBuffer(command_queue,input_sorted_buffer,CL_TRUE,0,sizeof(cl_uint) * length,inEdgesSorted,0,NULL,NULL);
    err = clEnqueueReadBuffer(command_queue,old_to_new_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->V,oldToNew,0,NULL,NULL);


    //Finalize
    err = clFlush(command_queue);
    err = clFinish(command_queue);
    err = clReleaseKernel(assign_kernel);
    err = clReleaseKernel(sort_kernel);
    err = clReleaseMemObject(input_buffer);
    err = clReleaseMemObject(input_sorted_buffer);
    err = clReleaseMemObject(old_to_new_buffer);
    err = clReleaseMemObject(offset_buffer);
    err = clReleaseMemObject(bucket_count_buffer);
    err = clReleaseMemObject(bucket_index_buffer);

}

void remapMassageBuffer_parallel(Graph* graph,cl_uint *messageWriteIndex, cl_uint *numEdgesSorted, cl_uint* offset, cl_uint *oldToNew, cl_uint *messageBufferSize)
{
    /*Calculate maxima array (Each entry saves the maximum value of the Group multiplied by GROUP_NUM*/
    size_t globalSize = round_up_globalSize(graph->V,GROUP_NUM);
    size_t buckets = globalSize / GROUP_NUM;

    unsigned* maxima = (unsigned*)malloc(sizeof(unsigned) * buckets);

    cl_int err;

    cl_mem numEdges_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (globalSize), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating numEdgesbuffer");

    cl_mem maxima_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * buckets, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating maxima_buffer");

    cl_uint* zeroes = (cl_uint*) calloc(GROUP_NUM,sizeof(cl_uint));
    err = clEnqueueWriteBuffer(command_queue, numEdges_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint), numEdgesSorted , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, numEdges_buffer, CL_TRUE, graph->V * sizeof(cl_uint), (globalSize - graph->V)*sizeof(cl_uint) , zeroes, 0, NULL, NULL);
    free(zeroes);

    cl_kernel maxima_kernel = clCreateKernel(program,"maxima",&err);
    CLU_ERRCHECK(err,"Failed to create maxima kernel from program");

    cluSetKernelArguments(maxima_kernel,2,sizeof(cl_mem),(void*)&numEdges_buffer,sizeof(cl_mem),(void*)&maxima_buffer);

    // Execute the OpenCL kernels
    size_t localSize = GROUP_NUM;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, maxima_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL),"Error executing InitializeKernel");

    err = clEnqueueReadBuffer(command_queue,maxima_buffer,CL_TRUE,0,sizeof(cl_uint) * buckets,maxima,0,NULL,NULL);


    err = clFlush(command_queue);
    err = clFinish(command_queue);

    //Prefix Scan over maxima
    /*CAN BE CALCULATED IN PARALLEL WITH PREFIX SCAN*/
    offset[0] = 0;
    for(int i = 0; i<buckets-1; i++)
    {
       offset[i+1] = offset[i] + maxima[i];
    }

    *messageBufferSize = offset[buckets-1] + maxima[buckets-1];
    free(maxima);

    // Calculate the Write Indices for the Edges
    cl_mem edges_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating numEdgesbuffer");

    cl_mem oldToNew_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating maxima_buffer");

    cl_mem offset_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * buckets, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating maxima_buffer");

    cl_mem helper_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating maxima_buffer");

    cl_mem writeIndices_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * messageBufferSize[0], NULL, &err);
    CLU_ERRCHECK(err,"Failed creating maxima_buffer");

    zeroes = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    err = clEnqueueWriteBuffer(command_queue, edges_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, oldToNew_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), oldToNew, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, offset_buffer, CL_FALSE, 0, buckets * sizeof(cl_uint), offset, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, helper_buffer, CL_TRUE, 0, graph->V * sizeof(cl_uint) , zeroes, 0, NULL, NULL);
    free(zeroes);

    cl_kernel calcWriteIndices_kernel = clCreateKernel(program,"calculateWriteIndices",&err);
    CLU_ERRCHECK(err,"Failed to create calculateWriteIndices kernel from program");

    cluSetKernelArguments(calcWriteIndices_kernel,5,sizeof(cl_mem),(void*)&edges_buffer,sizeof(cl_mem),(void*)&oldToNew_buffer,sizeof(cl_mem),(void*)&offset_buffer,sizeof(cl_mem),(void*)&helper_buffer,sizeof(cl_mem),(void*)&writeIndices_buffer);
    // Execute the OpenCL kernels
    globalSize = graph->E;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, calcWriteIndices_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    err = clEnqueueReadBuffer(command_queue,writeIndices_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->E,messageWriteIndex,0,NULL,NULL);

    err = clFlush(command_queue);
    err = clFinish(command_queue);
    err = clReleaseKernel(maxima_kernel);
    err = clReleaseKernel(calcWriteIndices_kernel);
    err = clReleaseMemObject(numEdges_buffer);
    err = clReleaseMemObject(maxima_buffer);
    err = clReleaseMemObject(edges_buffer);
    err = clReleaseMemObject(oldToNew_buffer);
    err = clReleaseMemObject(offset_buffer);
    err = clReleaseMemObject(helper_buffer);
    err = clReleaseMemObject(writeIndices_buffer);
}


void sortSourceVertices(Graph* graph, cl_uint* sourceVertices, cl_uint* oldToNew,cl_uint* sourceVerticesSorted)
{
    cl_int err;

    cl_mem sourceVertex_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    cl_mem oldToNew_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->V, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    cl_mem sorted_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertexSorted_buffer");

    err = clEnqueueWriteBuffer(command_queue, sourceVertex_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), sourceVertices, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, oldToNew_buffer, CL_TRUE, 0,graph->V * sizeof(cl_uint),oldToNew, 0, NULL, NULL);

    cl_kernel sort_kernel = clCreateKernel(program,"sort_source_vertex",&err);
    CLU_ERRCHECK(err,"Failed to create sort_source_vertex kernel from program");

    cluSetKernelArguments(sort_kernel,3,sizeof(cl_mem),(void*)&sourceVertex_buffer,sizeof(cl_mem),(void*)&oldToNew_buffer,sizeof(cl_mem),(void*)&sorted_buffer);

    // Execute the OpenCL kernels
    size_t globalSize = graph->E;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, sort_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    err = clEnqueueReadBuffer(command_queue,sorted_buffer,CL_TRUE,0,sizeof(cl_uint) * graph->E,sourceVerticesSorted,0,NULL,NULL);

    //Finalize
    err = clFlush(command_queue);
    err = clFinish(command_queue);
    err = clReleaseKernel(sort_kernel);
    err = clReleaseMemObject(sourceVertex_buffer);
    err = clReleaseMemObject(oldToNew_buffer);
    err = clReleaseMemObject(sorted_buffer);

}

void preprocessing(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVerticesSorted, cl_uint* numEdgesSorted, cl_uint* oldToNew, cl_uint* offset,cl_uint* messageBufferSize)
{
        /* Fill sourceVertice array and numEdges which is the number of every incoming edges for every vertice*/
        int  edge_count = 0;
        int num_neighbors = 0;
        /* Can be parallelized */
        for(int i = 0; i<graph->V-1;i++)
        {
            num_neighbors = graph->vertices[i+1] - graph->vertices[i];
            for(int j = 0; j<num_neighbors;j++)
            {
                sourceVerticesSorted[edge_count+j] = i;
                numEdgesSorted[graph->edges[edge_count+j]]++;
            }
            edge_count += num_neighbors;
        }
        num_neighbors = graph->E - graph->vertices[graph->V-1];
        for(int j = 0; j<num_neighbors;j++)
        {
            sourceVerticesSorted[edge_count+j] = graph->V-1;
            numEdgesSorted[graph->edges[edge_count+j]]++;
        }

        // sort MessageBuffer
        messageBufferSort(graph,numEdgesSorted,oldToNew,offset);

        //remap MessageBuffer
        remapMassageBuffer(graph,messageWriteIndex,numEdgesSorted,offset,oldToNew,messageBufferSize);


}

void messageBufferSort(Graph* graph,cl_uint* inEdgesSorted, cl_uint* oldToNew, cl_uint* offset)
{
    //Calculate offsets
    offset[0] = 0;
    int i = 0;
    for(; i<graph->V-1;i++)
    {
        oldToNew[i] = i;
        offset[i+1] = offset[i] + inEdgesSorted[i];
    }
    oldToNew[i] = i;
}

void remapMassageBuffer(Graph* graph,cl_uint *messageWriteIndex, cl_uint *numEdgesSorted, cl_uint* offset, cl_uint *oldToNew, cl_uint *messageBufferSize)
{
     // Remap MassageBuffer
    // use this to calculate the amount of buckets for the remapping
    unsigned group_num = 1;

    size_t buckets = graph->V / group_num;
    if(graph->V % group_num != 0)
        buckets++;

    offset[0] = 0;
    *messageBufferSize = 0;

    //Calculate offset for read operation in vertexStage and the total amount of memory for the messageBuffer
    /* Can be parallelized , maximum is a reduction operation, messageBufferSize is a reduction and offset a scan*/
    for(int i = 0; i<buckets; i++)
    {
        unsigned max = 0;
        for(int j = 0; j<group_num;j++)
        {
            unsigned index = i*group_num+ j;
            if(index<graph->V && numEdgesSorted[index] > max)
                max = numEdgesSorted[index];
        }
        *messageBufferSize += max*group_num;
        if(i<buckets-1)
             offset[i+1] = offset[i] + max*group_num;
    }

    // Calculate the Write Indices for the Edges
    unsigned* helper = (unsigned*)calloc(graph->V,sizeof(unsigned));
    for(int i = 0; i<graph->E;i++)
    {
        cl_uint old_source = graph->edges[i];
        cl_uint new_source = oldToNew[old_source];
        unsigned group_id = new_source/group_num;
        unsigned inner_id = new_source%group_num;
        messageWriteIndex[i] = offset[group_id] + inner_id + (helper[old_source]*group_num);
        helper[old_source]++;
    }
    free(helper);
}

void CalculateWriteIndices(Graph* graph, cl_uint *oldToNew, cl_uint *messageWriteIndex, cl_uint *offset, cl_uint* inEdgesSorted, cl_uint *messageBufferSize)
{
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




