#include <stdlib.h>
#include <stdio.h>
#include <edge_vertice_message.h>
#include <stdbool.h>
#include <float.h>
#include <cl_utils.h>
#include <time_ms.h>
#include <dijkstra_serial.h>
#include <limits.h>

#define CL_DEVICE 1
#define GROUP_NUM 64
#define BUCKET_NUM 100


void preprocessing(Graph* graph,cl_uint* messageWriteIndex,cl_uint* sourceVertices,cl_uint* sourceVerticesSorted, cl_uint* numEdges,cl_uint* numEdgesSorted, cl_uint* oldToNew, cl_uint* offset,cl_uint* messageBufferSize,bool enable)
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
                sourceVertices[edge_count+j] = i;
                numEdges[graph->edges[edge_count+j]]++;
            }
            edge_count += num_neighbors;
        }
        num_neighbors = graph->E - graph->vertices[graph->V-1];
        for(int j = 0; j<num_neighbors;j++)
        {
            sourceVertices[edge_count+j] = graph->V-1;
            numEdges[graph->edges[edge_count+j]]++;
        }

        // sort MessageBuffer
        messageBufferSort(graph,numEdges,numEdgesSorted,oldToNew,offset,messageBufferSize,enable);

        //remap MessageBuffer
        remapMassageBuffer(graph,messageWriteIndex,numEdgesSorted,offset,oldToNew,messageBufferSize,enable);

        for(int i = 0; i<graph->E;i++)
        {
            sourceVerticesSorted[i] = oldToNew[sourceVertices[i]];
        }

}

void remapMassageBuffer(Graph* graph,cl_uint *messageWriteIndex, cl_uint *numEdgesSorted, cl_uint* offset, cl_uint *oldToNew, cl_uint *messageBufferSize, bool enable)
{
     // Remap MassageBuffer
    // use this to calculate the amount of buckets for the remapping
    unsigned group_num;
    if(enable)
        group_num = GROUP_NUM;

    else
        group_num = 1;

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
    /*Can be parallelized but without using the helper object*/
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

void messageBufferSort(Graph* graph, cl_uint* inEdges, cl_uint* inEdgesSorted, cl_uint* oldToNew, cl_uint* offset,unsigned* messageBufferSize,bool enable)
{
    if(!enable)
    {
      //Calculate offsets
        offset[0] = 0;
        int i = 0;
        for(; i<graph->V-1;i++)
        {
            inEdgesSorted[i] = inEdges[i];
            oldToNew[i] = i;
            offset[i+1] = offset[i] + inEdges[i];
        }
        inEdgesSorted[i] = inEdges[i];
        oldToNew[i] = i;
        return;
    }

    unsigned length = graph->V;
    cl_uint max = 0;
    cl_uint min = INT_MAX;
    cl_uint bucket_num = BUCKET_NUM;

    for(int i = 0; i<length;i++)
    {
        if(inEdges[i] > max)
            max = inEdges[i];

        if(inEdges[i] < min)
            min = inEdges[i];
    }
    //Setup OpenCL
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device = cluInitDevice(0,&context,&command_queue);
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

    //Build Program and create Kernels
    char tmp[1024];
    sprintf(tmp, "-DGROUP_NUM=%d",1);
    const char* kernel_file = "/home/chris/Dokumente/OpenCL/CodeBlocks Projekte/GraphSearchLibrary/EVM.cl";
    cl_program program = cluBuildProgramFromFile(context,device,kernel_file,tmp);

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

    /*for(int i = 0; i<length;i++)
    {
        printf("%u\t",inEdges[i]);
    }
    printf("\n");

    for(int i = 0; i<length;i++)
    {
        printf("%u\t",inEdgesSorted[i]);
    }
    printf("\n");*/

    //Finalize

    err = clFlush(command_queue);
    err = clFinish(command_queue);
    err = clReleaseKernel(assign_kernel);
    err = clReleaseKernel(sort_kernel);
    err = clReleaseProgram(program);
    err = clReleaseMemObject(input_buffer);
    err = clReleaseMemObject(input_sorted_buffer);
    err = clReleaseMemObject(old_to_new_buffer);
    err = clReleaseMemObject(offset_buffer);
    err = clReleaseMemObject(bucket_count_buffer);
    err = clReleaseMemObject(bucket_index_buffer);
    err = clReleaseCommandQueue(command_queue);
    err = clReleaseContext(context);


}

void sssp(Graph* graph, unsigned source, bool enable)
{
    for(int i = 01; i<2;i++){
    //Allocate Data
    cl_uint* sourceVertices = (cl_uint*)malloc(graph->E * sizeof(cl_uint));
    cl_uint* sourceVerticesSorted = (cl_uint*)malloc(graph->E * sizeof(cl_uint));
    cl_uint* messageWriteIndex = (cl_uint*) malloc(graph->E * sizeof(cl_uint));
    cl_uint* numEdges = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* numEdgesSorted = (cl_uint*) malloc(graph->V * sizeof(cl_uint));
    cl_uint* offset = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint* oldToNew = (cl_uint*) calloc(graph->V,sizeof(cl_uint));
    cl_uint messageBuffersize;

    //Preprocess Indices and initialize VertexBuffer, Calculate the number of incoming Edges for each vertex and set the sourceVertex of each edge
    preprocessing(graph,messageWriteIndex,sourceVertices,sourceVerticesSorted,numEdges,numEdgesSorted,oldToNew,offset,&messageBuffersize,enable);

    cl_float* messageBuffer = (cl_float*) malloc(messageBuffersize * sizeof(cl_float));
    for(int i = 0; i<messageBuffersize;i++){
        messageBuffer[i] = FLT_MAX;
    }

    //Setup OpenCL
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device = cluInitDevice(i,&context,&command_queue);

    cl_int err;
    printf("%s\n",cluGetDeviceDescription(device,i));


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

    //Build Program and create Kernels
    int group_num;
    (enable == true)? (group_num = GROUP_NUM) : (group_num = 1);
    char tmp[1024];
    sprintf(tmp, "-DGROUP_NUM=%d",group_num);
    const char* kernel_file = "/home/chris/Dokumente/OpenCL/CodeBlocks Projekte/GraphSearchLibrary/EVM.cl";
    cl_program program = cluBuildProgramFromFile(context,device,kernel_file,tmp);

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

    unsigned long start_time = time_ms();
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
    unsigned long total_time = time_ms() - start_time;

    printf("total time : %lu\n",total_time);

    float* cost_parallel = (float*) malloc(sizeof(float) * graph->V);
    float* cost_parallel_ordered = (float*) malloc(sizeof(float) * graph->V);
    err = clEnqueueReadBuffer(command_queue,cost_buffer,CL_TRUE,0,sizeof(cl_float) * graph->V,cost_parallel,0,NULL,NULL);

    for(int i = 0; i<graph->V;i++)
    {
        cost_parallel_ordered[i] = cost_parallel[oldToNew[i]];
    }

    /*float* cost_serial = dijkstra_serial(graph,source);

    for(int i = 0;i<graph->V;i++)
    {
        if(cost_serial[i] != cost_parallel_ordered[i])
        {
            printf("Wrong Results an Stelle %d: Serial says %f and parallel says %f\n",i,cost_serial[i],cost_parallel_ordered[i] );
            break;
        }
    }

    /*for(int i = 0; i<graph->V;i++)
    {
        printf("%f\t",cost_parallel_ordered[i]);
    }
    printf("\n");
    free(cost_serial);*/
    free(cost_parallel);
    free(cost_parallel_ordered);


    //Clean up
    free(sourceVertices);
    free(sourceVerticesSorted);
    free(messageBuffer);
    free(messageWriteIndex);
    free(numEdges);
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
    }
}

