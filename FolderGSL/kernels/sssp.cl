
//Theoretically you could use another bool array to check wheter MessageBuffer was changed for current Vertex
void combine(__global float *messageBuffer,__global unsigned *messageBuffer_path, unsigned numMessages, unsigned index, float *min, unsigned *predecessor)
{
    *min = FLT_MAX;
    for(int i = 0; i<numMessages;i++)
    {
        unsigned new_index = index + (i*GROUP_NUM);
        if(*min>messageBuffer[new_index]){
            *min = messageBuffer[new_index];
            *predecessor = messageBuffer_path[new_index];
        }
    }
}
__kernel void initialize(__global float *cost,__global unsigned *path,__global bool *active, unsigned source)
{
    size_t id = get_global_id(0);
    if(id == source)
    {
        cost[id] = 0.0f;
        active[id] = true;
        path[id] = id;
    }
    else
    {
        cost[id] = FLT_MAX;
        active[id] = false;
        path[id] = UINT_MAX;
    }
}

__kernel void edgeCompute(__global unsigned *edges, __global unsigned *sourceVertex, __global unsigned *messageWriteIndex, __global float *messageBuffer, __global unsigned *messageBuffer_path,__global float *cost,__global float *weight, __global bool *active)
{
    size_t id = get_global_id(0);
    unsigned source = sourceVertex[id];
    // can this be achieved coalescing?
    if(active[source])
    {
        messageBuffer[messageWriteIndex[id]] = cost[source] + weight[id];
        messageBuffer_path[messageWriteIndex[id]] = source;
    }
}

__kernel void vertexCompute(__global unsigned *offset, __global float *messageBuffer,__global unsigned *messageBuffer_path, __global unsigned *numEdges,__global float *cost,__global unsigned *path, __global bool *active, global bool* finished)
{
    size_t id = get_global_id(0);
    unsigned numMessages = numEdges[id];
    unsigned index = offset[id/GROUP_NUM];
    index += id%GROUP_NUM;
    float msg_min;
    unsigned predecessor;
    combine(messageBuffer,messageBuffer_path,numMessages, index, &msg_min,&predecessor);
    if(cost[id] > msg_min)
    {
        path[id] = predecessor;
        cost[id] = msg_min;
        active[id] = true;
        *finished = false;
    }
    else
        active[id] = false;
}
