//Theoretically you could use another bool array to check wheter MessageBuffer was changed for current Vertex
void combine(__global float *messageBuffer_cost,__global unsigned *messageBuffer_path,unsigned numMessages, unsigned index, float *min_cost, unsigned *predecessor)
{
    *min_cost = FLT_MAX;
    for(int i = 0; i<numMessages;i++)
    {
        float cost = messageBuffer_cost[index+i];
        if(*min_cost>cost)
        {
            *min_cost = cost;
            *predecessor = messageBuffer_path[index+i];
        }
    }
}
__kernel void initialize(__global float *cost,__global unsigned *path,__global bool *active, unsigned source)
{
    size_t id = get_global_id(0);
    if(id == source)
    {
        cost[id] = 0.0f;
        path[id] = source;
        active[id] = true;
    }
    else
    {
        cost[id] = FLT_MAX;
        path[id] = UINT_MAX;
        active[id] = false;
    }
}

__kernel void edgeCompute(__global unsigned *edges, __global unsigned *sourceVertex, __global unsigned *messageWriteIndex, __global float *messageBuffer_cost, __global unsigned *messageBuffer_path,__global float *cost,__global float *weight, __global bool *active)
{
    size_t id = get_global_id(0);
    unsigned source = sourceVertex[id];
    if(active[source])
    {
        messageBuffer_cost[messageWriteIndex[id]] = cost[source] + weight[id];
        messageBuffer_path[messageWriteIndex[id]] = source;
    }
}

__kernel void vertexCompute(__global unsigned *offset, __global float *messageBuffer_cost, __global unsigned *messageBuffer_path, __global unsigned *inEdges,__global float *cost,__global unsigned *path, __global bool *active, global bool* finished)
{
    size_t id = get_global_id(0);
    unsigned numMessages = inEdges[id];
    unsigned index = offset[id];
    float msg_min;
    unsigned out_path;
    combine(messageBuffer_cost,messageBuffer_path,numMessages, index, &msg_min, &out_path);
    if(cost[id] > msg_min)
    {
        cost[id] = msg_min;
        path[id] = out_path;
        active[id] = true;
        *finished = false;
    }
    else
        active[id] = false;
}

