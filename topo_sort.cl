
//Theoretically you could use another bool array to check wheter MessageBuffer was changed for current Vertex
bool combine(__global bool *messageBuffer, unsigned numMessages, unsigned index)
{
    for(int i = 0; i<numMessages;i++)
    {
        unsigned new_index = index + (i*GROUP_NUM);
        if(messageBuffer[new_index] == 0)
            return false;
    }
    messageBuffer[index] = 0;
    return true;
}
__kernel void initialize(__global unsigned *inEdges,__global unsigned *order,__global bool *active, __global bool *finished)
{
    size_t id = get_global_id(0);
    if(inEdges[id] == 0)
    {
        order[id] = 0;
        active[id] = true;
        *finished = false;
    }
    else
    {
        active[id] = false;
    }
}

__kernel void edgeCompute(__global unsigned *inEdges,__global unsigned *sourceVertex, __global unsigned *messageWriteIndex, __global bool *messageBuffer, __global bool *active)
{
    size_t id = get_global_id(0);
    unsigned source = sourceVertex[id];
    // can this be achieved coalescing?
    if(active[source])
    {
        messageBuffer[messageWriteIndex[id]] = 1;
    }
}

__kernel void vertexCompute(__global unsigned *offset, __global bool *messageBuffer, __global unsigned *numEdges,__global unsigned *order, __global bool *active, global bool* finished, unsigned current_order)
{
    size_t id = get_global_id(0);
    unsigned numMessages = numEdges[id];
    unsigned index = offset[id/GROUP_NUM];
    index += id%GROUP_NUM;
    bool flag;

    if(numMessages == 0)
        flag = false;
    else
        flag = combine(messageBuffer,numMessages, index);

    if(flag)
    {
        order[id] = current_order;
        active[id] = true;
        *finished = false;
    }
    else
        active[id] = false;
}

