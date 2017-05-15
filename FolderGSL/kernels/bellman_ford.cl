__kernel void negativeCycle(__global unsigned *vertices,__global unsigned *edges,__global float *weight, __global float *costArray,__global short *negCycle, __global int *detected)
{
    size_t id = get_global_id(0);
    unsigned edge_offset = vertices[id];
    unsigned num_neighbors = vertices[id+1] - edge_offset;
    negCycle[id] = 0;
    for(int i = 0; i<num_neighbors;i++)
    {
        unsigned nid = edges[edge_offset+i];
        if(costArray[id] + weight[nid] < costArray[nid])
        {
            negCycle[id] = 1;
            *detected = 1;
        }
    }
}
