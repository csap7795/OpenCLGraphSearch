__kernel void initializeBuffers(__global bool *maskArray, __global float *costArray,__global float *updatingCostArray,__global unsigned *predecessor,volatile __global int *semaphore, unsigned sourceVertex)
{

   size_t id = get_global_id(0);

   semaphore[id] = 0;

   if(sourceVertex == id)
   {
        maskArray[id] = 1;
        predecessor[id] = id;
        costArray[id] = 0.0f;
        updatingCostArray[id] = 0.0f;
   }

   else
   {
        maskArray[id] = 0;
        predecessor[id] = UINT_MAX;
        costArray[id] = FLT_MAX;
        updatingCostArray[id] = FLT_MAX;
   }

}

/* Dijkstra Kernel. Note that calculating value could lead to an OVERFLOW, if the cost to the path exceeds FLT_MAX.*/
__kernel void dijkstra1(__global unsigned *vertexArray, __global unsigned *edgeArray, __global float *weightArray, __global bool *maskArray, __global float *costArray, __global float *updatingCostArray,__global unsigned *predecessor,volatile __global int *semaphore)
{
    size_t id = get_global_id(0);

    if(maskArray[id] != 0)
    {
        maskArray[id] = 0;
        unsigned edgeStart = vertexArray[id];
        unsigned edgeEnd = vertexArray[id+1];

        for(unsigned edge=edgeStart;edge<edgeEnd;edge++)
        {
            unsigned nid = edgeArray[edge];

            // use atomics to avoid raceCondition
            //while(atomic_cmpxchg(semaphore+nid,0,1));
            bool not_enter = atomic_cmpxchg(semaphore+nid,0,1);

            if(!not_enter)
            {
                float value = updatingCostArray[nid];
                if(value > costArray[id] + weightArray[edge])
                {
                    updatingCostArray[nid] = costArray[id] + weightArray[edge];
                    //saves the predecessor node so that it is possible to backtrack the shortest paths
                    predecessor[nid] = id;
                }

                atomic_cmpxchg(semaphore+nid,1,0);
            }

            else
               edge--;

        }
    }
}

__kernel void dijkstra2(__global bool *maskArray, __global float *costArray, __global float *updatingCostArray,__global bool *finished)
{
    size_t id = get_global_id(0);

    if(costArray[id] > updatingCostArray[id])
    {
        costArray[id] = updatingCostArray[id];
        maskArray[id] = 1;
        *finished = false;
    }
    // Really need this?? Guess not.
    updatingCostArray[id] = costArray[id];
}
//Note: Unlike cl_ types in cl_platform.h, cl_bool is not guaranteed to be the same size as the bool in kernels.
__kernel void negativeCycle(__global unsigned *vertices,__global unsigned *edges,__global float *weight, __global float *costArray,__global short *negCycle, __global bool *detected)
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
            *detected = true;
        }
    }
}



