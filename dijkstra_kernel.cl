__kernel void initializeBuffers(__global bool *maskArray, __global float *costArray,__global float *updatingCostArray,__global float *semaphore, unsigned sourceVertex)
{

   int id = get_global_id(0);

   semaphore[id] = 0;

   if(sourceVertex == id)
   {
        maskArray[id] = 1;
        costArray[id] = 0.0f;
        updatingCostArray[id] = 0.0f;
   }

   else
   {
        maskArray[id] = 0;
        costArray[id] = FLT_MAX;
        updatingCostArray[id] = FLT_MAX;
   }

}

__kernel void dijkstra1(__global unsigned *vertexArray, __global unsigned *edgeArray, __global float *weightArray, __global bool *maskArray, __global float *costArray, __global float *updatingCostArray,__global volatile int *semaphore)
{
    unsigned id = get_global_id(0);

    if(maskArray[id] != 0)
    {
        maskArray[id] = 0;
        unsigned edgeStart = vertexArray[id];
        unsigned edgeEnd = vertexArray[id+1];

        for(unsigned edge=edgeStart;edge<edgeEnd;edge++)
        {
            unsigned nid = edgeArray[edge];

            // use atomics to avoid raceCondition

            while(atomic_cmpxchg(semaphore+nid,0,1));

            float value = updatingCostArray[nid];

            //compute things to spend time before comparison

	        /*for(int i = 0; i< 10000;i++)
            {
                i%2 == 0 ? maskArray[id]++ : maskArray[id]--;
            }*/

            if(value > costArray[id] + weightArray[edge])
            {
                updatingCostArray[nid] = costArray[id] + weightArray[edge];
                //if you want to save the path, save id in an extra predecessor array on index nid
            }

            atomic_cmpxchg(semaphore+nid,1,0);
        }
    }
}

__kernel void dijkstra2(__global bool *maskArray, __global float *costArray, __global float *updatingCostArray,__global bool *finished)
{
    int id = get_global_id(0);

    if(costArray[id] > updatingCostArray[id])
    {
        costArray[id] = updatingCostArray[id];
        maskArray[id] = 1;
        *finished = false;
    }
    // Really need this?? Guess not.
    //updatingCostArray[id] = costArray[id];
}



