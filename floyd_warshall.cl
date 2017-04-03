#define BLOCK_SIZE 8

__kernel void matrixMul(
    __global float* A, __global float* C, int width)
    {
        // Block index
        int bx = get_group_id(0);
        int by = get_group_id(1);

        //Thread index
        int tx = get_local_id(0);
        int ty = get_local_id(1);

        int c = width*BLOCK_SIZE*by + BLOCK_SIZE*bx;
        float Csub = A[c + width*ty +tx];
        float tmp;

        //Index of the first sub-matrix of A processed by the block
        int aBegin = width * BLOCK_SIZE * by;
        //Index of last sub-matrix of A processed by the block
        int aEnd = aBegin + width -1;
        //Stepsize to iterate through the sub-matrices of A
        int aStep = BLOCK_SIZE;
        //Index of the first sub-matrix of B processed by the block ( B is in this case the same as A)
        int bBegin = BLOCK_SIZE * bx;
        //Step size used to iterate through the sub-matrices of B
        int bStep = BLOCK_SIZE * width;
        //Loop over all the sub-matrices of A and B
        //required to compute the block sub-matrix
        for(int a = aBegin, b = bBegin; a<=aEnd;a+=aStep,b+=bStep)
        {

            // Local memory As used to store the sub-matrix of A
            __local float As[BLOCK_SIZE][BLOCK_SIZE];

            // Local memory array Bs used to store the sub-matrix of B
            __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

            //Load the matrices from global memory to local memory
            As[ty][tx] = A[a + width * ty + tx];
            Bs[ty][tx] = A[b + width * ty +tx];

            //Synchronize to make sure the matrices are loaded
            barrier(CLK_LOCAL_MEM_FENCE);

            //Compute the shortest path of the Submatrices
            //Each thread computes one element
            //of the block sub-matrix
            for(int k = 0; k<BLOCK_SIZE;++k)
            {
                if( As[ty][k] == FLT_MAX || Bs[k][tx] == FLT_MAX)
                    continue;
                tmp = As[ty][k] + Bs[k][tx];
                if(Csub>tmp)
                    Csub = tmp;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

        }
        C[c+ width * ty + tx] = Csub;
    }

