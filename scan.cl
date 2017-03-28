__kernel void sum_scan(
    __global uint *g_odata, __global uint *g_idata, __global uint *sums)
    {
        __local uint temp[GROUP_SIZE];
        size_t lid = get_local_id(0);
        size_t gid = get_global_id(0);
        //size_t gid = get_group_id(0) * get_local_size(0) + lid;
        int offset = 1;
        temp[2*lid] = g_idata[2*gid];
        temp[2*lid+1] = g_idata[2*gid+1];

        for(int d = GROUP_SIZE>>1;d>0;d>>=1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lid<d){
                int ai = offset*(2*lid+1)-1;
                int bi = offset*(2*lid +2) -1;
                temp[bi] += temp[ai];

            }
            offset *=2;
        }
        if(lid == 0) {temp[GROUP_SIZE-1] = 0;}

        for(int d = 1 ; d<GROUP_SIZE; d*=2)
        {
            offset>>=1;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lid < d)
            {
                int ai = offset*(2*lid+1)-1;
                int bi = offset*(2*lid+2)-1;
                uint t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid == 0) {sums[get_group_id(0)] = temp[GROUP_SIZE-1] + g_idata[GROUP_SIZE-1];}

        barrier(CLK_LOCAL_MEM_FENCE);
        g_odata[2*gid] = temp[2*lid];
        g_odata[2*gid+1] = temp[2*lid+1];
    }

    __kernel void add_offsets(__global uint *g_odata, __global uint *g_idata, __global uint *offset)
    {
        size_t gid = get_global_id(0);
        size_t group_id = gid/GROUP_SIZE;
        g_odata[gid] = g_idata[gid] + offset[group_id];
    }

    #define LENGTH 1024

    __kernel void scan(__global uint *g_odata, __global uint *g_idata, int length)
    {
        __local uint temp[LENGTH*2];
        int id = get_global_id(0);
        int pout = 0, pin = 1;

        temp[id] = (id > 0) ? g_idata[id-1] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int offset = 1; offset<LENGTH;offset<<=1)
        {
            pout = 1-pout;
            pin = 1-pout;
            if(id >= offset)
                temp[pout*length+id] = temp[pin*length+id] + temp[pin*length +id -offset];
             else temp[pout*length+id] = temp[pin*length+id];
             barrier(CLK_LOCAL_MEM_FENCE);
        }
        g_odata[id] = temp[pout*length +id];
    }
