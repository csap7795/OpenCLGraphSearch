#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_fp64: enable

struct groupmem_t {

    unsigned levels[CHUNK_SIZE];
    unsigned nodes [CHUNK_SIZE+1];

};

void memcpy_SIMD (unsigned W_OFF, int cnt, __local unsigned* dest, __global unsigned* src)
{
    for(int IDX = W_OFF; IDX<cnt; IDX+=W_SZ)
    {
        dest[IDX] = src[IDX];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

}

void expand_bfs_SIMD(unsigned W_OFF,int cnt,__global unsigned* edges, __global unsigned* levels, unsigned curr, __global bool* finished)
{
    for(int IDX = W_OFF;IDX<cnt;IDX+=W_SZ)
    {
        unsigned v = edges[IDX];
        if(levels[v] == INT_MAX)
        {
            levels[v] = curr+1;
            *finished = false;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void workgroup_bfs_kernel(__global unsigned *vertices, __global unsigned *edges, __global unsigned *levels, __global bool *finished, unsigned curr)
{
    size_t global_thread_id = get_global_id(0);
    size_t local_thread_id = get_local_id(0);
    size_t group_id = get_group_id(0);

    __local struct groupmem_t my;
    unsigned start_index = group_id * CHUNK_SIZE;

    // copy my work to local
    memcpy_SIMD(local_thread_id,CHUNK_SIZE,my.levels,&levels[start_index]);
    memcpy_SIMD(local_thread_id,CHUNK_SIZE+1,my.nodes,&vertices[start_index]);

    // iterate over my work
    for( unsigned v = 0; v < CHUNK_SIZE; v++)
    {
        if(my.levels[v] == curr)
        {
            unsigned num_nbr = my.nodes[v+1] - my.nodes[v];
            __global unsigned* nbrs = &edges[my.nodes[v]];
            expand_bfs_SIMD(local_thread_id,num_nbr,nbrs,levels,curr,finished);
        }
    }

}

__kernel void initialize_bfs_kernel(__global int *levels,__global bool *finished, unsigned src)
{
    size_t id = get_global_id(0);
    if(id == src)
    {
        levels[src] = 0;
        *finished = false;
    }

    else
        levels[id] = INT_MAX;
}


__kernel void baseline_kernel(__global unsigned *vertices, __global unsigned *edges, __global unsigned *levels,__global bool* finished,  unsigned curr)
{
	size_t v = get_global_id(0);

	if(levels[v] == curr)
	{
        unsigned num_nbrs = vertices[v+1] - vertices[v];

        for(int i = 0; i<num_nbrs;i++)
        {
            unsigned w = edges[vertices[v]+i];
            if(levels[w] == INT_MAX)
            {
                *finished = false;
                levels[w] = curr+1;
            }
        }
    }
}
