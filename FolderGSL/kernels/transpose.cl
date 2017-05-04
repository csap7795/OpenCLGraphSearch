__kernel void calcInEdges(__global unsigned* edges, volatile __global unsigned* inEdges)
{
    size_t id = get_global_id(0);
    unsigned writeIndex = edges[id];
    atomic_inc(inEdges+writeIndex);
}


__kernel void transpose(__global unsigned *vertices, __global unsigned *edges, __global unsigned *weight, volatile __global unsigned *offset, __global unsigned *newEdges, __global unsigned *newWeight)
{
    size_t id = get_global_id(0);
    unsigned edge_offset = vertices[id];
    unsigned num_neighbors = vertices[id+1] - edge_offset;
    for(int i = 0 ;i<num_neighbors;i++)
    {
        unsigned index = atomic_inc(&offset[edges[edge_offset + i]]);
        newEdges[index] = id;
        newWeight[index] = weight[edge_offset+i];
    }
}

__kernel void calcSrcDestVertices(__global unsigned *vertices,__global unsigned *edges, __global unsigned *src, __global unsigned *dest, __global unsigned *inEdges)
{
    size_t id = get_global_id(0);
    unsigned edge_offset = vertices[id];
    unsigned num_neighbors = vertices[id+1] - vertices[id];
    for(int i = 0 ;i<num_neighbors;i++)
    {
        src[edge_offset+i] = id;
        unsigned destination = edges[edge_offset+i];
        dest[edge_offset+i] = destination ;
        atomic_inc(inEdges+destination);
    }
}

__kernel void edge_transpose(volatile __global unsigned *offset, __global unsigned* source, __global unsigned *dest,__global unsigned *weight ,__global unsigned *newEdges, __global unsigned *newWeight)
{
    size_t id = get_global_id(0);
    unsigned offset_index = source[id];
    unsigned index = atomic_inc(&offset[offset_index]);
    newEdges[index] = dest[id];
    newWeight[index] = weight[id];
}
