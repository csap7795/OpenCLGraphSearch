void calculateSDBlock(__local float (*mat)[BLOCK_SIZE],__local unsigned (*path)[BLOCK_SIZE],int lx, int ly, unsigned min, unsigned max);
void calculate1BDependentRow(__local float (*mat)[BLOCK_SIZE],__local float(*dep)[BLOCK_SIZE],__local unsigned (*path)[BLOCK_SIZE],int lx , int ly, unsigned min, unsigned max);
void calculate1BDependentCol(__local float (*mat)[BLOCK_SIZE],__local float (*dep)[BLOCK_SIZE],__local unsigned (*path)[BLOCK_SIZE], int lx , int ly, unsigned min, unsigned max);
void calculate2BDependent(__local float (*mat)[BLOCK_SIZE],__local float (*dep_row)[BLOCK_SIZE],__local float (*dep_col)[BLOCK_SIZE],__local unsigned (*path)[BLOCK_SIZE], int lx , int ly, unsigned min, unsigned max);

__kernel void global_floyd_warshall(__global float* in, __global float* out, __global unsigned* path, int width, int i)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);

    out[x*width+y] = in[x*width+y];

    if(x != i && y != i)
	{
		float cost = in[x*width+i] + in[i*width+y];

		if(in[x*width+y] > cost)
		{
			out[x*width+y] = cost;
			path[x*width+y] = i;
        }
	}
}

__kernel void global_floyd_warshall_gpu(__global float* in, __global float* out, __global unsigned* path, int width, int i)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
    out[x+width*y] = in[x+width*y];

    if(x != i && y != i)
	{
		float cost = in[x+width*i] + in[i+width*y];

		if(in[x+width*y] > cost)
		{
			out[x+width*y] = cost;
			path[x+width*y] = i;
        }
	}
}

__kernel void phase1(__global float* in,__global unsigned* gpath, unsigned width, int i)
{

	size_t gx = get_group_id(0);
	size_t gy = get_group_id(1);

    // only work on the group which is in the diagonal of the matrix
	if(!(gx == i && gy ==i))
            return;

    // declare local variables
    __local float matrix[BLOCK_SIZE][BLOCK_SIZE];
    __local unsigned path[BLOCK_SIZE][BLOCK_SIZE];

	size_t lx = get_local_id(0);
	size_t ly = get_local_id(1);

    // Calculate the offset where to read from the data for the actual block
	int block_offset = BLOCK_SIZE*width*gx + BLOCK_SIZE*BLOCK_SIZE*gy;

    // Min and max denote boundaries for the k in floyd warshall algorithm
    unsigned min = i*BLOCK_SIZE;
    unsigned max = (i+1)*BLOCK_SIZE;

    //copy work to local
	matrix[lx][ly] = in[block_offset+lx*BLOCK_SIZE+ly];
	path[lx][ly] = gpath[block_offset+lx*BLOCK_SIZE+ly];

    //Calculate the new Matrix
    calculateSDBlock(&matrix[0],&path[0],lx,ly,min,max);

    //copy work back to global
    in[block_offset+lx*BLOCK_SIZE+ly] = matrix[lx][ly];
    gpath[block_offset+lx*BLOCK_SIZE+ly] = path[lx][ly];
}

__kernel void phase2(__global float* in, __global unsigned* gpath, unsigned width, int i)
{
	size_t gx = get_group_id(0);
	size_t gy = get_group_id(1);

	size_t lx = get_local_id(0);
	size_t ly = get_local_id(1);

    // Only work on tiles which are in the same row or column of the tile [i,i]
	if(gx == i && gy == i)
        return;
    if(gx != i && gy != i)
        return;

    // Declare local variables
    __local float matrix[BLOCK_SIZE][BLOCK_SIZE];
    __local float dep[BLOCK_SIZE][BLOCK_SIZE];
    __local unsigned path[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate the offset where to read from the data for the actual block
	int block_offset = BLOCK_SIZE*width*gx + BLOCK_SIZE*BLOCK_SIZE*gy;

    // Min and max denote boundaries for the k in floyd warshall algorithm
    unsigned min = i*BLOCK_SIZE;
    unsigned max = (i+1)*BLOCK_SIZE;

    //copy work to local
	matrix[lx][ly] = in[block_offset+lx*BLOCK_SIZE+ly];
	path[lx][ly] = gpath[block_offset+lx*BLOCK_SIZE+ly];
    dep[lx][ly] = in[i*BLOCK_SIZE*width+i*BLOCK_SIZE*BLOCK_SIZE+lx*BLOCK_SIZE+ly];

    if(gx == i)
        calculate1BDependentRow(&matrix[0],&dep[0],&path[0],lx,ly,min,max);

    else if(gy == i)
        calculate1BDependentCol(&matrix[0],&dep[0],&path[0],lx,ly,min,max);

    //copy work back to global
    in[block_offset+lx*BLOCK_SIZE+ly] = matrix[lx][ly];
    gpath[block_offset+lx*BLOCK_SIZE+ly] = path[lx][ly];
}

__kernel void phase3(__global float* in, __global unsigned* gpath, unsigned width, int i)
{

	size_t gx = get_group_id(0);
	size_t gy = get_group_id(1);

	size_t lx = get_local_id(0);
	size_t ly = get_local_id(1);

    // Work on all tiles which are not in the same row or column of tile [i,i]
	if(gx == i || gy == i)
        return;

    // Declare local variables
    __local float matrix[BLOCK_SIZE][BLOCK_SIZE];
    __local float dep_row[BLOCK_SIZE][BLOCK_SIZE];
    __local float dep_col[BLOCK_SIZE][BLOCK_SIZE];
    __local unsigned path[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate the offset where to read from the data for the actual block
	int block_offset = BLOCK_SIZE*width*gx + BLOCK_SIZE*BLOCK_SIZE*gy;

    // Min and max denote boundaries for the k in floyd warshall algorithm
    unsigned min = i*BLOCK_SIZE;
    unsigned max = (i+1)*BLOCK_SIZE;

    //copy work to local
	matrix[lx][ly] = in[block_offset+lx*BLOCK_SIZE+ly];
    dep_row[lx][ly] = in[gx*BLOCK_SIZE*width+i*BLOCK_SIZE*BLOCK_SIZE+lx*BLOCK_SIZE+ly];
    dep_col[lx][ly] = in[i*BLOCK_SIZE*width+gy*BLOCK_SIZE*BLOCK_SIZE+lx*BLOCK_SIZE+ly];
    path[lx][ly] = gpath[block_offset+lx*BLOCK_SIZE+ly];
    calculate2BDependent(&matrix[0],&dep_row[0],&dep_col[0],&path[0],lx,ly,min,max);

    //copy work back to global
    in[block_offset+lx*BLOCK_SIZE+ly] = matrix[lx][ly];
    gpath[block_offset+lx*BLOCK_SIZE+ly] = path[lx][ly];
}

//Calculate the Self Dependent Block
void calculateSDBlock(__local float (*mat)[BLOCK_SIZE],__local unsigned (*path)[BLOCK_SIZE],int lx, int ly, unsigned min, unsigned max)
{
    for(unsigned k = min; k < max; k++)
    {
	// a path from lx to ly won't lead over lx
        if(lx != k%BLOCK_SIZE && ly != k%BLOCK_SIZE)
        {
            float cost = mat[lx][k%BLOCK_SIZE] + mat[k%BLOCK_SIZE][ly];//in[index1] + in[index2];
            if(mat[lx][ly] > cost)
            {
                mat[lx][ly] = cost;
                path[lx][ly] = k;
            }

        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
// Calculate the blocks which depend on just one other block in the same row
void calculate1BDependentRow(__local float (*mat)[BLOCK_SIZE],__local float(*dep)[BLOCK_SIZE],__local unsigned (*path)[BLOCK_SIZE],int lx , int ly, unsigned min, unsigned max)
{
    for(unsigned k = min; k<max;k++)
    {
        float cost = dep[lx][k%BLOCK_SIZE] + mat[k%BLOCK_SIZE][ly];
        if(mat[lx][ly] > cost)
        {
            mat[lx][ly] = cost;
            path[lx][ly] = k;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Calculate the blocks which depend on just one other block in the same column
void calculate1BDependentCol(__local float (*mat)[BLOCK_SIZE],__local float (*dep)[BLOCK_SIZE],__local unsigned (*path)[BLOCK_SIZE], int lx , int ly, unsigned min, unsigned max)
{
    for(unsigned k = min; k<max;k++)
    {
        float cost = mat[lx][k%BLOCK_SIZE] + dep[k%BLOCK_SIZE][ly];
        if(mat[lx][ly] > cost)
        {
           mat[lx][ly] = cost;
           path[lx][ly] = k;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Calculate the blocks which depend on 2 blocks, one in the same row, one in the same col.
void calculate2BDependent(__local float (*mat)[BLOCK_SIZE],__local float (*dep_row)[BLOCK_SIZE],__local float (*dep_col)[BLOCK_SIZE],__local unsigned (*path)[BLOCK_SIZE], int lx , int ly, unsigned min, unsigned max)
{
    for(unsigned k = min; k<max;k++)
    {
        float cost = dep_row[lx][k%BLOCK_SIZE] + dep_col[k%BLOCK_SIZE][ly];
        if(mat[lx][ly] > cost)
        {
            mat[lx][ly] = cost;
            path[lx][ly] = k;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Reads data from a row-organized matrix and saves it in a tile-organized way, where each tile is of size BLOCK_SIZE*BLOCK_SIZE
__kernel void row_to_tile_layout(__global float* in, __global float* out, unsigned width)
{
    unsigned idx = get_global_id(0);
    unsigned idy = get_global_id(1);

    unsigned tile_offset = idx/BLOCK_SIZE * BLOCK_SIZE*width + idy/BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
    unsigned index = tile_offset + idx%BLOCK_SIZE * BLOCK_SIZE + idy%BLOCK_SIZE;

    out[index] = in[idx * width + idy];
}

// The opposite of row_to_tile_layout, takes a matrix organized in tiles and writes it in a row-organized way into out
__kernel void tile_to_row_layout(__global float* in, __global float* out, unsigned width)
{
    unsigned idx = get_global_id(0);
    unsigned idy = get_global_id(1);

    unsigned tile_offset = idx/BLOCK_SIZE * BLOCK_SIZE*width + idy/BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
    unsigned index = tile_offset + idx%BLOCK_SIZE * BLOCK_SIZE + idy%BLOCK_SIZE;

    out[idx*width+idy] = in[index];
}

// Same as tile_to_row_layout, except it can be used for the cost-array and the path-array
__kernel void tile_to_row_layout_path(__global float* in, __global float* out, __global unsigned* path_in, __global unsigned* path_out, unsigned width)
{
    unsigned idx = get_global_id(0);
    unsigned idy = get_global_id(1);

    unsigned tile_offset = idx/BLOCK_SIZE * BLOCK_SIZE*width + idy/BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
    unsigned index = tile_offset + idx%BLOCK_SIZE * BLOCK_SIZE + idy%BLOCK_SIZE;

    out[idx*width+idy] = in[index];
    path_out[idx*width+idy] = path_in[index];
}

//Simple kernel to transpose a matrix which contains float values
__kernel void float_matrix_transpose(__global float* in, __global float* out,unsigned width)
{
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);

    out[x+width*y] = in[x*width+y];
}

//Simple kernel to transpose a matrix which contains unsigned values
__kernel void unsigned_matrix_transpose(__global unsigned* in, __global unsigned* out, unsigned width)
{
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);

    out[x+width*y] = in[x*width+y];
}

