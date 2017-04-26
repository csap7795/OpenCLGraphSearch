#include <matrix.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <alloca.h>
#include <graph.h>

//Returns a copy of matrix in_mat resized to the next length equaly devided by blocksize, fill new entries with CL_FLT_MAX
cl_float** resizeFloatMatrix(cl_float** in_mat, unsigned length, unsigned blocksize)
{
    unsigned new_length = length + (blocksize - (length%blocksize));
    cl_float** ret_mat = createFloatMatrix(new_length);

    for(int i = 0; i<length;i++)
    {
        for(int j = 0; j<length;j++)
            ret_mat[i][j] = in_mat[i][j];

        for(int j = length;j<new_length;j++)
        {
            if(i == j)
                ret_mat[i][j] = 0;

            else
                ret_mat[i][j] = CL_FLT_MAX;
        }
    }

    for(int i = length; i<new_length;i++)
    {
        for(int j = 0;j<new_length;j++)
        {
            if(i == j)
                ret_mat[i][j] = 0;

            else
                ret_mat[i][j] = CL_FLT_MAX;
        }
    }
    return ret_mat;

}

//Fill the Path Matrix with CL_UINT_MAX
void fillPathMatrix(cl_uint** matrix, unsigned length)
{
    for(int i = 0; i<length;i++)
        for(int j = 0; j<length;j++)
            matrix[i][j] = CL_UINT_MAX;
}

// Copys the content from one matrix to another
void copyMatrixContent(cl_float** in, cl_float** out, unsigned length)
{
    for(int i = 0; i<length;i++)
        for(int j = 0; j<length;j++)
            out[i][j] = in[i][j];
}

/*Creates a matrix out of a Graph*/
cl_float** GraphToMatrix(Graph* graph)
{
    cl_float **mat = createFloatMatrix(graph->V);
    for(int i = 0; i<graph->V;i++)
        for(int j = graph->vertices[i]; j<graph->vertices[i+1]; j++)
            mat[i][graph->edges[j]] = graph->weight[j];

    for(int i = 0; i<graph->V;i++)
        for(int j = 0; j<graph->V;j++)
            if(i != j && mat[i][j] == 0)
                mat[i][j] = CL_FLT_MAX;

    return mat;
}

/*Returns a copy of matrix*/
cl_float** copyMatrix(cl_float** matrix, unsigned length)
{
    cl_float** out = (cl_float**)malloc(sizeof(cl_float*) * length);
    for(int i = 0; i<length;i++)
    {
        out[i] = (cl_float*)malloc(sizeof(cl_float)*length);
        for(int j = 0; j<length;j++)
            out[i][j] = matrix[i][j];
    }
    return out;
}

// Creates an empty matrix of type cl_float filled with zeroes
cl_float** createFloatMatrix(unsigned length)
{
    cl_float** out = (cl_float**)malloc(sizeof(cl_float*) * length);
    for(int i = 0; i<length;i++)
    {
        out[i] = (cl_float*)calloc(length,sizeof(cl_float));
    }
    return out;
}

// Creates an empty matrix of type cl_uint filled with zeroes
cl_uint** createUnsignedMatrix(unsigned length)
{
    cl_uint** out = (cl_uint**)malloc(sizeof(cl_uint*) * length);
    for(int i = 0; i<length;i++)
    {
        out[i] = (cl_uint*)calloc(length,sizeof(cl_uint));
    }
    return out;
}

//Frees a cl_float matrix
void freeFloatMatrix(cl_float** matrix, unsigned length)
{
    for(int i = 0; i<length;i++)
        free(matrix[i]);

    free (matrix);
}

//Frees a cl_uint matrix
void freeUnsignedMatrix(cl_uint** matrix, unsigned length)
{
    for(int i = 0; i<length;i++)
        free(matrix[i]);

    free (matrix);
}

void printMatrix(cl_float** mat, unsigned length)
{
    printf("MATRIX\n");
    for(int i = 0; i< length;i++)
    {
        for(int j = 0; j< length;j++)
        {
            printf("%.1f\t",mat[i][j]);
        }
        printf("\n");
    }
}

cl_float** getAdjMatrix(unsigned vertices,unsigned edges)
{
    cl_float** mat = (cl_float**)malloc(sizeof(cl_float*) * vertices);
    for(int i = 0; i<vertices;i++)
        mat[i] = (cl_float*)malloc(sizeof(cl_float)*vertices);

    for(int i = 0; i<vertices; i++)
        for(int j = 0; j<vertices; j++)
            if(i == j)
                mat[i][j] = 0.0f;
            else
                mat[i][j] = CL_FLT_MAX;

    srand(time(NULL));

    for(int k = 0; k<edges; k++)
    {
        int j;
        int i = rand()%vertices;
        while(( j = rand()%vertices) == i || mat[i][j] != CL_FLT_MAX);
        mat[i][j] = (float)((((double)rand())/(double)RAND_MAX)*100);

    }
    return mat;
}

cl_float** getTestMatrix(unsigned vertices)
{
    cl_float** mat = (cl_float**)malloc(sizeof(cl_float*) * vertices);
    for(int i = 0; i<vertices;i++)
        mat[i] = (cl_float*)malloc(sizeof(cl_float)*vertices);

    for(int i = 0; i<vertices; i++)
        for(int j = 0; j<vertices; j++)
            mat[i][j] = vertices*i + j;

    return mat;
}

bool float_matrix_equal(cl_float** mat1, cl_float** mat2, unsigned length)
{
    for(int i = 0; i<length;i++)
        for(int j = 0; j<length;j++)
        {
            if(!AlmostEqual2sComplement(mat1[i][j], mat2[i][j]))
                return false;
        }

    return true;
}

int index;
unsigned createPath(cl_uint** path_matrix, cl_uint* path, cl_uint s, cl_uint t)
{
    index = 0;
    calc_path_rec(path_matrix,path,s,t);
    int ret_val = index;
    index = 0;
    return ret_val;
}

void calc_path_rec(cl_uint** path_matrix, cl_uint* path, cl_uint i, cl_uint j)
{
    cl_uint k = path_matrix[i][j];
    if(k == CL_UINT_MAX)
    {
        path[index++] = j;
        return;
    }
    else
    {
        calc_path_rec(path_matrix,path,i,k);
        calc_path_rec(path_matrix,path,k,j);
    }
}

bool path_matrix_equal(cl_uint** mat1, cl_uint** mat2, unsigned length)
{
    cl_uint* path_mat1 = (cl_uint*)alloca(length*sizeof(cl_uint));
    cl_uint* path_mat2 = (cl_uint*)alloca(length*sizeof(cl_uint));

    for(int i = 0; i<length;i++)
        for(int j = 0; j<length;j++)
        {
            unsigned path_length1 = createPath(mat1,path_mat1,i,j);
            unsigned path_length2 = createPath(mat2,path_mat2,i,j);

            if(path_length1 != path_length2)
                return false;

            for(int k = 0; k<path_length1;k++)
                    if(path_mat1[k] != path_mat2[k])
                        return false;
        }

   return true;
}

int maxUlps = 10;
bool AlmostEqual2sComplement(cl_float A, cl_float B)
{

    // Make sure maxUlps is non-negative and small enough that the
    // default NAN won't compare as equal to anything
    assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
    int aInt = *(int*)&A;

    // Make aInt lexicographically ordered as a twos-complement int
    if (aInt < 0)
        aInt = 0x80000000 - aInt;

    // Make bInt lexicographically ordered as a twos-complement int
    int bInt = *(int*)&B;
    if (bInt < 0)
        bInt = 0x80000000 - bInt;

    int intDiff = abs(aInt - bInt);

    if (intDiff <= maxUlps)

        return true;

    else
        return false;
}
