#include <floyd_warshall.h>
#include <CL/cl.h>
#include <cl_utils.h>
#include <unistd.h>
#include <libgen.h>
#include <stdbool.h>

#define BLOCK_SIZE 8

static cl_program program;
static cl_context context;
static cl_command_queue command_queue;
static cl_device_id device;

static void build_kernel(size_t device_num)
{
    device = cluInitDevice(device_num,&context,&command_queue);
    //printf("%s\n\n",cluGetDeviceDescription(device,device_num));
    char* filename = "/floyd_warshall.cl";
    char cfp[1024];
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s",dirname(cfp),filename);
    filename = kernel_file;

    program = cluBuildProgramFromFile(context,device,kernel_file,NULL);
}

// The graph is available in a adjacency Matrix, where length denotes the number of vertices and device_num the device the user wants to work on.
void parallel_floyd_warshall(cl_float** matrix, unsigned length, size_t device_num)
{
    build_kernel(device_num);

    cl_int err;
    cl_mem matrix_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_float) * length * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating matrixbuffer");

    cl_mem out_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*length*length, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating out_buffer");

    // Copy Graph Data to their respective memory buffers
    for(int i = 0; i< length; i++)
        err = clEnqueueWriteBuffer(command_queue, matrix_buffer, CL_FALSE,i*sizeof(cl_float)*length, sizeof(cl_float) * length , matrix[i] , 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program,"matrixMul",&err);
        CLU_ERRCHECK(err,"Failed to create matrixMul kernel from program");

    cluSetKernelArguments(kernel,3,sizeof(cl_mem),(void*)&matrix_buffer,sizeof(cl_mem),(void*)&out_buffer, sizeof(unsigned), &length);

    // Execute the OpenCL kernel
    size_t globalSize[2] = {length,length};
    size_t localSize[2] = {BLOCK_SIZE, BLOCK_SIZE};
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL),"Error executing InitializeKernel");

    /*Checks if results are equal*/
    cl_float** test_matrix = createMatrix(matrix,length);
    cl_float** output_matrix = createMatrix(matrix,length);

    serial_floyd_warshall(test_matrix,length);

    for(int i = 0; i< length;i++)
        err = clEnqueueReadBuffer(command_queue,out_buffer,CL_TRUE,sizeof(cl_float) * length * i,sizeof(cl_float) * length,output_matrix[i],0,NULL,NULL);

    printf("%s\n",verify(test_matrix,output_matrix,length) ? "TRUE" : "FALSE");

    free(test_matrix);
    free(output_matrix);

    /*Finalize*/

    err = clFlush(command_queue);
    err = clFinish(command_queue);
    err = clReleaseKernel(kernel);
    err = clReleaseProgram(program);
    err = clReleaseMemObject(matrix_buffer);
    err = clReleaseMemObject(out_buffer);
    err = clReleaseCommandQueue(command_queue);
    err = clReleaseContext(context);

}

/*Serial Floyd Warshall, changes the value of the input matrix to the apsp output*/
void serial_floyd_warshall(cl_float** matrix, unsigned length)
{
    for(int i = 0; i<length;i++)
        for(int j = 0; j<length;j++)
            for(int k = 0; k<length;k++)
            {
                if(matrix[i][k] == CL_FLT_MAX || matrix[k][j] == CL_FLT_MAX)
                    continue;
                cl_float tmp = matrix[i][k] + matrix[k][j];
                if(matrix[i][j]>tmp)
                    matrix[i][j] = tmp;
            }
}

/*Returns a copy of matrix*/
cl_float** createMatrix(cl_float** matrix, unsigned length)
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

bool verify(cl_float** mat1, cl_float** mat2, unsigned length)
{
    for(int i = 0; i<length;i++)
        for(int j = 0; j<length;j++)
            if(mat1[i][j] != mat2[i][j])
                return false;

    return true;

}

