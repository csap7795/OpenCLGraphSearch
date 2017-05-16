#include <floyd_warshall.h>
#include <benchmark_utils.h>
#include <matrix.h>

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
    char tmp[1024];
    sprintf(tmp, "-DBLOCK_SIZE=%i",BLOCK_SIZE);
    char cfp[1024];
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s%s",dirname(dirname(cfp)),"/kernels",filename);
    program = cluBuildProgramFromFile(context,device,kernel_file,tmp);
}

// The graph is available in a adjacency Matrix, where length denotes the number of vertices and device_num the device the user wants to work on.
void parallel_floyd_warshall_row(cl_float** in_matrix, cl_float** out_matrix, cl_uint** out_path, unsigned length, size_t device_num, unsigned long * time)
{
    build_kernel(device_num);

    unsigned long start_time = time_ms();

    //cl_float** test_matrix = copyMatrix(matrix,length);

    cl_int err;
    cl_mem matrix_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * length * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating matrixbuffer");
    cl_mem out_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*length*length, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating out_buffer");
    cl_mem path_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*length*length, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating out_buffer");

    //COPY CL_UINT_MAX TO PATH_BUFFER FOR EVERY ELEMENT
    cl_uint* uintMax = (cl_uint*) malloc(sizeof(cl_uint)* length*length);
    for(int i = 0; i<length*length;i++)
    {
        uintMax[i] = CL_UINT_MAX;
    }
    clEnqueueWriteBuffer(command_queue,path_buffer,CL_TRUE,0,sizeof(cl_uint)*length*length,uintMax,0,NULL,NULL);
    free(uintMax);

    // Copy Graph Data to their respective memory buffers
    for(int i = 0; i< length; i++)
        err = clEnqueueWriteBuffer(command_queue, matrix_buffer, CL_FALSE,i*sizeof(cl_float)*length, sizeof(cl_float) * length , in_matrix[i] , 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program,"global_floyd_warshall",&err);
        CLU_ERRCHECK(err,"Failed to create floydWarshall kernel from program");


    // Execute the OpenCL kernel
    size_t globalSize[2] = {length,length};
    /*for(i = 0; i<length;i++)
    {
        if(i%2 == 0)
            cluSetKernelArguments(kernel,4,sizeof(cl_mem),(void*)&matrix_buffer,sizeof(cl_mem),(void*)&out_buffer,sizeof(unsigned), &length,sizeof(cl_int),&i);
        if(i%2 == 1)
            cluSetKernelArguments(kernel,4,sizeof(cl_mem),(void*)&out_buffer,sizeof(cl_mem),(void*)&matrix_buffer,sizeof(unsigned), &length,sizeof(cl_int),&i);
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");
    }*/
    cl_mem help;
    for(cl_int i = 0; i<length;i++)
    {
        if(i%2 == 0)
        {
            help = out_buffer;
            cluSetKernelArguments(kernel,5,sizeof(cl_mem),(void*)&matrix_buffer,sizeof(cl_mem),(void*)&out_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(unsigned), &length,sizeof(cl_int),&i);
        }
        if(i%2 == 1)
        {
            help = matrix_buffer;
            cluSetKernelArguments(kernel,5,sizeof(cl_mem),(void*)&out_buffer,sizeof(cl_mem),(void*)&matrix_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(unsigned), &length,sizeof(cl_int),&i);

        }
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    }

    for(int i = 0; i< length;i++)
    {
        err = clEnqueueReadBuffer(command_queue,help,CL_FALSE,sizeof(cl_float) * length * i,sizeof(cl_float) * length,out_matrix[i],0,NULL,NULL);
        err = clEnqueueReadBuffer(command_queue,path_buffer,CL_FALSE,sizeof(cl_uint) * length * i,sizeof(cl_uint) * length,out_path[i],0,NULL,NULL);
    }

    // Wait until all queued commands finish
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    CLU_ERRCHECK(err,"Error finishing command_queue");

    // Save time if requested
    if(time != NULL)
        *time = time_ms()-start_time;

    //Finalize
    err = clReleaseKernel(kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(matrix_buffer);
    err |= clReleaseMemObject(out_buffer);
    err |= clReleaseMemObject(path_buffer);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err,"Failed finalizing OpenCL");

}

void parallel_floyd_warshall_column(cl_float** in_matrix, cl_float** out_matrix, cl_uint** out_path, unsigned length, size_t device_num, unsigned long *time)
{
    build_kernel(device_num);
    unsigned long start_time = time_ms();

    cl_int err;
    cl_mem row_matrix_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * length * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating matrixbuffer");
    cl_mem matrix_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * length * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating matrixbuffer");
    cl_mem out_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*length*length, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating out_buffer");
    cl_mem row_path_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*length*length, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating out_buffer");
    cl_mem path_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*length*length, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating out_buffer");

    //COPY CL_UINT_MAX TO PATH_BUFFER FOR EVERY ELEMENT
    cl_uint* uintMax = (cl_uint*) malloc(sizeof(cl_uint)* length*length);
    for(int i = 0; i<length*length;i++)
    {
        uintMax[i] = CL_UINT_MAX;
    }
    clEnqueueWriteBuffer(command_queue,path_buffer,CL_TRUE,0,sizeof(cl_uint)*length*length,uintMax,0,NULL,NULL);
    free(uintMax);

    // Copy Graph Data to their respective memory buffers
    for(int i = 0; i< length; i++)
        err = clEnqueueWriteBuffer(command_queue, row_matrix_buffer, CL_FALSE,i*sizeof(cl_float)*length, sizeof(cl_float) * length , in_matrix[i] , 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program,"global_floyd_warshall_gpu",&err);
        CLU_ERRCHECK(err,"Failed to create floydWarshall kernel from program");

    cl_kernel float_transpose_kernel = clCreateKernel(program,"float_matrix_transpose",&err);
        CLU_ERRCHECK(err,"Failed to create float_matrix_transpose kernel from program");

    cl_kernel unsigned_transpose_kernel = clCreateKernel(program,"unsigned_matrix_transpose",&err);
        CLU_ERRCHECK(err,"Failed to create unsigned_matrix_transpose kernel from program");

    //Transpose matrix so that it can be processed columnwise
    size_t globalSize[2] = {length,length};
    cluSetKernelArguments(float_transpose_kernel,3,sizeof(cl_mem),(void*)&row_matrix_buffer,sizeof(cl_mem),(void*)&matrix_buffer,sizeof(unsigned),&length);
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, float_transpose_kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");


    // Execute the OpenCL kernel

    cl_mem* help;
    for(cl_int i = 0; i<length;i++)
    {
        if(i%2 == 0)
        {
            help = &out_buffer;
            cluSetKernelArguments(kernel,5,sizeof(cl_mem),(void*)&matrix_buffer,sizeof(cl_mem),(void*)&out_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(unsigned), &length,sizeof(cl_int),&i);
        }
        if(i%2 == 1)
        {
            help = &matrix_buffer;
            cluSetKernelArguments(kernel,5,sizeof(cl_mem),(void*)&out_buffer,sizeof(cl_mem),(void*)&matrix_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(unsigned), &length,sizeof(cl_int),&i);

        }
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    }

    //Transpose the matrices back
    cluSetKernelArguments(float_transpose_kernel,3,sizeof(cl_mem),(void*)help,sizeof(cl_mem),(void*)&row_matrix_buffer,sizeof(unsigned),&length);
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, float_transpose_kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL),"Error executing FloatTransposeKernel");

    cluSetKernelArguments(unsigned_transpose_kernel,3,sizeof(cl_mem),(void*)&path_buffer,sizeof(cl_mem),(void*)&row_path_buffer,sizeof(unsigned),&length);
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, unsigned_transpose_kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL),"Error executing UnsignedTransposeKernel");

    for(int i = 0; i< length;i++)
    {
        err = clEnqueueReadBuffer(command_queue,row_matrix_buffer,CL_FALSE,sizeof(cl_float) * length * i,sizeof(cl_float) * length,out_matrix[i],0,NULL,NULL);
        err = clEnqueueReadBuffer(command_queue,row_path_buffer,CL_FALSE,sizeof(cl_uint) * length * i,sizeof(cl_uint) * length,out_path[i],0,NULL,NULL);
    }

    // Wait until all queued commands finish
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    CLU_ERRCHECK(err,"Error finishing command_queue");

    if(time != NULL)
       *time = time_ms()-start_time;

    //Finalize
    err = clReleaseKernel(kernel);
    err |= clReleaseKernel(float_transpose_kernel);
    err |= clReleaseKernel(unsigned_transpose_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(row_matrix_buffer);
    err |= clReleaseMemObject(matrix_buffer);
    err |= clReleaseMemObject(out_buffer);
    err |= clReleaseMemObject(path_buffer);
    err |= clReleaseMemObject(row_path_buffer);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err,"Failed finalizing OpenCL");

}


void parallel_floyd_warshall_workgroup(cl_float** in_matrix, cl_float** out_matrix, cl_uint** out_path, unsigned length, size_t device_num, unsigned long *time)
{
    build_kernel(device_num);
    unsigned out_length = length;
    cl_float** matrix;
    //Check if the matrix can equally devided into Blocks of size BLOCK_SIZE, if not resize it
    if(length % BLOCK_SIZE != 0)
    {
        matrix = resizeFloatMatrix(in_matrix,length,BLOCK_SIZE);
        length += (BLOCK_SIZE - (length%BLOCK_SIZE));
    }
    else
        matrix = copyMatrix(in_matrix,length);

    cl_int err;
    cl_mem matrix_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * length * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating matrixbuffer");
    cl_mem tile_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * length * length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating matrixbuffer");
    cl_mem out_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*length*length, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating out_buffer");
    cl_mem path_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*length*length, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating path_buffer");
    cl_mem path_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint)*length*length, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating path_buffer");

    // Copy Matrix Data to its respective memory buffer
    for(int i = 0; i< length; i++)
        err = clEnqueueWriteBuffer(command_queue, matrix_buffer, CL_FALSE,i*sizeof(cl_float)*length, sizeof(cl_float) * length , matrix[i] , 0, NULL, NULL);

    // Copy CL_UINT_MAX to path_buffer for every element
    cl_uint* uintMax = (cl_uint*) malloc(sizeof(cl_uint)* length*length);
    for(int i = 0; i<length*length;i++)
    {
        uintMax[i] = CL_UINT_MAX;
    }
    clEnqueueWriteBuffer(command_queue,path_buffer,CL_TRUE,0,sizeof(cl_uint)*length*length,uintMax,0,NULL,NULL);
    free(uintMax);

    // Create the Kernels
    cl_kernel phase1 = clCreateKernel(program,"phase1",&err);
        CLU_ERRCHECK(err,"Failed to create phase1 kernel from program");

    cl_kernel phase2 = clCreateKernel(program,"phase2",&err);
        CLU_ERRCHECK(err,"Failed to create phase2 kernel from program");

    cl_kernel phase3 = clCreateKernel(program,"phase3",&err);
        CLU_ERRCHECK(err,"Failed to create phase3 kernel from program");

    cl_kernel row_to_tile_kernel = clCreateKernel(program,"row_to_tile_layout",&err);
        CLU_ERRCHECK(err,"Failed to create row_to_tile kernel from program");

    cl_kernel tile_to_row_kernel = clCreateKernel(program,"tile_to_row_layout_path",&err);
        CLU_ERRCHECK(err,"Failed to create tile_to_row kernel from program");

    /* Set the Kernel Arguments for the layout transformation kernels*/
    cluSetKernelArguments(row_to_tile_kernel,3,sizeof(cl_mem),(void*)&matrix_buffer,sizeof(cl_mem),(void*)&tile_buffer, sizeof(cl_uint), &length);
    cluSetKernelArguments(tile_to_row_kernel,5,sizeof(cl_mem),(void*)&tile_buffer,sizeof(cl_mem),(void*)&out_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(cl_mem),(void*)&path_out, sizeof(cl_uint), &length);

    size_t globalSize[2] = {length,length};
    size_t localSize[2] = {BLOCK_SIZE,BLOCK_SIZE};
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue,row_to_tile_kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    unsigned long start_time = time_ms();
    for(cl_int i = 0; i<length/BLOCK_SIZE;i++)
    {
        cluSetKernelArguments(phase1,4,sizeof(cl_mem),(void*)&tile_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(unsigned), &length,sizeof(cl_int),&i);
        cluSetKernelArguments(phase2,4,sizeof(cl_mem),(void*)&tile_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(unsigned), &length,sizeof(cl_int),&i);
        cluSetKernelArguments(phase3,4,sizeof(cl_mem),(void*)&tile_buffer,sizeof(cl_mem),(void*)&path_buffer,sizeof(unsigned), &length,sizeof(cl_int),&i);
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue,phase1, 2, NULL, globalSize, localSize, 0, NULL, NULL),"Error executing phase1 kernel");
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue,phase2, 2, NULL, globalSize, localSize, 0, NULL, NULL),"Error executing phase2 kernel");
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue,phase3, 2, NULL, globalSize, localSize, 0, NULL, NULL),"Error executing phase3 kernel");
    }

    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue,tile_to_row_kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");


    for(int i = 0; i< out_length;i++)
    {
        err = clEnqueueReadBuffer(command_queue,out_buffer,CL_FALSE,sizeof(cl_float) * length * i,sizeof(cl_float) * out_length,out_matrix[i],0,NULL,NULL);
        err = clEnqueueReadBuffer(command_queue,path_out,CL_FALSE,sizeof(cl_uint) * length * i,sizeof(cl_uint) * out_length,out_path[i],0,NULL,NULL);
    }

    // Wait until all queued commands finish
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    CLU_ERRCHECK(err,"Error finishing command_queue");

    if(time != NULL)
        *time = time_ms()-start_time;

    //Free Allocated Memory
    freeFloatMatrix(matrix,length);

    //Finalize
    err = clReleaseKernel(phase1);
    err |= clReleaseKernel(phase2);
    err |= clReleaseKernel(phase3);
    err |= clReleaseKernel(row_to_tile_kernel);
    err |= clReleaseKernel(tile_to_row_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(path_buffer);
    err |= clReleaseMemObject(matrix_buffer);
    err |= clReleaseMemObject(tile_buffer);
    err |= clReleaseMemObject(out_buffer);
    err |= clReleaseMemObject(path_out);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err,"Failed finalizing OpenCL");

}



