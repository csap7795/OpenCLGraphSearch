#include <scan.h>
#include <cl_utils.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time_ms.h>

#define GROUP_SIZE 512

static cl_program program;
static cl_context context;
static cl_command_queue command_queue;
static cl_device_id device;


static void build_kernel(size_t device_num)
{
    device = cluInitDevice(device_num,&context,&command_queue);
    //printf("%s\n\n",cluGetDeviceDescription(device,device_num));
    char tmp[1024];
    sprintf(tmp, "-DGROUP_SIZE=%d",GROUP_SIZE);
    char* filename = "/scan.cl";
    char cfp[1024];
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s%s",dirname(dirname(cfp)),"/kernels",filename);
    program = cluBuildProgramFromFile(context,device,kernel_file,tmp);
}

void scan_parallel(const cl_uint *input, cl_uint* output, uint length, uint device)
{
    build_kernel(device);

    long unsigned start_time = time_ms();
    sum_scan(input,output,length,device);
    printf("Time parallel: %lu\n",time_ms()-start_time);
    //check result:
    cl_uint* serial_out = (cl_uint*)malloc(sizeof(cl_uint)*length);

    start_time = time_ms();
    scan_serial(input,serial_out,length);
    printf("Time serial: %lu\n",time_ms()-start_time);

    for(int i = 0; i<length;i++)
        if(serial_out[i] != output[i]){
            printf("FALSE!");
            break;}


    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(serial_out);
}

void sum_scan(const cl_uint *input, cl_uint* output, uint length, uint device)
{
    size_t num_elems = round_up_globalSize(length,GROUP_SIZE);

    cl_int err;
    //Create Buffers
    cl_mem input_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * num_elems, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating input_buffer");

    cl_mem output_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * num_elems, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    cl_mem offset_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * (num_elems/GROUP_SIZE), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    cl_mem actual_output_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * num_elems, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    //Enqueue Input Data
    cl_uint* zeroes = (cl_uint*)calloc(GROUP_SIZE,sizeof(cl_uint));
    err = clEnqueueWriteBuffer(command_queue, input_buffer, CL_FALSE, 0, length * sizeof(cl_uint), input , 0, NULL, NULL);
    if(length < num_elems)
        err = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE,  length * sizeof(cl_uint),(num_elems-length)*sizeof(cl_uint), zeroes , 0, NULL, NULL);
    CLU_ERRCHECK(err,"error writing to input_buffer");
    free(zeroes);

    cl_kernel sum_scan_kernel = clCreateKernel(program,"sum_scan",&err);
    CLU_ERRCHECK(err,"Failed to create sum_scan kernel from program");
    cl_kernel add_offset_kernel = clCreateKernel(program,"add_offsets",&err);
    CLU_ERRCHECK(err,"Failed to create add_offset kernel from program");

    cluSetKernelArguments(sum_scan_kernel,3,sizeof(cl_mem),(void*)&output_buffer,sizeof(cl_mem),(void*)&input_buffer,sizeof(cl_mem),(void*)&offset_buffer);

    size_t global_size = num_elems/2;
    size_t local_size = GROUP_SIZE;

    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, sum_scan_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL),"Error executing InitializeKernel");

    err = clFinish(command_queue);
    unsigned new_length = num_elems/(GROUP_SIZE*2);
    cl_uint* offset = (cl_uint*)malloc(sizeof(cl_uint) * new_length);
    cl_uint* sum_offset = (cl_uint*)malloc(sizeof(cl_uint) * new_length);

    err = clEnqueueReadBuffer(command_queue,offset_buffer,CL_TRUE,0,sizeof(cl_uint) * new_length,offset,0,NULL,NULL);
    err = clEnqueueReadBuffer(command_queue,output_buffer,CL_TRUE,0,sizeof(cl_uint) * length,output,0,NULL,NULL);

    err = clFlush(command_queue);
    err = clFinish(command_queue);
    //err = clReleaseMemObject(output_buffer);

    if(new_length > GROUP_SIZE)
        sum_scan(offset,sum_offset,new_length,device);
    else
        scan_serial(offset,sum_offset,new_length);

    /*output_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * global_size, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    offset_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * new_length, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");*/

    err = clEnqueueWriteBuffer(command_queue, offset_buffer, CL_TRUE, 0, new_length * sizeof(cl_uint), sum_offset , 0, NULL, NULL);
    cluSetKernelArguments(add_offset_kernel,3,sizeof(cl_mem),(void*)&actual_output_buffer,sizeof(cl_mem),(void*)&output_buffer,sizeof(cl_mem),(void*)&offset_buffer);


    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, add_offset_kernel, 1, NULL, &num_elems, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    err = clEnqueueReadBuffer(command_queue,actual_output_buffer,CL_TRUE,0,sizeof(cl_uint) * length,output,0,NULL,NULL);

    free(offset);
    free(sum_offset);

    //Finalize
    err = clFlush(command_queue);
    err = clFinish(command_queue);


    clReleaseKernel(sum_scan_kernel);
    clReleaseKernel(add_offset_kernel);

    err = clReleaseMemObject(input_buffer);
    CLU_ERRCHECK(err, "failed releasing input_buffer");
    err = clReleaseMemObject(output_buffer);
    CLU_ERRCHECK(err, "failed releasing output_buffer");
    err = clReleaseMemObject(actual_output_buffer);
    CLU_ERRCHECK(err, "failed releasing actual_output_buffer");
    err = clReleaseMemObject(offset_buffer);
    CLU_ERRCHECK(err, "failed releasing offset_buffer");

}

unsigned long scan(const cl_uint *input, cl_uint* output, uint length, uint device)
{
    unsigned long start_time = time_ms();

    size_t global_size = round_up_globalSize(length,GROUP_SIZE) ;

    cl_int err;
    //Create Buffers
    cl_mem input_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * global_size, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating input_buffer");

    cl_mem output_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_uint) * global_size, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating sourceVertice_buffer");

    //Enqueue Input Data
    cl_uint* zeroes = (cl_uint*)calloc(global_size-length,sizeof(cl_uint));
    err = clEnqueueWriteBuffer(command_queue, input_buffer, CL_FALSE, 0, length * sizeof(cl_uint), input , 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE,  length * sizeof(cl_uint),(global_size-length)*sizeof(cl_uint), zeroes , 0, NULL, NULL);
    free(zeroes);

    cl_kernel scan_kernel = clCreateKernel(program,"scan",&err);
    CLU_ERRCHECK(err,"Failed to create scan kernel from program");

   // global_size /= 2;
    size_t local_size = GROUP_SIZE/2;

    cluSetKernelArguments(scan_kernel,3,sizeof(cl_mem),(void*)&output_buffer,sizeof(cl_mem),(void*)&input_buffer,sizeof(cl_uint),(void*)&global_size);

    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, scan_kernel, 1, NULL, &global_size,&local_size, 0, NULL, NULL),"Error executing InitializeKernel");

    err = clEnqueueReadBuffer(command_queue,output_buffer,CL_TRUE,0,sizeof(cl_uint) * length,output,0,NULL,NULL);

    //cl_uint* offset = (cl_uint*)malloc (sizeof(cl_uint) * (global_size/GROUP_SIZE));
    //err = clEnqueueReadBuffer(command_queue,offset_buffer,CL_TRUE,0,sizeof(cl_uint) * (global_size/GROUP_SIZE),offset,0,NULL,NULL);

    //free(offset);
    //Finalize
    err = clFlush(command_queue);
    err = clFinish(command_queue);
    err = clReleaseKernel(scan_kernel);
    err = clReleaseMemObject(input_buffer);
    err = clReleaseMemObject(output_buffer);
    //err = clReleaseMemObject(offset_buffer);
    err = clReleaseProgram(program);
    err = clReleaseCommandQueue(command_queue);
    err = clReleaseContext(context);

    return time_ms()-start_time;

}

unsigned long scan_serial(const uint *input, uint *output, int length)
{
    unsigned long start_time = time_ms();
    output[0] = 0;
    for(int i = 0; i<length-1;i++)
    {
        output[i+1] = input[i] + output[i];
    }
    return time_ms()-start_time;
}
