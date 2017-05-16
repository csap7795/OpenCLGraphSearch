#include <dijkstra_parallel.h>
#include <stdio.h>
#include <CL/cl.h>
#include <stdbool.h>
#include <limits.h>
#include <float.h>
#include <dijkstra_serial.h>
#include <cl_utils.h>
#include <benchmark_utils.h>
#include <unistd.h>
#include <libgen.h>

static cl_program program;
static cl_context context;
static cl_command_queue command_queue;
static cl_device_id device;

static void build_program(size_t device_num)
{
    device = cluInitDevice(device_num,&context,&command_queue);
    //printf("%s\n\n",cluGetDeviceDescription(device,device_num));

    char* filename = "/dijkstra_kernel.cl";
    char cfp[1024];

    /* Create path to the kernel file*/
    char kernel_file[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(kernel_file,"%s%s%s",dirname(dirname(cfp)),"/kernels",filename);
    program = cluBuildProgramFromFile(context,device,kernel_file,NULL);
}

/*Is able to detect negative cycles in the graph under the condition that they are connected to the source node */
void dijkstra_parallel(Graph* graph, unsigned source, unsigned device_num, cl_float* out_cost, cl_uint* out_path, unsigned long *time) // bool* check_cycles, bool* negative_cycles,
{
    /*First, build the program*/
    build_program(device_num);

    unsigned long start_time = time_ms();

    cl_int err;

    // Create Memory Buffers for Graph Data
    cl_mem vertice_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * (graph->V+1), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating verticebuffer");

    cl_mem edge_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_uint) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating edgebuffer");

    cl_mem weight_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * graph->E, NULL, &err);
    CLU_ERRCHECK(err,"Failed creating weight_buffer");

    // Create Memory Buffers for Helper Objects
    cl_mem mask_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_bool) * graph->V, NULL,&err);
    CLU_ERRCHECK(err,"Failed creating mask_buffer");

    cl_mem cost_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating cost_buffer");

    cl_mem predecessor_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating predecessor_buffer");

    cl_mem update_cost_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * (graph->V), NULL, &err);
    CLU_ERRCHECK(err,"Failed creating update_cost_buffer");

    cl_mem semaphore_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_int) * graph->V,NULL,&err);
    CLU_ERRCHECK(err,"Failed creating semaphore_buffer");

    cl_mem finished_flag = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_bool), NULL,&err);
    CLU_ERRCHECK(err,"Failed creating finished_flag");

    // Copy Graph Data to their respective memory buffers
    err = clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, 0, graph->V * sizeof(cl_uint), graph->vertices , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, vertice_buffer, CL_FALSE, graph->V * sizeof(cl_uint), sizeof(cl_uint), &graph->E , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, edge_buffer, CL_FALSE, 0, graph->E * sizeof(cl_uint), graph->edges , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, weight_buffer, CL_TRUE,0, graph->E * sizeof(cl_uint), graph->weight,0, NULL, NULL);
    CLU_ERRCHECK(err,"Failed copying graph data to buffers");

    // Create Kernels
    cl_kernel init_kernel = clCreateKernel(program,"initializeBuffers",&err);
    CLU_ERRCHECK(err,"Failed to create initializing_buffers kernel from program");

    cl_kernel dijkstra1_kernel = clCreateKernel(program,"dijkstra1",&err);
    CLU_ERRCHECK(err,"Failed to create dijkstra1 kernel from program");

    cl_kernel dijkstra2_kernel = clCreateKernel(program,"dijkstra2",&err);
    CLU_ERRCHECK(err,"Failed to create dijkstra2 kernel from program");


    //Set KernelArguments
    cluSetKernelArguments(init_kernel,6,sizeof(cl_mem),&mask_buffer,sizeof(cl_mem),&cost_buffer,sizeof(cl_mem),&update_cost_buffer,sizeof(cl_mem),(void*)&predecessor_buffer,sizeof(cl_mem),&semaphore_buffer,sizeof(cl_uint),&source);
    cluSetKernelArguments(dijkstra1_kernel,8,sizeof(cl_mem),(void*)&vertice_buffer,sizeof(cl_mem),(void*)&edge_buffer,sizeof(cl_mem),(void*)&weight_buffer,sizeof(cl_mem),(void*)&mask_buffer,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),(void*)&update_cost_buffer,sizeof(cl_mem),(void*)&predecessor_buffer,sizeof(cl_mem),(void*)&semaphore_buffer);
    cluSetKernelArguments(dijkstra2_kernel,4,sizeof(cl_mem),(void*)&mask_buffer,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),(void*)&update_cost_buffer,sizeof(cl_mem),(void*)&finished_flag);

    // Execute the OpenCL kernel for initializing Buffers
    size_t globalSize = graph->V;
    CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, init_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

    // start looping both main kernels
    bool finished;
    int i;
    for(i = 0; i<graph->V;i++)
    {
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, dijkstra1_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "Failed to enqueue Dijkstra1 kernel");
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, dijkstra2_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "Failed to enqueue Dijkstra2 kernel");
        err = clEnqueueReadBuffer(command_queue,finished_flag,CL_TRUE,0,sizeof(cl_bool),&finished,0,NULL,NULL);

        if(finished)
            break;

        finished = true;
        err = clEnqueueWriteBuffer(command_queue, finished_flag, CL_TRUE, 0, sizeof(cl_bool), &finished , 0, NULL, NULL);

    }

    // if the calculation stops before the i'th iteration, you can be sure there's no negative cycle
    /*if(i!=graph->V && check_cycles != NULL)
    {
        *check_cycles = false;
    }

    // if asked, save the results in the respective out variables
    if(out_cost != NULL && out_path != NULL)
    {
        err = clEnqueueReadBuffer(command_queue,cost_buffer,CL_FALSE,0,sizeof(cl_float) * graph->V,out_cost,0,NULL,NULL);
        err = clEnqueueReadBuffer(command_queue,predecessor_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->V,out_path,0,NULL,NULL);
    }

    // if check_cycles is still true, check if there exists a negative cycle
    if(check_cycles != NULL && *check_cycles)
    {

        // create the kernel
        cl_kernel negativeCycle_kernel = clCreateKernel(program,"negativeCycle",&err);
        CLU_ERRCHECK(err,"Failed to create negative cycle kernel from program");

        // Create the buffers
        cl_mem negative_cycle_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(cl_short) * (graph->V), NULL, &err);
        CLU_ERRCHECK(err,"Failed creating verticebuffer");

        cl_mem detected_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
        CLU_ERRCHECK(err,"Failed creating verticebuffer");

        // Set the kernel Arguments
        cluSetKernelArguments(negativeCycle_kernel,6,sizeof(cl_mem),&vertice_buffer,sizeof(cl_mem),&edge_buffer,sizeof(cl_mem),&weight_buffer,sizeof(cl_mem),(void*)&cost_buffer,sizeof(cl_mem),&negative_cycle_buffer,sizeof(cl_mem),&detected_buffer);

        cl_int detected = 0;
        err = clEnqueueWriteBuffer(command_queue, detected_buffer, CL_FALSE, 0, sizeof(cl_int), &detected , 0, NULL, NULL);
        CLU_ERRCHECK(err,"Failed copying detected_flag to detected_buffers");

        //Execute the kernel
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, negativeCycle_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL),"Error executing InitializeKernel");

        //read back results
        err = clEnqueueReadBuffer(command_queue,detected_buffer,CL_TRUE,0,sizeof(cl_int),&detected,0,NULL,NULL);

        // If the kernel detected negative cycles, read back where the negative cycles are located, else write false into check_cycles
        if(detected == 1 && negative_cycles != NULL)
        {
            cl_short *tmp = (cl_short*)malloc(sizeof(cl_short)*graph->V);
            err = clEnqueueReadBuffer(command_queue,negative_cycle_buffer,CL_TRUE,0,sizeof(cl_short)*graph->V,tmp,0,NULL,NULL);
            for(int i = 0; i<graph->V;i++)
                negative_cycles[i] = tmp[i];
            free(tmp);
        }
        else
        {
            *check_cycles = false;
        }
        err = clReleaseMemObject(negative_cycle_buffer);
        err |= clReleaseMemObject(detected_buffer);
        err |= clReleaseKernel(negativeCycle_kernel);

        CLU_ERRCHECK(err, "Failed during finalizing negativeCycle buffers & kernel");

    }*/

    // if asked, save the results in the respective out variables
    if(out_cost != NULL && out_path != NULL)
    {
        err = clEnqueueReadBuffer(command_queue,cost_buffer,CL_FALSE,0,sizeof(cl_float) * graph->V,out_cost,0,NULL,NULL);
        err |= clEnqueueReadBuffer(command_queue,predecessor_buffer,CL_FALSE,0,sizeof(cl_uint) * graph->V,out_path,0,NULL,NULL);
        CLU_ERRCHECK(err,"Error reading back results");
    }

    // Finish all commands in Commandqueue
    err = clFlush(command_queue);
    err |= clFinish(command_queue);
    CLU_ERRCHECK(err,"Error finishing command_queue");

    //save time if asked
    if(time != NULL)
        *time = time_ms()-start_time;

    // Clean up
    err = clReleaseKernel(init_kernel);
    err |= clReleaseKernel(dijkstra1_kernel);
    err |= clReleaseKernel(dijkstra2_kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(vertice_buffer);
    err |= clReleaseMemObject(edge_buffer);
    err |= clReleaseMemObject(weight_buffer);
    err |= clReleaseMemObject(cost_buffer);
    err |= clReleaseMemObject(update_cost_buffer);
    err |= clReleaseMemObject(predecessor_buffer);
    err |= clReleaseMemObject(mask_buffer);
    err |= clReleaseMemObject(semaphore_buffer);
    err |= clReleaseMemObject(finished_flag);
    err |= clReleaseCommandQueue(command_queue);
    err |= clReleaseContext(context);

    CLU_ERRCHECK(err, "Failed during finalizing OpenCL");

}
