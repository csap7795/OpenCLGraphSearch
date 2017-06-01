#include <floyd_warshall.h>
#include <cl_utils.h>
#include <CL/cl.h>
#include <matrix.h>
#include <Test_floyd_warshall.h>
#include <benchmark_utils.h>


#define REPEATS 10

void benchmark_floyd_warshall(cl_float **mat, unsigned length, unsigned epv)
{
    // Create path to the kernel file
    char csv_file_fw_gpu[1024];
    char csv_file_fw_global[1024];
    char csv_file_fw_workgroup[1024];

    generate_path_name_csv(CSVFILENAME_FLOYD_WARSHALL_GLOBAL,csv_file_fw_global);
    generate_path_name_csv(CSVFILENAME_FLOYD_WARSHALL_GPU,csv_file_fw_gpu);
    generate_path_name_csv(CSVFILENAME_FLOYD_WARSHALL_WORKGROUP,csv_file_fw_workgroup);

    unsigned num_devices = cluCountDevices();

    //Create CSV File for documenting results
    initCsv(csv_file_fw_gpu,num_devices);
    initCsv(csv_file_fw_global,num_devices);
    initCsv(csv_file_fw_workgroup,num_devices);

    for(unsigned device = 0; device < num_devices;device++)
    {
        printf("Processing floyd warshall for device : %u\n",device);
        long unsigned time_global = 0;
        long unsigned time_workgroup = 0;
        long unsigned time_gpu = 0;

        for(int i = 0; i<REPEATS;i++)
        {
           time_global += measure_time_floyd_warshall_row(mat,length,device);
           time_gpu += measure_time_floyd_warshall_column(mat,length,device);
           time_workgroup += measure_time_floyd_warshall_workgroup(mat,length,device);
        }

        time_global = time_global/REPEATS;
        time_gpu = time_gpu/REPEATS;
        time_workgroup = time_workgroup/REPEATS;

        writeToCsv(csv_file_fw_gpu,length,epv,device,time_gpu);
        writeToCsv(csv_file_fw_global,length,epv,device,time_global);
        writeToCsv(csv_file_fw_workgroup,length,epv,device,time_workgroup);
    }

}

unsigned long measure_time_floyd_warshall_column(cl_float** mat, unsigned length,unsigned device_id)
{
    //create result variables
    cl_float** out_cost_parallel = createFloatMatrix(length);
    cl_uint** out_path_parallel = createUnsignedMatrix(length);
    unsigned long time;

    parallel_floyd_warshall_column(mat,out_cost_parallel,out_path_parallel,length,device_id,&time);

    //Free resources
    freeFloatMatrix(out_cost_parallel,length);
    freeUnsignedMatrix(out_path_parallel,length);

    return time;
}

unsigned long measure_time_floyd_warshall_row(cl_float** mat, unsigned length, unsigned device_id)
{
    //create result variables
    cl_float** out_cost_parallel = createFloatMatrix(length);
    cl_uint** out_path_parallel = createUnsignedMatrix(length);
    unsigned long time;

    parallel_floyd_warshall_row(mat,out_cost_parallel,out_path_parallel,length,device_id,&time);

    //Free resources
    freeFloatMatrix(out_cost_parallel,length);
    freeUnsignedMatrix(out_path_parallel,length);

    return time;
}

unsigned long measure_time_floyd_warshall_workgroup(cl_float** mat, unsigned length,unsigned device_id)
{
    //create result variables
    cl_float** out_cost_parallel = createFloatMatrix(length);
    cl_uint** out_path_parallel = createUnsignedMatrix(length);
    unsigned long time;

    parallel_floyd_warshall_workgroup(mat,out_cost_parallel,out_path_parallel,length,device_id,&time);

    //Free resources
    freeFloatMatrix(out_cost_parallel,length);
    freeUnsignedMatrix(out_path_parallel,length);

    return time;
}

void verify_floyd_warshall_column(cl_float** mat, unsigned length)
{
    //create result variables
    cl_float** out_cost_parallel = createFloatMatrix(length);
    cl_uint** out_path_parallel = createUnsignedMatrix(length);

    printf("%s\n","test_floyd_warshall_gpu");

    //Iterate over available devices and calculate APSP, verify the result
    for(int i = 0; i<cluCountDevices();i++)
    {
        //cl_device_id tmp = cluInitDevice(i,NULL,NULL);
         printf("%s\n",cluDeviceTypeStringFromNum(i));
        parallel_floyd_warshall_column(mat,out_cost_parallel,out_path_parallel,length,i,NULL);
        printf("Parallel and serial execution produce same results? ");
        printf("%s\n",verify_floyd_warshall(mat,out_cost_parallel,out_path_parallel,length) ? "TRUE" : "FALSE");
    }

    //Free resources
    freeFloatMatrix(out_cost_parallel,length);
    freeUnsignedMatrix(out_path_parallel,length);
}

void verify_floyd_warshall_row(cl_float** mat, unsigned length)
{
    //create result variables
    cl_float** out_cost_parallel = createFloatMatrix(length);
    cl_uint** out_path_parallel = createUnsignedMatrix(length);

    printf("%s\n","test_floyd_warshall_global");

    //Iterate over available devices and calculate APSP, verify the result
    for(int i = 0; i<cluCountDevices();i++)
    {

        cl_device_id tmp = cluInitDevice(i,NULL,NULL);
        printf("%s\n",cluGetDeviceDescription(tmp,i));
        parallel_floyd_warshall_row(mat,out_cost_parallel,out_path_parallel,length,i,NULL);
        printf("Parallel and serial execution produce same results? ");
        printf("%s\n",verify_floyd_warshall(mat,out_cost_parallel,out_path_parallel,length) ? "TRUE" : "FALSE");
    }

    //Free resources
    freeFloatMatrix(out_cost_parallel,length);
    freeUnsignedMatrix(out_path_parallel,length);
}

void verify_floyd_warshall_workgroup(cl_float** mat, unsigned length)
{
    //create result variables
    cl_float** out_cost_parallel = createFloatMatrix(length);
    cl_uint** out_path_parallel = createUnsignedMatrix(length);

    printf("%s\n","test_floyd_warshall_workgroup");

    //Iterate over available devices and calculate APSP, verify the result
    for(int i = 0; i<cluCountDevices();i++)
    {
        cl_device_id tmp = cluInitDevice(i,NULL,NULL);
        printf("%s\n",cluGetDeviceDescription(tmp,i));
        parallel_floyd_warshall_workgroup(mat,out_cost_parallel,out_path_parallel,length,i,NULL);
        printf("Parallel and serial execution produce same results? ");
        printf("%s\n",verify_floyd_warshall(mat,out_cost_parallel,out_path_parallel,length) ? "TRUE" : "FALSE");
    }

    //Free resources
    freeFloatMatrix(out_cost_parallel,length);
    freeUnsignedMatrix(out_path_parallel,length);
}

//Seems to be a race condition as the gpu calculates wrong results from time to time
bool verify_floyd_warshall(cl_float** matrix,cl_float** out_cost_parallel, cl_uint** out_path_parallel, unsigned length)
{
        //create result variables
        cl_float** out_cost_serial = createFloatMatrix(length);
        cl_uint** out_path_serial = createUnsignedMatrix(length);
        //calculate the algorithm
        serial_floyd_warshall(matrix,out_cost_serial,out_path_serial,length);

        bool result = float_matrix_equal(out_cost_parallel,out_cost_serial,length);
        result &= path_matrix_equal(out_path_parallel, out_path_serial, length);

        free(out_cost_serial);
        free(out_path_serial);

        return result;

}

/*Serial Floyd Warshall, changes the value of the input matrix to the apsp output*/
void serial_floyd_warshall(cl_float** matrix,cl_float** out_cost, cl_uint** out_path, unsigned length)
{
    copyMatrixContent(matrix,out_cost,length);
    fillPathMatrix(out_path,length);

    unsigned long start_time = time_ms();
    for(cl_uint k = 0; k<length;k++)
        for(int i = 0; i<length;i++)
            for(int j = 0; j<length;j++)
            {
                if(k != i && k != j &&out_cost[i][k] != CL_FLT_MAX && out_cost[k][j] != CL_FLT_MAX)
                {
                    cl_float tmp = out_cost[i][k] + out_cost[k][j];

                    if(out_cost[i][j]>tmp)
                    {
                        out_cost[i][j] = tmp;
                        out_path[i][j] = k;
                    }
                }
            }

    printf("Time for serial approach : %lu ms\n",time_ms()-start_time);
}

