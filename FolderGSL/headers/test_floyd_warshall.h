#ifndef TEST_FLOYD_WARSHALL_H_INCLUDED
#define TEST_FLOYD_WARSHALL_H_INCLUDED

#include <CL/cl.h>
#include <stdbool.h>

#define CSVFILENAME_FLOYD_WARSHALL_GLOBAL "fwglobal.csv"
#define CSVFILENAME_FLOYD_WARSHALL_GPU "fwgpu.csv"
#define CSVFILENAME_FLOYD_WARSHALL_WORKGROUP "fwworkgroup.csv"

/* calculates the average time for each algorithm REPEATS times and documents it into a csvfile*/
void benchmark_floyd_warshall(cl_float **mat, unsigned length, unsigned epv);

/*Calculates the time for one execution of each algorithm*/
unsigned long measure_time_floyd_warshall_gpu(cl_float** mat, unsigned length,unsigned device_id);
unsigned long measure_time_floyd_warshall_global(cl_float** mat, unsigned length, unsigned device_id);
unsigned long measure_time_floyd_warshall_workgroup(cl_float** mat, unsigned length,unsigned device_id);

/*Checks the algorithms on right results*/
void verify_floyd_warshall_global_gpu(cl_float** mat, unsigned length);
void verify_floyd_warshall_global(cl_float** mat, unsigned length);
void verify_floyd_warshall_workgroup(cl_float** mat, unsigned length);

/*Is used by the above functions to determine if the algorihtms produce right results*/
void serial_floyd_warshall(cl_float** matrix,cl_float** out_cost, cl_uint** _out_path, unsigned length);
bool verify_floyd_warshall(cl_float** matrix,cl_float** out_cost_parallel, cl_uint** out_path_parallel, unsigned length);


#endif // TEST_FLOYD_WARSHALL_H_INCLUDED
