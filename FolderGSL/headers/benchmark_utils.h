#ifndef BENCHMARK_UTILS_H_INCLUDED
#define BENCHMARK_UTILS_H_INCLUDED

void initCsv(const char* filename, unsigned num_devices);
void writeToCsv(const char* filename, unsigned V, unsigned E, unsigned device_id, unsigned long time);
bool cl_uint_arr_equal(cl_uint* arr1, cl_uint* arr2, unsigned length);
bool cl_float_arr_equal(cl_float* arr1, cl_float* arr2, unsigned length);
void generate_path_name(const char* filename, char* pathname);
unsigned long time_ms();


#endif // BENCHMARK_UTILS_H_INCLUDED
