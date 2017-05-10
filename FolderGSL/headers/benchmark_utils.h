#ifndef BENCHMARK_UTILS_H_INCLUDED
#define BENCHMARK_UTILS_H_INCLUDED


// Function to check if to arrays of size length and type cl_uint are identical, i.e. arr1[i][j] equals arr2[i][j] for i,j > 0
bool cl_uint_arr_equal(cl_uint* arr1, cl_uint* arr2, unsigned length);

// Function to check if to arrays of size length and type cl_float are identical, i.e. arr1[i] equals arr2[i] for all i >= 0
bool cl_float_arr_equal(cl_float* arr1, cl_float* arr2, unsigned length);

// Function to generate a pathname for the csv files
void generate_path_name_csv(const char* filename, char* pathname);

// Creates Csv File if not existing and generates the columns, one for the graph and num_devices for the devices
void initCsv(const char* filename, unsigned num_devices);

// Function to write results of benchmarking in the csv File, V/E denotes the number of vertices/edges of the graph, device_id the device and time the actual time it took
void writeToCsv(const char* filename, unsigned V, unsigned E, unsigned device_id, unsigned long time);

// Function that returns the time in milliseconds, can be used for time measuring
unsigned long time_ms();


#endif // BENCHMARK_UTILS_H_INCLUDED
