#include<stdlib.h>
#include<stdio.h>
#include <sys/stat.h>
#include <graph.h>
#include <stdbool.h>
#include <unistd.h>
#include <libgen.h>

void generate_path_name(const char* filename, char* pathname)
{
    char cfp[1024];
    sprintf(cfp, "%s",__FILE__);
    sprintf(pathname,"%s%s%s",dirname(dirname(cfp)),"/Diagramms/csv_files/",filename);
}

void initCsv(const char* filename, unsigned num_devices)
{
    if(num_devices == 0)
        return;

    struct stat fileStat;


    if(stat(filename,&fileStat) != -1 && fileStat.st_size != 0)
        return;

    FILE *fp = fopen(filename,"w");
    if(fp == NULL)
    {
        printf("No such file or directory");
    }


    fputs("category, ",fp);
    int i;
    for(i = 0; i<num_devices;i++)
        fprintf(fp,"device%u, ",i);

    fclose(fp);
}

void writeToCsv(const char* filename, unsigned V, unsigned E, unsigned device_id, unsigned long time)
{
    FILE *fp = fopen(filename,"a");
    if(device_id == 0)
        fprintf(fp,"\nV%uk/E%uk, ",V/1000, E/1000);

    fprintf(fp,"%lu, ",time);
    fclose(fp);
}


// Check if two cl_uint arrays contain the same values
bool cl_uint_arr_equal(cl_uint* arr1, cl_uint* arr2, unsigned length)
{
    for(int i = 0; i<length;i++)
        if(arr1[i] != arr2[i])
            return false;

    return true;
}

// Check if two cl_float arrays contain the same values
bool cl_float_arr_equal(cl_float* arr1, cl_float* arr2, unsigned length)
{
    for(int i = 0; i<length;i++)
        if(arr1[i] != arr2[i])
            return false;

    return true;
}
