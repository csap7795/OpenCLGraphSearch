#include<benchmark_utils.h>
#include<stdlib.h>
#include<stdio.h>
#include <sys/stat.h>
#include <graph.h>
#include <stdbool.h>
#include <unistd.h>
#include <libgen.h>
#include <sys/time.h>
#include <assert.h>

void generate_path_name_csv(const char* filename, char* pathname)
{
    char cfp[1024];
    sprintf(cfp, "%s",__FILE__);
    // As this function is folder sources, you need to call dirname a second time to get to the root folder
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

    char vertice_char=' ';
    char edge_char=' ';
    float vertices = (float)V;
    if(vertices >= 1000)
    {
        vertices /= 1000;
        vertice_char = 'k';
        if(vertices > 1000)
        {
            vertices/=1000;
            vertice_char = 'm';
            if(vertices > 1000)
            {
                vertices/=1000;
                vertice_char = 'b';
            }
        }

    }

    float edges = (float)E;
    if(edges >= 1000)
    {
        edges /= 1000;
        edge_char = 'k';
        if(edges > 1000)
        {
            edges/=1000;
            edge_char = 'm';
            if(edges > 1000)
            {
                edges/=1000;
                edge_char = 'b';
            }
        }


    }
    if(device_id == 0)
        fprintf(fp,"\nV%.1f%c/E%.1f%c, ",vertices,vertice_char,edges,edge_char);

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
        if(!AlmostEqual2sComplement(arr1[i],arr2[i]))
            return false;

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

