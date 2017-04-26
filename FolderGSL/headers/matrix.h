#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <CL/cl.h>
#include <stdbool.h>
#include <graph.h>


void fillPathMatrix(cl_uint** matrix, unsigned length);
void copyMatrixContent(cl_float** in, cl_float** out, unsigned length);
cl_float** getAdjMatrix(unsigned vertices,unsigned edges);
cl_float** getTestMatrix(unsigned vertices);
void printMatrix(cl_float** mat, unsigned length);
cl_float** GraphToMatrix(Graph* graph);
void calc_path_rec(cl_uint** path_matrix, cl_uint* path, cl_uint i, cl_uint j);
unsigned createPath(cl_uint** path_matrix, cl_uint* path, cl_uint s, cl_uint t);
bool verify_cost_path(cl_float** mat1, cl_float** mat2,cl_uint** path1, cl_uint** path2, unsigned length);
cl_float** copyMatrix(cl_float** matrix, unsigned length);
cl_float** createFloatMatrix(unsigned length);
cl_uint** createUnsignedMatrix(unsigned length);
bool AlmostEqual2sComplement(float A, float B);
bool float_matrix_equal(cl_float** mat1, cl_float** mat2, unsigned length);
void freeUnsignedMatrix(cl_uint** matrix, unsigned length);
void freeFloatMatrix(cl_float** matrix, unsigned length);
bool path_matrix_equal(cl_uint** mat1, cl_uint** mat2, unsigned length);
cl_float** resizeFloatMatrix(cl_float** mat, unsigned length, unsigned blocksize);


#endif // MATRIX_H_INCLUDED
