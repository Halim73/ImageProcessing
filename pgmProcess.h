
/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
__device__ float distance( int p1[], int p2[] );
__device__ int* sort(int* window, int size);
__device__ int orientation(int p1[], int p2[], int p3[]);
__device__ int onSegment(int p1[], int p2[], int p3[]);
__device__ float gaussian(float x, float y, float sigma);
__device__ int checkStrongNeighbor(float vector, float p1, float threshold);
__device__ float laplacianGauss(float x, float y, float sigma);
__device__ void createGaussMask(int size, float sigma, float* kernel);
__device__ void approxGauss5x5(int* pixels, int* outPixels, int row, int column, int numRows, int numCols);
__device__ void sobelFilter(int* pixels, int row, int column, int numRows, int numCols, float* values);
__device__ int doesIntersect(int pointA1[], int pointB1[], int pointA2[], int pointB2[]);
__device__ void localMaxMin(int* values, int* max, int* min, int* regionalMax, int* regionalMin, int numRegions, int numRows, int numCols, int xOrY, int size);
__global__ void setBlockPixels(int* pixels, int* outPixels, int numRows, int numCols, int sharedSize);
__global__ void maxMin(int* values, int* max, int* min, int numRegions, int numRows, int numCols,int xOrY);
__global__ void medianFilter(int* pixels, int* outPixels, int numRows, int numCols, int numFilter);
__global__ void drawLine(int* pixels, int numCol, int numRows, int p1[], int p2[]);
__global__ void drawCircle(int* pixels, int numCol, int numRows, int point[], int radius);
__global__ void drawEdge(int* pixels, int numRows, int numCols, int edgeWidth);
__global__ void edgeDetect(int* pixels, int* outPixels, int numRows, int numCols,int threshold);
__global__ void gaussBlur(int* pixels, int* outPixels, int numRows, int numCols, float radius);
__global__ void zsThinning(int* pixels, int* outPixels, int numRows, int numCols, int radius);
__global__ void guoHallThinning(int* pixels, int* outPixels, int numRows, int numCols, int iter);
__global__ void gaussBlurKernel(int* pixels, int* outPixels, int numRows, int numCols, float sigma);