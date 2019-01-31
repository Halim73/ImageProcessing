/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
#include "pgmProcess.h"

__device__ float distance( int p1[], int p2[] ){
	int xDiff = p2[0] - p1[0];
	int yDiff = p2[1] - p1[1];

	xDiff *= xDiff;
	yDiff *= yDiff;

	float distance = (float)sqrt((float)(xDiff + yDiff));

	return distance;
}

__device__ int* sort(int* window,int size) {
	int i, j;
	for (i = 0; i < size-1; i++) {
		for (j = 0; j < size; j++) {
			if (window[i] > window[j]) {
				int temp = window[i];
				window[i] = window[j];
				window[j] = temp;
			}
		}
	}
	return window;
}

//x=[0],y=[1]
__device__ int orientation(int p1[],int p2[], int p3[]) {
	int value = (p2[1] - p1[1])*(p3[0] - p2[0]) - (p2[0] - p1[0])*(p3[1] - p2[1]);
	if (value == 0)return 0;//colinear
	return (value > 0) ? 1 : 2;//1=cw 2=ccw
}

//1=true 0=false
__device__ int onSegment(int p1[], int p2[], int p3[]) {
	if (p2[0] <= max(p1[0], p3[0]) && p2[0] >= min(p1[0], p3[0]) &&
		p2[1] <= max(p1[1], p3[1]) && p2[1] >= min(p1[1], p3[1])) {
		return 1;
	}
	return 0;
}

__device__ void sobelFilter(int* pixels, int row, int column, int numRows, int numCols, float* values) {//0 = magnitude 1 = theta
	float xMat[9] = { -1,0,1,-2,0,2,-1,0,1 };//general
	float yMat[9] = { -1,-2,-1,0,0,0,1,2,1 };
	
	/*float xMat[9] = { 0,-1,0,
					-1,4,-1,
					0,-1,0 };
	float yMat[9] = { -1,-1,-1,
					  -1,8,-1,
					-1,-1,-1 };*/
	float pi = 3.141592653;

	float magX = 0, magY = 0;
	int i, j;
	int index, size = (numRows*numCols);
	
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			int y = row + j - 1;
			int x = column + i - 1;

			index = y * numCols + x;

			if (index < size) {
				magX += (float)((float)pixels[index] * xMat[i * 3 + j] * (.125));
				magY += (float)((float)pixels[index] * yMat[i * 3 + j] * (.125));
			}
		}
	}
	values[0] = sqrt((float)(magX*magX + magY * magY));//magnitude
	values[1] = atan((float)(magY / magX))*(180 / pi);//radians
	//if (values[1] < 0)values[1] += 360;
}

__device__ void approxGauss5x5(int* pixels,int* outPixels, int row, int column, int numRows, int numCols) {
	int mask[] = { 1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1 };
	float scalar = 1 / 256;

	int i, j;
	int index, size = (numRows*numCols);

	float weight, sum = 0;
	for (i = 0; i < 5; i++) {
		for (j = 0; j < 5; j++) {
			int y = row + j - 1;
			int x = column + i - 1;

			index = y * numCols + x;

			if (index < size) {
				weight += (float)(pixels[index] * mask[i * 5 + j] * scalar);
				sum += weight;
			}
		}
	}
	outPixels[index] = (int)round(weight / sum);
}

__device__ float laplacianGauss(float x,float y, float sigma) {
	float c = 2.0 * sigma * sigma, pi = 3.141592653;
	float e = exp((-(x * x + y * y) / c));
	float det = (1 - (x*x + y * y) / c);

	return (-1 / (pi*sigma*sigma*sigma*sigma))*(det)*e;
}

__device__ float gaussian(float x,float y, float sigma) {
	float c = 2.0 * sigma * sigma, pi = 3.141592653;
	float e = exp(-(x * x) / c) / sqrt(c * pi);

	return e;
}

__device__ void createGaussMask(int size, float sigma, float* kernel) {
	int i,j;
	float sum = 0;
	int halfSize = size / 2;
	for (i = -halfSize; i < halfSize; i++) {
		for (j = -halfSize; j < halfSize; j++) {
			float r = sqrt((float)(i * i + j * j));
			float g = abs(gaussian(r, j, sigma));
			//float g = abs(laplacianGauss(i, j, sigma));
			kernel[(i + halfSize) * size + (j + halfSize)] = g;
			sum += g;
		}
	}
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			kernel[i*size + j] /= sum;
		}
	}
}

__device__ int checkStrongNeighbor(float vector, float p1, float threshold) {
	if (p1 >= threshold && vector < p1)return 1;//not strong
	return 0;//strong
}

//1=true 0=false
__device__ int doesIntersect(int pointA1[],int pointB1[],int pointA2[],int pointB2[]) {
	int ori1 = orientation(pointA1, pointB1, pointA2);
	int ori2 = orientation(pointA1, pointB1, pointB2);
	int ori3 = orientation(pointA2, pointB2, pointA1);
	int ori4 = orientation(pointA2, pointB2, pointB1);

	if (ori1 != ori2 && ori3 != ori4) {
		return 1;
	}

	if (ori1 == 0 && onSegment(pointA1, pointA2, pointB1))return 1;//they are colinear and a2 lies on a1b1
	if (ori2 == 0 && onSegment(pointA1, pointB2, pointB1))return 1;//they are colinear and b2 lies on a1b1
	if (ori3 == 0 && onSegment(pointA2, pointA1, pointB2))return 1;//they are colinear and a1 lies on a2b2
	if (ori4 == 0 && onSegment(pointA2, pointB1, pointB2))return 1;//they are colinear and b1 lies on a2b2

	return 0;
}

__device__ void localMaxMin(int* values, int* max, int* min, int* regionalMax,int* regionalMin, int numRegions, int numRows, int numCols, int xOrY,int size) {
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int blockID = (blockDim.x*threadIdx.y) + threadIdx.x;
	int index = row * numCols + column;

	if (blockID < size) {
		if (values[blockID] > 0) {
			if (xOrY == 0) {
				int value = row;
				int region = index % numRegions;

				if (atomicMax(&regionalMax[region], value) < value) {
					*max = value;
					//atomicMax(max, value);
				}

				if (atomicMin(&regionalMin[region], value) > value) {
					*min = value;
					//atomicMin(min, value);
				}
			}
			else {
				int value = column;
				int region = index % numRegions;

				if (atomicMax(&regionalMax[region], value) < value) {
					*max = value;
					//atomicMax(max, value);
				}

				if (atomicMin(&regionalMin[region], value) > value) {
					*min = value;
					//atomicMin(min, value);
				}
			}
		}
	}
}

__global__ void setBlockPixels(int* pixels, int* outPixels, int numRows, int numCols,int sharedSize) {//this is going to be the graham scan kernel?
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int index = row * numCols + column;

	int size = numRows * numCols;

	int maxX, minX, maxY, minY;

	extern __shared__ int blockPixels[];
	extern __shared__ int regionalMax[];
	extern __shared__ int regionalMin[];

	int blockID = (blockDim.x*threadIdx.y) + threadIdx.x;
	
	int ll[2], lr[2], tl[2], tr[2],p[2];

	if (index < size) {
		if (blockID < sharedSize) {
			blockPixels[blockID] = pixels[index];
		}
	}
	__syncthreads();

	localMaxMin(blockPixels, &maxX, &minX, regionalMax, regionalMin, sharedSize, numRows, numCols, 1, size);
	localMaxMin(blockPixels, &maxY, &minY, regionalMax, regionalMin, sharedSize, numRows, numCols, 0, size);

	ll[0] = minX; ll[1] = minY;
	lr[0] = maxX; lr[1] = minY;
	tl[0] = minX; tl[1] = maxY;
	tr[0] = maxX; tr[1] = maxY;
	p[0] = column; p[1] = row;
	
	if (doesIntersect(ll, lr, p, tr) == 1 || doesIntersect(tr, tl, p, lr) == 1
		|| doesIntersect(ll, lr, p, tl) == 1 || doesIntersect(tr, tl, p, ll) == 1) {
		outPixels[index] = 0;
	}
	//todo have threads evaluate contours within their shared memory region
	//they they then should output their results to the output array e.g final image
	// get max min values for this region then remove all points that lie within a given polygon
	//lowerLeft = minX, minY
	//lowerRight = minY, maxX
	//topLeft = maxY, minX
	//topRight = maxX, maxY
}

__global__ void maxMin(int* values, int* max, int* min,int numRegions, int numRows, int numCols,int xOrY) {
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int index = row * numCols + column;

	int size = numRows * numCols;
	
	extern __shared__ int regionalMax[];
	extern __shared__ int regionalMin[];

	if (index < size) {
		if (values[index] > 0) {
			if (xOrY == 0) {
				int value = row;
				int region = index % numRegions;

				if (atomicMax(&regionalMax[region], value) < value) {
					atomicMax(max, value);
				}

				if (atomicMin(&regionalMin[region], value) > value) {
					atomicMin(min, value);
				}
			}
			else {
				int value = column;
				int region = index % numRegions;

				if (atomicMax(&regionalMax[region], value) < value) {
					atomicMax(max, value);
				}

				if (atomicMin(&regionalMin[region], value) > value) {
					atomicMin(min, value);
				}
			}
		}
	}
}

__global__ void medianFilter(int* pixels, int* outPixels, int numRows, int numCols, int numFilter) {
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int index = row * numCols + column;

	int size = numRows * numCols;

	const int limit = 50;
	
	if ((index < size) && (numFilter < limit)) {
		int window[limit],i;
		for (i = 0; i < limit; i++) {
			window[i] = pixels[index];
		}

		for (i = 0; i < numFilter; i++) {
			if (index + i < size) {
				window[i] = pixels[index + i];
			}
		}
		sort(window, numFilter);
		outPixels[index] = window[numFilter/2];
	}
}

__global__ void guoHallThinning(int* pixels, int* outPixels, int numRows, int numCols, int iter) {
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int index = row * numCols + column;

	int size = numRows * numCols;

	int p2, p3, p4, p5, p6, p7, p8, p9;
	int a2, a3, a4, a5, a6, a7, a8, a9;

	if (index < size) {
		p2 = pixels[(row * numCols + (column - 1))];
		p3 = pixels[((row + 1) * numCols + (column - 1))];
		p4 = pixels[((row + 1) * numCols + column)];
		p5 = pixels[((row + 1) * numCols + (column + 1))];
		p6 = pixels[(row * numCols + (column + 1))];
		p7 = pixels[((row - 1) * numCols + (column + 1))];
		p8 = pixels[((row - 1) * numCols + column)];
		p9 = pixels[((row - 1) * numCols + (column - 1))];

		a2 = (p2 > 0) ? 1 : 0;
		a3 = (p3 > 0) ? 1 : 0;
		a4 = (p4 > 0) ? 1 : 0;
		a5 = (p5 > 0) ? 1 : 0;
		a6 = (p6 > 0) ? 1 : 0;
		a7 = (p7 > 0) ? 1 : 0;
		a8 = (p8 > 0) ? 1 : 0;
		a9 = (p9 > 0) ? 1 : 0;

		int C = (!a2 & (a3 | a4)) + (!a4 & (a5 | a6)) +
			(!a6 & (a7 | a8)) + (!a8 & (a9 | a2));
		
		int N1 = (a9 | a2) + (a3 | a4) + (a5 | a6) + (a7 | a8);
		int N2 = (a2 | a3) + (a4 | a5) + (a6 | a7) + (a8 | a9);
		
		int N = N1 < N2 ? N1 : N2;
		
		int m = iter == 0 ? ((a6 | a7 | !a9) & a8) : ((a2 | a3 | !a5) & a4);

		if (C == 1 && (N >= 2 && N <= 3) & m == 0) {
			outPixels[index] = 0;
		}
	}
}

__global__ void zsThinning(int* pixels, int* outPixels, int numRows, int numCols, int iter) {
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int index = row * numCols + column;

	int size = numRows * numCols;

	int p2, p3, p4, p5, p6, p7, p8, p9;
	int a2, a3, a4, a5, a6, a7, a8, a9;

	if (index < size-1) {
		if (!(column + 1 < numCols && column - 1 > 0 && row + 1 < numRows && row - 1 > 0))return;

		p2 = pixels[(row * numCols + (column - 1))];
		p3 = pixels[((row + 1) * numCols + (column - 1))];
		p4 = pixels[((row + 1) * numCols + column)];
		p5 = pixels[((row + 1) * numCols + (column + 1))];
		p6 = pixels[(row * numCols + (column + 1))];
		p7 = pixels[((row - 1) * numCols + (column + 1))];
		p8 = pixels[((row - 1) * numCols + column)];
		p9 = pixels[((row - 1) * numCols + (column - 1))];

		int A = (p2 == 0 && p3 == 255) + (p3 == 0 && p4 == 255) +
				(p4 == 0 && p5 == 255) + (p5 == 0 && p6 == 255) +
				(p6 == 0 && p7 == 255) + (p7 == 0 && p8 == 255) +
				(p8 == 0 && p9 == 255) + (p9 == 0 && p2 == 255);
		
		a2 = (p2 > 0) ? 1 : 0;
		a3 = (p3 > 0) ? 1 : 0;
		a4 = (p4 > 0) ? 1 : 0;
		a5 = (p5 > 0) ? 1 : 0;
		a6 = (p6 > 0) ? 1 : 0;
		a7 = (p7 > 0) ? 1 : 0;
		a8 = (p8 > 0) ? 1 : 0;
		a9 = (p9 > 0) ? 1 : 0;
		
		int B = a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9;
		int m1 = iter == 0 ? (a2 * a4 * a6) : (a2 * a4 * a8);
		int m2 = iter == 0 ? (a4 * a6 * a8) : (a2 * a6 * a8);

		if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
			outPixels[index] = 0;
		}
	}
}

__global__ void gaussBlur(int* pixels, int* outPixels,int numRows,int numCols,float radius) {
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	
	int index = row * numCols + column;
	int size = (numRows*numCols);
	int xIndex, yIndex;
	
	int sigRadii = ceil(radius*2.57);
	
	float weight, weightedSum = 1, value, pi = 3.141592653;

	if ( index < size-1) {
		for (yIndex = (row - sigRadii); yIndex < (row + sigRadii + 1); yIndex++) {
			for (xIndex = (column - sigRadii); xIndex < (column + sigRadii + 1); xIndex++) {
				int x = min(numCols - 1, max(0, xIndex));
				int y = min(numRows - 1, max(0, yIndex));
				
				int blurId = y * numCols + x;

				float sqDist = (xIndex - row)*(xIndex - row) + (yIndex - column)*(yIndex - column);
				
				weight = exp(-sqDist / (2 * radius*radius) / (pi * 2 * radius*radius));
				
				if (blurId < size) {
					value += pixels[blurId] * weight;
					weightedSum += weight;
				}
			}
		}
		if (weightedSum > 0) {
			outPixels[index] = (int)round((value / weightedSum));
		}
	}
}

__global__ void gaussBlurKernel(int* pixels, int* outPixels, int numRows, int numCols, float sigma) {
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int index = row * numCols + column;
	int size = (numRows*numCols);

	/*float kernel[7*7] = {
		0.000977,	0.00332,	0.006914,	0.008829,	0.006914,	0.00332,	0.000977,
		0.00332,	0.011286,	0.023505,	0.030014,	0.023505,	0.011286,	0.00332,
		0.006914,	0.023505,	0.048952,	0.062509,	0.048952,	0.023505,	0.006914,
		0.008829,	0.030014,	0.062509,	0.07982,	0.062509,	0.030014,	0.008829,
		0.006914,	0.023505,	0.048952,	0.062509,	0.048952,	0.023505,	0.006914,
		0.00332,	0.011286,	0.023505,	0.030014,	0.023505,	0.011286,	0.00332,
		0.000977,	0.00332,	0.006914,	0.008829,	0.006914,	0.00332,	0.000977 };
	*/
	/*float xMat[9] = { 0,-1,0,
					-1,4,-1,
					0,-1,0 };
	float yMat[9] = { -1,-1,-1,
					  -1,8,-1,
					-1,-1,-1 };*/ //laplace of gauss kernels

	const int kernalSize = 5 * 5;
	float kernel[kernalSize];
	createGaussMask(5, sigma, kernel);
	if (index == 10) {
		int k;
		for (k = 0; k < kernalSize; k++) {
			printf("%f, ", kernel[k]);
		}
	}
	int i, j;

	float value = 1,sum = 1;
	if (index < size) {
		//approxGauss5x5(pixels, outPixels, row, column, numRows, numCols);
		for (i = 0; i < 5; i++) {
			for (j = 0; j < 5; j++) {
				int y = row + j - 1;
				int x = column + i - 1;

				index = y * numCols + x;

				if (index < size) {
					value += pixels[index] * kernel[i * 5 + j];
					sum += value;
				}
			}
		}
		outPixels[index] = (int)round(value/sum);
	}
}

__global__ void sharpen(int* pixels, int* outPixels, int numRows, int numCols) {
	float mat[9] = { 0,-1,0,-1,5,-1,0,-1,0 };

	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int index, size = (numRows*numCols);

	int mag = 0;

	int i, j;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			int y = row + j - 1;
			int x = column + i - 1;

			index = y * numCols + x;

			if (index < size) {
				mag += pixels[index] * mat[i * 3 + j];
			}
		}
	}

	index = row * numCols + column;
}

__global__ void edgeDetect(int* pixels, int* outPixels, int numRows, int numCols, int threshold) {
	float highRatio = threshold;// ((float)threshold)*.07843137*.75;
	float lowRatio = highRatio/1.2;//.25*highRatio;
	float magX = 0, magY = 0;

	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int index = row * numCols + column, size = (numRows*numCols);

	float vector[2];
	
	sobelFilter(pixels, row, column, numRows, numCols, vector);
	
	/*if (index == 10) {
		printf("vector values are mag=%f, ori=%f lower limit=%f \n", vector[0], vector[1], lowRatio);
	}*/

	if(index < size){
		if (vector[0] < lowRatio) {//not edge
			outPixels[index] = 0;
		}
		else if(vector[0] > highRatio) {//strong edge
			outPixels[index] = 255;//max((int)vector[0],threshold);//255;
		}
		else if(vector[0] > lowRatio && vector[0] < highRatio) {//weak edge perform non max suppression
			if (row > 1 && column > 1 && row < size - 1 && column < size - 1) {
				
				if (vector[1] >= 0 && vector[1] <= 45 || vector[1] < -135 && vector[1] >= -180) {//interpolation r = alpha*B +(1-alpha)A
					float mul = abs(magY / vector[0]);

					float northEast[2];
					sobelFilter(pixels, row + 1, column + 1, numRows, numCols, northEast);
					//northEast[0] = magnitudes[(sRow + 1) * sWidth + (sColumn + 1)];

					float southWest[2];
					sobelFilter(pixels, row - 1, column - 1, numRows, numCols, southWest);
					//southWest[0] = magnitudes[(sRow - 1) * sWidth + (sColumn - 1)];

					float east[2];
					sobelFilter(pixels, row, column + 1, numRows, numCols, east);
					//east[0] = magnitudes[sRow  * sWidth + (sColumn + 1)];

					float west[2];
					sobelFilter(pixels, row, column - 1, numRows, numCols, west);
					//west[0] = magnitudes[sRow  * sWidth + (sColumn - 1)];

					if (vector[0] >= (northEast[0] - east[0] * mul + east[0])
						&& vector[0] >= (southWest[0] - west[0] * mul + west[0])) {
						int check = checkStrongNeighbor(vector[0], northEast[0], highRatio)
									+ checkStrongNeighbor(vector[0], southWest[0], highRatio)
									+ checkStrongNeighbor(vector[0], east[0], highRatio)
									+ checkStrongNeighbor(vector[0], west[0], highRatio);
						if (check > 0) {
							outPixels[index] = 255;// (int)vector[0];
						}
						else {
							outPixels[index] = 0;
						}
					}
					else {
						outPixels[index] = 0;
					}
				}
				else if (vector[1] > 45 && vector[1] <= 90 || vector[1] < -90 && vector[1] > 135) {
					float mul = abs(magX / vector[0]);

					float northEast[2];
					sobelFilter(pixels, row + 1, column + 1, numRows, numCols, northEast);

					float southWest[2];
					sobelFilter(pixels, row - 1, column - 1, numRows, numCols, southWest);

					float north[2];
					sobelFilter(pixels, row + 1, column, numRows, numCols, north);

					float south[2];
					sobelFilter(pixels, row - 1, column, numRows, numCols, south);

					if (vector[0] >= (northEast[0] - north[0] * mul + north[0])
						&& vector[0] >= (southWest[0] - south[0]* mul + south[0])) {
						int check = checkStrongNeighbor(vector[0], northEast[0], highRatio)
									+ checkStrongNeighbor(vector[0], southWest[0], highRatio)
									+ checkStrongNeighbor(vector[0], north[0], highRatio)
									+ checkStrongNeighbor(vector[0], south[0], highRatio);
						if (check > 0) {
							outPixels[index] = 255;// (int)vector[0];
						}
						else {
							outPixels[index] = 0;
						}
					}
					else {
						outPixels[index] = 0;
					}
				}
				else if (vector[1] > 90 && vector[1] <= 135 || vector[1] < -45 && vector[1] >= -90) {
					float mul = abs(magX / vector[0]);

					float northWest[2];
					sobelFilter(pixels, row + 1, column - 1, numRows, numCols, northWest);

					float southEast[2];
					sobelFilter(pixels, row - 1, column + 1, numRows, numCols, southEast);
					
					float north[2];
					sobelFilter(pixels, row + 1, column, numRows, numCols, north);

					float south[2];
					sobelFilter(pixels, row - 1, column, numRows, numCols, south);

					if (vector[0] >= (northWest[0] - north[0] * mul + north[0])
						&& vector[0] >= (southEast[0] - south[0]* mul + south[0])) {
						int check = checkStrongNeighbor(vector[0], northWest[0], highRatio)
							+ checkStrongNeighbor(vector[0], southEast[0], highRatio)
							+ checkStrongNeighbor(vector[0], north[0], highRatio)
							+ checkStrongNeighbor(vector[0], south[0], highRatio);
						if (check > 0) {
							outPixels[index] = 255;// (int)vector[0];
						}
						else {
							outPixels[index] = 0;
						}
					}
					else {
						outPixels[index] = 0;
					}
				}
				else if (vector[1] > 135 && vector[1] <= 180 || vector[1] < 0 && vector[1] >= -45) {
					float mul = abs(magX / vector[0]);

					float northWest[2];
					sobelFilter(pixels, row + 1, column - 1, numRows, numCols, northWest);

					float southEast[2];
					sobelFilter(pixels, row - 1, column + 1, numRows, numCols, southEast);

					float east[2];
					sobelFilter(pixels, row, column + 1, numRows, numCols, east);

					float west[2];
					sobelFilter(pixels, row, column - 1, numRows, numCols, west);

					if (vector[0] >= (northWest[0] - west[0] * mul + west[0])
						&& vector[0] >= (southEast[0] - east[0] * mul + east[0])) {
						int check = checkStrongNeighbor(vector[0], northWest[0], highRatio)
							+ checkStrongNeighbor(vector[0], southEast[0], highRatio)
							+ checkStrongNeighbor(vector[0], east[0], highRatio)
							+ checkStrongNeighbor(vector[0], west[0], highRatio);
						if (check > 0) {
							outPixels[index] = 255;// (int)vector[0];
						}
						else {
							outPixels[index] = 0;
						}
					}
					else {
						outPixels[index] = 0;
					}
				}
			}
		}
	}	
	
}

__global__ void drawEdge(int* pixels, int numRows, int numCols, int edgeWidth) {
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int id = row * numCols + column;
	int size = (numRows*numCols);

	int leftCE = edgeWidth;
	int rightCE = numCols - edgeWidth;

	int topRE = edgeWidth;
	int bottomRE = numRows - edgeWidth;

	if (id < size) {
		if (column <= leftCE || column >= rightCE) {
			pixels[id] = 0;
		}
		if (row <= topRE || row >= bottomRE) {
			pixels[id] = 0;
		}
	}
}

//p[0]=row p[1]=column
__global__ void drawCircle(int* pixels, int numCol, int numRows, int point[], int radius) {
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int id = row * numCol + column;
	int size = (numRows*numCol);

	int origin[2];
	origin[0] = row;
	origin[1] = column;

 	float dist = distance(origin, point);

	if (id < (size)) {
		if (dist < radius) {
			pixels[id] = 0;
		}
	}
}

//p[0] = row p[1] = column
__global__ void drawLine(int* pixels, int numCol, int numRows, int p1[], int p2[]) {//y=mx+b
	int size = numRows * numCol;

	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int id = row * numCol + column;
	
	if (id < size) {
		if( p1[1] == p2[1]){
			if((row >= p1[0] && row <= p2[0]) && column == p1[1]){//infinite slope
				pixels[id] = 0;
			}
		}else if(p1[0] == p2[0]){
			if((column >= p1[1] && column <= p2[1]) && row == p1[0]){
				pixels[id] = 0;
			}
		}else{
			float yDiff = p2[0]-p1[0];
			float xDiff = p2[1]-p1[1];
			
			float slope = (yDiff / xDiff);//slope=m b=0

			float calcValue = slope*(float)column;
			float onLine = abs((float)row-calcValue);

			if(slope < 0){
				calcValue = slope*(float)(column) + (float)p2[1]/1.32;
				onLine = abs((float)row-calcValue);
			}
			if(onLine < 0.5 && row <= max(p1[0],p2[0]) && column <= (max(p1[1],p2[1]))) {//if y=mx+b
				pixels[id] = 0;
				//printf("the calcValue=%f, the xDiff=%f ,the slope is %f\n",calcValue,xDiff,slope);
			}
		}
	}
}
