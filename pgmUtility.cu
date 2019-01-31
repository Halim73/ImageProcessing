
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "pgmUtility.h"

// Implement or define each function prototypes listed in pgmUtility.h file.
// NOTE: You can NOT change the input, output, and argument type of the functions in pgmUtility.h
// NOTE: You can NOT change the prototype of any functions listed in pgmUtility.h

int* pgmRead(char** header, int* numRows, int* numCols, FILE* in) {
	int pixelIntensity = 0,i,linearScale,* pgmFileIn;
	long index;
	char buff[100];
	for (i = 0; i < 4; i++) {
		fgets(buff, 100, in);
		strcpy(header[i], buff);
		//printf("buffer contains: %s\n", buff);
	}
	//printf("magicNumber=%s, comment=%s\n",header[0],header[1]);

	sscanf(header[2], "%d %d", numCols, numRows);
	sscanf(header[3], "%d", &pixelIntensity);

	linearScale = (*numRows)*(*numCols) + (*numRows/4);
	pgmFileIn = (int*)malloc(linearScale*sizeof(int));

	//printf("rows=%d, columns=%d\n", *numRows, *numCols);
	//printf("intensity is %d\n",pixelIntensity);
	//printf("the total number of elements is %d\n", linearScale);
	index = 0;
	while ((fgets(buff, 100, in)) != NULL) {
		char* token = (char*)malloc(100);
		char* start = buff;
		
		while ((token = strtok_r(start, " ", &start)) != NULL) {
			int value = -1;
			if (strcmp(token, " ") != 0) {
				sscanf(token, "%d ", &value);
				//printf("value=%d\n", value);
				if(value >= 0 && index < linearScale)pgmFileIn[index++] = value;
			}
		}
	}

	return pgmFileIn;
}

int* pgmRead2(char** header, int* numRows, int* numCols,int* intensity, FILE* in) {
	int i, linearScale, *pgmFileIn;
	long index;
	char buff[100];
	for (i = 0; i < 4; i++) {
		fgets(buff, 100, in);
		strcpy(header[i], buff);
		//printf("buffer contains: %s\n", buff);
	}
	//printf("magicNumber=%s, comment=%s\n",header[0],header[1]);

	sscanf(header[2], "%d %d", numCols, numRows);
	sscanf(header[3], "%d", intensity);

	linearScale = (*numRows)*(*numCols) + (*numRows / 4);
	pgmFileIn = (int*)malloc(linearScale * sizeof(int));

	//printf("rows=%d, columns=%d\n", *numRows, *numCols);
	//printf("intensity is %d\n",*intensity);
	//printf("the total number of elements is %d\n", linearScale);
	index = 0;
	while ((fgets(buff, 100, in)) != NULL) {
		char* token = (char*)malloc(100);
		char* start = buff;

		while ((token = strtok_r(start, " ", &start)) != NULL) {
			int value = -1;
			if (strcmp(token, " ") != 0) {
				sscanf(token, "%d ", &value);
				//printf("value=%d\n", value);
				if (value >= 0 && index < linearScale)pgmFileIn[index++] = value;
			}
		}
	}

	return pgmFileIn;
}

int pgmGetCountour(int* pixels, int* outPixels, int numRows, int numCols) {
	int* devicePixels, *deviceOutPixels;

	cudaError_t memError = cudaMalloc((void**)&devicePixels, (size_t)((numRows*numCols) * sizeof(int)));
	cudaError_t memError2 = cudaMalloc((void**)&deviceOutPixels, (size_t)((numRows*numCols) * sizeof(int)));

	if (memError != cudaSuccess) {
		perror("error allocating memory");
		exit(7);
	}
	if (memError2 != cudaSuccess) {
		perror("error allocating memory2");
		exit(7);
	}

	cudaMemcpy(devicePixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutPixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 32;
	block.y = 32;
	grid.x = ceil((float)numCols / block.x);
	grid.y = ceil((float)numRows / block.y);

	setBlockPixels << <grid, block, (numRows)*sizeof(int) >> > (devicePixels, deviceOutPixels, numRows, numCols, numRows);

	cudaMemcpy(pixels, deviceOutPixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyDeviceToHost);

	cudaFree(devicePixels);
	cudaFree(deviceOutPixels);

	return 0;
}

int pgmMedianFilter(int* pixels, int* outPixels, int numRows, int numCols, int numFilter) {
	int* devicePixels, *deviceOutPixels;

	cudaError_t memError = cudaMalloc((void**)&devicePixels, (size_t)((numRows*numCols) * sizeof(int)));
	cudaError_t memError2 = cudaMalloc((void**)&deviceOutPixels, (size_t)((numRows*numCols) * sizeof(int)));

	if (memError != cudaSuccess) {
		perror("error allocating memory");
		exit(7);
	}
	if (memError2 != cudaSuccess) {
		perror("error allocating memory2");
		exit(7);
	}

	cudaMemcpy(devicePixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutPixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 32;
	block.y = 32;
	grid.x = ceil((float)numCols / block.x);
	grid.y = ceil((float)numRows / block.y);

	medianFilter << <grid, block >> > (devicePixels, deviceOutPixels, numRows, numCols, numFilter);

	cudaMemcpy(pixels, deviceOutPixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyDeviceToHost);

	cudaFree(devicePixels);
	cudaFree(deviceOutPixels);

	return 0;
}

int pgmGuoHallThinning(int* pixels, int* outPixels, int numRows, int numCols) {
	int* devicePixels, *deviceOutPixels;

	cudaError_t memError = cudaMalloc((void**)&devicePixels, (size_t)((numRows*numCols) * sizeof(int)));
	cudaError_t memError2 = cudaMalloc((void**)&deviceOutPixels, (size_t)((numRows*numCols) * sizeof(int)));

	if (memError != cudaSuccess) {
		perror("error allocating memory");
		exit(7);
	}
	if (memError2 != cudaSuccess) {
		perror("error allocating memory2");
		exit(7);
	}

	cudaMemcpy(devicePixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutPixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 32;
	block.y = 32;
	grid.x = ceil((float)numCols / block.x);
	grid.y = ceil((float)numRows / block.y);

	guoHallThinning << <grid, block >> > (devicePixels, deviceOutPixels, numRows, numCols, 0);
	guoHallThinning << <grid, block >> > (deviceOutPixels, deviceOutPixels, numRows, numCols, 1);

	cudaMemcpy(pixels, deviceOutPixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyDeviceToHost);

	cudaFree(devicePixels);
	cudaFree(deviceOutPixels);

	return 0;
}

int pgmZSThinning(int* pixels, int* outPixels, int numRows, int numCols) {
	int* devicePixels, *deviceOutPixels;

	cudaError_t memError = cudaMalloc((void**)&devicePixels, (size_t)((numRows*numCols) * sizeof(int)));
	cudaError_t memError2 = cudaMalloc((void**)&deviceOutPixels, (size_t)((numRows*numCols) * sizeof(int)));

	if (memError != cudaSuccess) {
		perror("error allocating memory");
		exit(7);
	}
	if (memError2 != cudaSuccess) {
		perror("error allocating memory2");
		exit(7);
	}

	cudaMemcpy(devicePixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutPixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 32;
	block.y = 32;
	grid.x = ceil((float)numCols / block.x);
	grid.y = ceil((float)numRows / block.y);

	zsThinning << <grid, block >> > (devicePixels, deviceOutPixels, numRows, numCols, 0);
	zsThinning << <grid, block >> > (deviceOutPixels, deviceOutPixels, numRows, numCols, 1);

	cudaMemcpy(pixels, deviceOutPixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyDeviceToHost);

	cudaFree(devicePixels);
	cudaFree(deviceOutPixels);

	return 0;
}

int pgmGaussianBlur(int* pixels, int* outPixels, int numRows, int numCols, float radius) {
	int* devicePixels, *deviceOutPixels;

	cudaError_t memError = cudaMalloc((void**)&devicePixels, (size_t)((numRows*numCols) * sizeof(int)));
	cudaError_t memError2 = cudaMalloc((void**)&deviceOutPixels, (size_t)((numRows*numCols) * sizeof(int)));

	if (memError != cudaSuccess) {
		perror("error allocating memory");
		exit(7);
	}
	if (memError2 != cudaSuccess) {
		perror("error allocating memory2");
		exit(7);
	}

	cudaMemcpy(devicePixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutPixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 32;
	block.y = 32;
	grid.x = ceil((float)numCols / block.x);
	grid.y = ceil((float)numRows / block.y);

	if (radius >= 5) {
		gaussBlur << <grid, block >> > (devicePixels, deviceOutPixels, numRows, numCols, radius);
	}
	else {
		gaussBlurKernel << <grid, block >> > (devicePixels, deviceOutPixels, numRows, numCols,radius);
	}
	
	cudaMemcpy(pixels, deviceOutPixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyDeviceToHost);

	cudaFree(devicePixels);
	cudaFree(deviceOutPixels);

	return 0;
}

int pgmEdgeDetect(int* pixels, int* outPixels, int numRows, int numCols,int threshold) {
	int* devicePixels,*deviceOutPixels;

	clock_t start, end;
	start = clock();

	cudaError_t memError = cudaMalloc((void**)&devicePixels, (size_t)((numRows*numCols) * sizeof(int)));
	cudaError_t memError2 = cudaMalloc((void**)&deviceOutPixels, (size_t)((numRows*numCols) * sizeof(int)));

	//unsigned int sharedSize = ((34) * 4) * sizeof(float);

	if (memError != cudaSuccess) {
		perror("error allocating memory");
		exit(7);
	}
	if (memError2 != cudaSuccess) {
		perror("error allocating memory2");
		exit(7);
	}

	cudaMemcpy(devicePixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutPixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);
	
	dim3 grid, block;
	block.x = 32;
	block.y = 32;
	grid.x = ceil((float)numCols / block.x);
	grid.y = ceil((float)numRows / block.y);

	edgeDetect << <grid, block >> > (devicePixels, deviceOutPixels, numRows, numCols,threshold);
	cudaThreadSynchronize();
	zsThinning << <grid, block >> > (deviceOutPixels, deviceOutPixels, numRows, numCols, 0);
	zsThinning << <grid, block >> > (deviceOutPixels, deviceOutPixels, numRows, numCols, 1);

	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
		printf("Cuda Kernel Launch error -- %s\n", cudaGetErrorString(code));
	}

	cudaMemcpy(pixels, deviceOutPixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyDeviceToHost);
	end = clock();
	double time_taken = double(end - start) / double(CLOCKS_PER_SEC);

	printf("Processing time: %f (s)\n", time_taken);

	cudaFree(devicePixels);
	cudaFree(deviceOutPixels);

	return 0;
}

int pgmDrawCircle( int* pixels, int numRows, int numCols, int centerRow,int centerCol, int radius, char** header){
	int point[2];
	point[0] = centerRow;
	point[1] = centerCol;

	int* devicePixels, *devicePoint;

	cudaError_t memError = cudaMalloc((void**)&devicePixels, (size_t)((numRows*numCols) * sizeof(int)));
	cudaError_t memError1 = cudaMalloc((void**)&devicePoint, (size_t)(2 * sizeof(int)));

	if (memError != cudaSuccess) {
		//handle
	}
	if (memError1 != cudaSuccess) {
		//handle
	}

	cudaMemcpy(devicePixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);
	cudaMemcpy(devicePoint, point, (size_t)(2 * sizeof(int)), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 32;
	block.y = 32;
	grid.x = ceil((float)numCols / block.x);
	grid.y = ceil((float)numRows / block.y);

	drawCircle << <grid,block >> > (devicePixels,numCols,numRows,devicePoint,radius);

	cudaMemcpy(pixels, devicePixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyDeviceToHost);

	cudaFree(devicePixels);
	cudaFree(devicePoint);
	return 0;
}

int pgmDrawEdge(int* pixels, int numRows, int numCols, int edgeWidth, char** header) {
	int* devicePixels;

	cudaError_t memError = cudaMalloc((void**)&devicePixels, (size_t)((numRows*numCols) * sizeof(int)));

	if (memError != cudaSuccess) {
		//handle
	}

	cudaMemcpy(devicePixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 32;
	block.y = 32;
	grid.x = ceil((float)numCols / block.x);
	grid.y = ceil((float)numRows / block.y);

	drawEdge << <grid,block >> > (devicePixels, numRows, numCols, edgeWidth);

	cudaMemcpy(pixels, devicePixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyDeviceToHost);

	cudaFree(devicePixels);

	return 0;
}

int pgmDrawLine(int* pixels, int numRows, int numCols, char** header, int p1row, int p1col, int p2row, int p2col) {
	int* p1 = (int*)malloc(2 * sizeof(int));
	int* p2 = (int*)malloc(2 * sizeof(int));
	
	int* devicePixels,* deviceP1,* deviceP2;

	p1[0] = p1row; p1[1] = p1col;
	p2[0] = p2row; p2[1] = p2col;

	cudaError_t memError = cudaMalloc((void**)&devicePixels, (size_t)((numRows*numCols) * sizeof(int)));
	cudaError_t memError1 = cudaMalloc((void**)&deviceP1, (size_t)(2 * sizeof(int)));
	cudaError_t memError2 = cudaMalloc((void**)&deviceP2, (size_t)(2 * sizeof(int)));

	if (memError != cudaSuccess) {
		//handle
	}
	if (memError1 != cudaSuccess) {
		//handle
	}
	if (memError2 != cudaSuccess) {
		//handle
	}

	cudaMemcpy(devicePixels, pixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceP1, p1, (size_t)(2 * sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceP2, p2, (size_t)(2 * sizeof(int)), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 32;
	block.y = 32;
	grid.x = ceil((float)numCols / block.x);
	grid.y = ceil((float)numRows / block.y);

	drawLine<<<grid,block>>>(devicePixels, numCols, numRows, deviceP1,deviceP2);
	
	cudaMemcpy(pixels, devicePixels, ((size_t)((numRows*numCols) * sizeof(int))), cudaMemcpyDeviceToHost);

	cudaFree(devicePixels);
	cudaFree(deviceP1);
	cudaFree(deviceP2);

	return 0;
}

int pgmWrite(const char** header, const int* pixels, int numRows, int numCols, FILE *out ){
	int i,bounds;
	for(i=0;i<4;i++){
		fprintf(out,"%s",header[i]);
	}	
	
	bounds = (numRows*numCols);

	for(int i=0;i<bounds;i++){
		fprintf(out,"%d ",(pixels[i]));
		//printf("pixels %d\n", pixels[i]);
	}
	fclose(out);
	return 0;
}
