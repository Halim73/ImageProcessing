#include "main.h"

int doEdge(int* numRows,int* numColumns,int* inPixels,char** header,char** argv,int ret,int argc) {
	if (argc < 5) {
		perror("Error wrong usage!  -e edgeWidth  oldImageFile  newImageFile\n");
		exit(4);
	}

	int edgeWidth, edgeDraw;
	
	clock_t start, end;
	double cost;

	sscanf(argv[2], "%d", &edgeWidth);

	FILE* file = fopen(argv[3], "r");
	FILE* outFile = fopen(argv[4], "w+");
	char* fileName = (char*)malloc(600);
	strcat(fileName, "serial_");
	strcat(fileName, argv[4]);
	FILE* serialOut = fopen(fileName, "w+");

	if (file == NULL) {
		perror("sorry couldnt find file\n");
		exit(2);
	}
	
	inPixels = (int*)pgmRead(header, numRows, numColumns, file);
	
	start = clock();
	edgeDraw = pgmDrawEdge(inPixels, *numRows, *numColumns, edgeWidth, header);
	end = clock();
	
	cost = ((double)(end - start)) / CLOCKS_PER_SEC;

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, outFile);
	printf("Time cost for GPU EDGE solution is %f\n", cost);

	fclose(file);
	file = fopen(argv[3], "r");

	inPixels = (int*)pgmRead(header, numRows, numColumns, file);

	start = clock();
	serialEdgeDraw(inPixels, *numRows, *numColumns, edgeWidth);
	end = clock();

	cost = ((double)(end - start)) / CLOCKS_PER_SEC;

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, serialOut);
	printf("Time cost for SERIAL EDGE solution is %f\n\n", cost);

	fclose(file);
	return ret;
}

int doCircle(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc) {
	if (argc < 7) {
		perror("Wrong usage! -c circleCenterRow circleCenterCol radius  oldImageFile  newImageFile\n\n");
		exit(4);
	}
	//printf("CIRCLE>>>>>>the command is %s %s %s %s %s %s\n\n", argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);

	int centerRow, centerCol, radius, circleDraw;
	
	clock_t start, end;
	double cost;

	sscanf(argv[2], "%d", &centerRow);
	sscanf(argv[3], "%d", &centerCol);
	sscanf(argv[4], "%d", &radius);
	//printf("retrieved parameters: cRow=%d,cCol=%d,radius=%d\n", centerRow, centerCol, radius);
	FILE* file = fopen(argv[5], "r");
	FILE* outFile = fopen(argv[6], "w+");
	char* fileName = (char*)malloc(600);
	strcat(fileName, "serial_");
	strcat(fileName, argv[6]);
	FILE* serialOut = fopen(fileName, "w+");

	if (file == NULL) {
		perror("Sorry couldnt find file\n");
		exit(2);
	}
	//printf("about to read in pixels\n");
	inPixels = (int*)pgmRead(header, numRows, numColumns, file);
	//printf("read in pixels\n");
	start = clock();
	circleDraw = pgmDrawCircle(inPixels, *numRows, *numColumns, centerRow, centerCol, radius, header);
	end = clock();
	cost = ((double)(end - start)) / CLOCKS_PER_SEC;

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, outFile);
	printf("Time cost for GPU CIRCLE solution is %f\n", cost);

	fclose(file);
	file = fopen(argv[5], "r");

	inPixels = (int*)pgmRead(header, numRows, numColumns, file);

	start = clock();
	serialDrawCircle(inPixels, *numRows, *numColumns, centerRow, centerCol, radius);
	end = clock();
	cost = ((double)(end - start)) / CLOCKS_PER_SEC;

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, serialOut);
	printf("Time cost for SERIAL CIRCLE solution is %f\n", cost);
	fclose(file);
	return ret;
}

int doCountourMap(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc) {
	if (argc < 4) {
		perror("Error Wrong Usag! ./myPaint -C inFile outFile");
		exit(5);
	}

	FILE* file = fopen(argv[2], "r");
	FILE* outFile = fopen(argv[3], "w+");

	if (file == NULL) {
		perror("sorry couldnt find file\n");
		exit(2);
	}

	inPixels = (int*)pgmRead(header, numRows, numColumns, file);

	int* contour = (int*)malloc(*numRows**numColumns * sizeof(int));//(int*)pgmRead(header, numRows, numColumns, file);
	memset(contour, 0, *numRows**numColumns * sizeof(int));

	printf("finished reading in pixels\n");
	int success = pgmGetCountour(inPixels, contour, *numRows, *numColumns);

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, outFile);
	printf("finished contour mapping of pixels\n");

	fclose(file);
}

int doMedianFilter(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc) {
	if (argc < 5) {
		perror("Error Wrong Usag! ./myPaint -m filterLevel inFile outFile");
		exit(5);
	}

	int numFilter;
	sscanf(argv[2], "%d", &numFilter);

	FILE* file = fopen(argv[3], "r");
	FILE* outFile = fopen(argv[4], "w+");

	if (file == NULL) {
		perror("sorry couldnt find file\n");
		exit(2);
	}

	inPixels = (int*)pgmRead(header, numRows, numColumns, file);

	int* thinPixels = (int*)malloc(*numRows**numColumns * sizeof(int));//(int*)pgmRead(header, numRows, numColumns, file);
	memset(thinPixels, 0, *numRows**numColumns * sizeof(int));

	printf("finished reading in pixels the filter is %d\n",numFilter);
	int success =  pgmMedianFilter(inPixels, thinPixels, *numRows, *numColumns,numFilter);

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, outFile);
	printf("finished filtering pixels\n");

	fclose(file);
}

int doZSThinning(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc) {
	if (argc < 4) {
		perror("Error Wrong Usag! ./myPaint -z inFile outFile");
		exit(5);
	}

	FILE* file = fopen(argv[2], "r");
	FILE* outFile = fopen(argv[3], "w+");

	if (file == NULL) {
		perror("sorry couldnt find file\n");
		exit(2);
	}

	inPixels = (int*)pgmRead(header, numRows, numColumns, file);

	int* thinPixels = (int*)malloc(*numRows**numColumns*sizeof(int));//(int*)pgmRead(header, numRows, numColumns, file);
	memset(thinPixels, 0, *numRows**numColumns * sizeof(int));

	printf("finished reading in pixels\n");
	int success = pgmZSThinning(inPixels, thinPixels, *numRows, *numColumns);

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, outFile);
	printf("finished thinning pixels (zhang suen)\n");

	fclose(file);
}

int doGuoHallThinning(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc) {
	if (argc < 4) {
		perror("Error Wrong Usag! ./myPaint -h inFile outFile");
		exit(5);
	}

	FILE* file = fopen(argv[2], "r");
	FILE* outFile = fopen(argv[3], "w+");

	if (file == NULL) {
		perror("sorry couldnt find file\n");
		exit(2);
	}

	inPixels = (int*)pgmRead(header, numRows, numColumns, file);

	int* thinPixels = (int*)malloc(*numRows**numColumns * sizeof(int));//(int*)pgmRead(header, numRows, numColumns, file);
	memset(thinPixels, 0, *numRows**numColumns * sizeof(int));

	printf("finished reading in pixels\n");
	int success = pgmGuoHallThinning(inPixels, thinPixels, *numRows, *numColumns);

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, outFile);
	printf("finished thinning pixels (Guo Hall)\n");

	fclose(file);
}

int doGaussBlur(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc) {
	if (argc < 5) {
		perror("Error Wrong Usag! ./myPaint -g radius inFile outFile");
		exit(5);
	}

	float radius;
	sscanf(argv[2], "%f", &radius);

	FILE* file = fopen(argv[3], "r");
	FILE* outFile = fopen(argv[4], "w+");

	if (file == NULL) {
		perror("sorry couldnt find file\n");
		exit(2);
	}

	inPixels = (int*)pgmRead(header, numRows, numColumns, file);
	fclose(file);

	file = fopen(argv[3], "r");

	int* blurPixels = (int*)pgmRead(header, numRows, numColumns, file);
	printf("finished reading in pixels\n");
	int success = pgmGaussianBlur(inPixels, blurPixels, *numRows, *numColumns, radius);

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, outFile);
	printf("finished blurring pixels\n");

	fclose(file);
}

int doEdgeDetect(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc) {
	if (argc < 5) {
		perror("Error Wrong Usag! ./myPaint -d threshold inFile outFile");
		exit(5);
	}

	int threshold,intensity;
	sscanf(argv[2], "%d", &threshold);

	//printf("finished getting threshold=%d\n", threshold);

	FILE* file = fopen(argv[3], "r");
	FILE* outFile = fopen(argv[4], "w+");

	if (file == NULL) {
		perror("sorry couldnt find file\n");
		exit(2);
	}

	inPixels = (int*)pgmRead2(header, numRows, numColumns,&intensity, file);
	fclose(file);

	file = fopen(argv[3], "r");
	
	int* edgePixels = (int*)pgmRead(header, numRows, numColumns, file);
	//printf("finished reading in pixels\n");
	int success = pgmEdgeDetect(inPixels,edgePixels,*numRows,*numColumns,threshold);
	sprintf(header[3], "%d\n", 255);
	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, outFile);
	printf("finished detecting edges on %s\n",argv[3]);

	fclose(file);
}

int doLine(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc) {
	if (argc < 8) {
		perror("Wrong usage! -l  p1row  p1col  p2row  p2col  oldImageFile  newImageFile\n\n");
		exit(5);
	}
	//printf("LINE>>>>>>the command is %s %s %s %s %s %s %s\n\n", argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);

	int p1row, p1col, p2row, p2col, lineWrite;
	
	clock_t start, end;
	double cost;

	sscanf(argv[2], "%d", &p1row);
	sscanf(argv[3], "%d", &p1col);
	sscanf(argv[4], "%d", &p2row);
	sscanf(argv[5], "%d", &p2col);

	FILE* file = fopen(argv[6], "r");
	FILE* outFile = fopen(argv[7], "w+");
	char* fileName = (char*)malloc(600);
	strcat(fileName, "serial_");
	strcat(fileName, argv[7]);
	FILE* serialOut = fopen(fileName, "w+");

	if (file == NULL) {
		perror("Sorry couldnt find file\n");
		exit(2);
	}

	inPixels = (int*)pgmRead(header, numRows, numColumns, file);

	start = clock();
	lineWrite = pgmDrawLine(inPixels, *numRows, *numColumns, header, p1row, p1col, p2row, p2col);
	end = clock();
	cost = ((double)(end - start)) / CLOCKS_PER_SEC;

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, outFile);
	printf("Time cost for GPU LINE solution is %f\n", cost);

	fclose(file);
	file = fopen(argv[6], "r");
	
	inPixels = (int*)pgmRead(header, numRows, numColumns, file);

	start = clock();
	serialLineDraw(inPixels, *numRows, *numColumns, p1row, p1col, p2row, p2col);
	end = clock();
	cost = ((double)(end - start)) / CLOCKS_PER_SEC;

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, serialOut);
	printf("Time cost for SERIAL LINE solution is %f\n", cost);

	fclose(file);
}

int doCircleEdge(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc) {
	if (argc < 8) {
		perror("Wrong usage! -c -e circleCenterRow circleCenterCol radius edgeWidth oldImageFile  newImageFile\n\n");
		exit(4);
	}
	//printf("the command is %s %s %s %s %s %s %s %s\n\n", argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],argv[8]);

	int centerRow, centerCol, radius, circleDraw;
	int edgeWidth, edgeDraw;
	
	clock_t start, end;
	double cost;

	sscanf(argv[3], "%d", &centerRow);
	sscanf(argv[4], "%d", &centerCol);
	sscanf(argv[5], "%d", &radius);
	sscanf(argv[6], "%d", &edgeWidth);

	FILE* file = fopen(argv[7], "r");
	FILE* outFile = fopen(argv[8], "w+");
	char* fileName = (char*)malloc(600);
	strcat(fileName, "serial_");
	strcat(fileName, argv[6]);
	FILE* serialOut = fopen(fileName, "w+");

	if (file == NULL) {
		perror("Sorry couldnt find file\n");
		exit(2);
	}

	inPixels = (int*)pgmRead(header, numRows, numColumns, file);

	start = clock();
	circleDraw = pgmDrawCircle(inPixels, *numRows, *numColumns, centerRow, centerCol, radius, header);
	edgeDraw = pgmDrawEdge(inPixels, *numRows, *numColumns, edgeWidth, header);
	end = clock();
	cost = ((double)(end - start)) / CLOCKS_PER_SEC;

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, outFile);
	printf("The GPU CIRCLE EDGE draw time cost is %f\n", cost);

	fclose(file);
	file = fopen(argv[7], "r");

	inPixels = (int*)pgmRead(header, numRows, numColumns, file);

	start = clock();
	serialDrawCircle(inPixels, *numRows, *numColumns, centerRow, centerCol, radius);
	serialEdgeDraw(inPixels, *numRows, *numColumns, edgeWidth);
	end = clock();
	cost = ((double)(end - start)) / CLOCKS_PER_SEC;

	ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, serialOut);
	printf("The SERIAL CIRCLE EDGE draw time cost is %f\n", cost);
	fclose(file);
}

int serialEdgeDraw(int* pixels, int numRows, int numCols, int edgeWidth) {
	int size = (numRows*numCols);

	int leftCE = edgeWidth;
	int rightCE = numCols - edgeWidth;

	int topRE = edgeWidth;
	int bottomRE = numRows - edgeWidth;

	int i,j,index,row,column;
	for (i = 0; i < numCols; i++) {
		column = i;
		for (j = 0; j < numRows; j++) {
			row = j;
			index = (row*numCols) + column;

			if (index < size) {
				if (column <= leftCE || column >= rightCE) {
					pixels[index] = 0;
				}
				if (row <= topRE || row >= bottomRE) {
					pixels[index] = 0;
				}
			}
		}
	}
	//printf("Finished Serial Edge Solution\n");
}

float _distance(int p1[], int p2[]) {
	int xDiff = p2[0] - p1[0];
	int yDiff = p2[1] - p1[1];

	xDiff *= xDiff;
	yDiff *= yDiff;

	float distance = (float)sqrt((float)(xDiff + yDiff));

	return distance;
}

int serialDrawCircle(int* pixels, int numCol, int numRows, int centerRow,int centerCol, int radius) {
	int i, j, index, row, column,size;

	int point[2];
	point[0] = centerRow;
	point[1] = centerCol;

	for (i = 0; i < numCol; i++) {
		for (j = 0; j < numRows; j++) {
			int origin[2];
			origin[0] = i;
			origin[1] = j;

			float dist = _distance(origin, point);
			size = (numCol*numRows);

			int id = i * numRows + j;
			if (id < size && dist < radius) {
				pixels[id] = 0;
			}
		}
	}
	
} 

int serialLineDraw(int* inPixels,int numRows,int numColumns,int p1row,int p1col,int p2row,int p2col){
	int x = p1col,y = p1row,error = 0;
	int dx = p2col-p1col;
	int dy = p2row-p1row;
	
	error = 2*(dy-dx);
	if(p1row < p2row){
		printf("POSITIVE slope\n");
		while(p1col < p2col){
			if(error >= 0){
				inPixels[p1row*numColumns+p1col] = 0;
				error += 2*dy-2*dx;
				p1row++;
			}else{
				inPixels[p1row*numColumns+p1col] = 0;
				error += 2*dy;
			}
			p1col++;	
		}	
	}else if(p1row > p2row){
		printf("NEGATIVE slope\n");
		while(p1col < p2col){
			if(error > 0){
				inPixels[p1row*numColumns+p1col] = 0;
				error += dy;
				p1row--;
			}else{
				inPixels[p1row*numColumns+p1col] = 0;
				error -= dy - dx;
			}
			p1col++;	
		}	
	}else if(p1col == p2col){
		printf("ZERO slope\n");
		while(p1row < p2row){
			if(error >= 0){
				inPixels[p1row*numColumns+p1col] = 0;
				error += 2*dy-2*dx;
				p1row++;
			}else{
				inPixels[p1row*numColumns+p1col] = 0;
				error += 2*dy;
			}
			p1col++;	
		}
	}
	//printf("finished serial LINE draw\n");
}

int synthesize(const char** header, const int* pixels, int numRows, int numCols, FILE *out) {
	int i, bounds, endLine = 0;
	for (i = 0; i < 4; i++) {
		fprintf(out, "%s", header[i]);
	}

	bounds = (numRows*numCols);

	for (int i = 0; i < bounds; i++) {
		if (endLine == 17) {
			endLine = 0;
			fprintf(out, "%d\n", (pixels[i]));
		}
		fprintf(out, "%d ", (pixels[i]));
		endLine++;
		//printf("pixels %d\n", pixels[i]);
	}
	fclose(out);
	return 0;
}

int main(int argc, char** argv) {
	int* numRows = (int*)calloc(1, sizeof(int));
	int* numColumns = (int*)calloc(1, sizeof(int));
	int* inPixels;

	char* header[6];
	
	int i, ret;
	
	for (i = 0; i < 6; i++) {
		header[i] = (char *)malloc(300);
	}
	
	if (argc < 2) {
		perror("-e edgeWidth  oldImageFile  newImageFile\n-c circleCenterRow circleCenterCol radius  oldImageFile  newImageFile\n-l  p1row  p1col  p2row  p2col  oldImageFile  newImageFile\n");
		exit(3);
	}

	if (strcmp(argv[1],"-e") == 0) {
		if (strcmp(argv[2], "-c") == 0) {
			ret = doCircleEdge(numRows, numColumns, inPixels, header, argv, ret, argc);
		}
		else {
			ret = doEdge(numRows, numColumns, inPixels, header, argv, ret, argc);
		}
	}
	else if (strcmp(argv[1],"-c") == 0) {
		if (strcmp(argv[2], "-e") == 0) {
			ret = doCircleEdge(numRows, numColumns, inPixels, header, argv, ret, argc);
		}
		else {
			ret = doCircle(numRows, numColumns, inPixels, header, argv, ret, argc);
		}
	}
	else if (strcmp(argv[1], "-ce") == 0) {
		if (argc < 8) {
			perror("Wrong usage! -ce circleCenterRow circleCenterCol radius edgeWidth oldImageFile  newImageFile\n\n");
			exit(4);
		}

		//printf("the command is %s %s %s %s %s %s %s\n\n", argv[1], argv[2], argv[3], argv[4], argv[5], argv[6],argv[7]);

		int centerRow, centerCol, radius, circleDraw;
		int edgeWidth, edgeDraw;

		sscanf(argv[2], "%d", &edgeWidth);
		sscanf(argv[2], "%d", &centerRow);
		sscanf(argv[3], "%d", &centerCol);
		sscanf(argv[4], "%d", &radius);
		sscanf(argv[5], "%d", &edgeWidth);

		FILE* file = fopen(argv[6], "r");
		FILE* outFile = fopen(argv[7], "w+");
		char* fileName = (char*)malloc(600);
		strcat(fileName, "serial_");
		strcat(fileName, argv[6]);
		FILE* serialOut = fopen(fileName, "w+");

		if (file == NULL) {
			perror("Sorry couldnt find file\n");
			exit(2);
		}

		inPixels = (int*)pgmRead(header, numRows, numColumns, file);

		circleDraw = pgmDrawCircle(inPixels, *numRows, *numColumns, centerRow, centerCol, radius, header);
		edgeDraw = pgmDrawEdge(inPixels, *numRows, *numColumns, edgeWidth, header);

		ret = pgmWrite((const char**)header, (const int*)inPixels, *numRows, *numColumns, outFile);

		fclose(file);
	}
	else if (strcmp(argv[1], "-sy") == 0) {
		char* magicNumber = (char*)malloc(100);
		char* bounds = (char*)malloc(100);
		char* bounds2 = (char*)malloc(100);
		char* comments = (char*)malloc(100);
		char* comments2 = (char*)malloc(100);
		char* highestPix = (char*)malloc(100);

		FILE* unEven = fopen("synthetic_Image1000x10000.ascii.pgm","w+");
		FILE* even = fopen("synthetic_Image1000x1000.ascii.pgm", "w+");

		strcat(magicNumber, "P2\n");
		strcat(bounds, "1000 10000\n");
		strcat(bounds2, "1000 1000\n");
		strcat(comments, "#this is a synthesized image by halim\n");
		strcat(highestPix, "255\n");

		header[0] = magicNumber;
		header[1] = comments;
		header[2] = bounds;
		header[3] = highestPix;

		int i;
		inPixels = (int*)malloc(1000 * 10000 * sizeof(int));
		for (i = 0; i < (1000 * 10000); i++) {
			inPixels[i] = 255;
			//printf("the pixel value is %d\n", inPixels[i]);
		}
		ret = synthesize((const char**)header, (const int*)inPixels, 1000, 10000, unEven);
		//printf("finished writing to file1\n");
		inPixels = (int*)malloc(1000 * 1000 * sizeof(int));
		for (i = 0; i < (1000 * 1000); i++) {
			inPixels[i] = 255;
		}
		header[2] = bounds2;
		ret = synthesize((const char**)header, (const int*)inPixels, 1000, 1000, even);
		//printf("finished writing to file2\n");
	}
	else if (strcmp(argv[1], "-d") == 0) {
		int* outPixels;
		ret = doEdgeDetect(numRows,numColumns,inPixels,header,argv,ret,argc);
	}
	else if (strcmp(argv[1], "-g") == 0) {
		int* outPixels;
		ret = doGaussBlur(numRows, numColumns, inPixels, header, argv, ret, argc);
	}
	else if (strcmp(argv[1], "-z") == 0) {
		int* outPixels;
		ret = doZSThinning(numRows, numColumns, inPixels, header, argv, ret, argc);
	}
	else if (strcmp(argv[1], "-m") == 0) {
		int* outPixels;
		ret = doMedianFilter(numRows, numColumns, inPixels, header, argv, ret, argc);
	}
	else if (strcmp(argv[1], "-C") == 0) {
		int* outPixels;
		ret = doCountourMap(numRows, numColumns, inPixels, header, argv, ret, argc);
	}
	else if (strcmp(argv[1], "-h") == 0) {
		int* outPixels;
		ret = doGuoHallThinning(numRows, numColumns, inPixels, header, argv, ret, argc);
	}
	else if (strcmp(argv[1], "-l") >= 0) {
		ret = doLine(numRows, numColumns, inPixels, header, argv, ret, argc);
	}
	else {
		printf("other\n");
	}

	return EXIT_SUCCESS;
}
