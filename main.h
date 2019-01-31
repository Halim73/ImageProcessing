#ifndef MAIN_H
#define MAIN_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "pgmUtility.h"
float _distance(int p1[], int p2[]);
int synthesize(const char** header, const int* pixels, int numRows, int numCols, FILE *out);
int doEdgeDetect(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc);
int doGaussBlur(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc);
int doGuoHallThinning(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc);
int doZSThinning(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc);
int doCountourMap(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc);
int doMedianFilter(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc);
int serialLineDraw(int* inPixels,int numRows,int numColumns, int p1row,int p1col,int p2row,int p2col);
int serialEdgeDraw(int* pixels, int numRows, int numCols, int edgeWidth);
int serialDrawCircle(int* pixels, int numCol, int numRows, int centerRow, int centerCol, int radius);
int doEdge(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc);
int doCircle(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc);
int doLine(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc);
int doCircleEdge(int* numRows, int* numColumns, int* inPixels, char** header, char** argv, int ret, int argc);
#endif
