#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#include "ImageLib/ImageLib.h"
#include "ImageDatabase.h"
#include <FL/Fl.H>
#include <vector>;



class Fl_Image;

float structSquare[8][2] = { { -1, -1 },{ -1, 0 },{ -1, 1 },{ 0,1 },{ 1,1 },{ 1,0 },{ 1,-1 },{ 0,-1 } };

float structureElement[12][15][2] = { { { -7,0 },{ -6,0 },{ -5,0 },{ -4,0 },{ -3,0 },{ -2,0 },{ -1,0 },{ 0,0 },{ 1,0 },{ 2,0 },{ 3,0 },{ 4,0 },{ 5,0 },{ 6,0 },{ 7,0 } },{ { -7,-2 },{ -6,-2 },{ -5,-1 },{ -4,-1 },{ -3,-1 },{ -2,-1 },{ -1,0 },{ 0,0 },{ 1,0 },{ 2,1 },{ 3,1 },{ 4,1 },{ 5,1 },{ 6,2 },{ 7,2 } },{ { -6,-4 },{ -5,-3 },{ -4,-2 },{ -3,-2 },{ -3,-2 },{ -2,-1 },{ -1,0 },{ 0,0 },{ 1,0 },{ 2,1 },{ 3,2 },{ 3,2 },{ 4,2 },{ 5,3 },{ 6,4 } },{ { -5,-5 },{ -4,-4 },{ -4,-4 },{ -3,-3 },{ -2,-2 },{ -1,-1 },{ -1,-1 },{ 0,0 },{ 1,1 },{ 1,1 },{ 2,2 },{ 3,3 },{ 4,4 },{ 4,4 },{ 5,5 } },{ { -4,-6 },{ -3,-5 },{ -2,-4 },{ -2,-3 },{ -2,-3 },{ -1,-2 },{ 0,-1 },{ 0,0 },{ 0,1 },{ 1,2 },{ 2,3 },{ 2,3 },{ 2,4 },{ 3,5 },{ 4,6 } },{ { -2,-7 },{ -2,-6 },{ -1,-5 },{ -1,-4 },{ -1,-3 },{ -1,-2 },{ 0,-1 },{ 0,0 },{ 0,1 },{ 1,2 },{ 1,3 },{ 1,4 },{ 1,5 },{ 2,6 },{ 2,7 } },{ { 0,-7 },{ 0,-6 },{ 0,-5 },{ 0,-4 },{ 0,-3 },{ 0,-2 },{ 0,-1 },{ 0,0 },{ 0,1 },{ 0,2 },{ 0,3 },{ 0,4 },{ 0,5 },{ 0,6 },{ 0,7 } },{ { 2,-7 },{ 2,-6 },{ 1,-5 },{ 1,-4 },{ 1,-3 },{ 1,-2 },{ 0,-1 },{ 0,0 },{ 0,1 },{ -1,2 },{ -1,3 },{ -1,4 },{ -1,5 },{ -2,6 },{ -2,7 } },{ { 4,-6 },{ 3,-5 },{ 2,-4 },{ 2,-3 },{ 2,-3 },{ 1,-2 },{ 0,-1 },{ 0,0 },{ 0,1 },{ -1,2 },{ -2,3 },{ -2,3 },{ -2,4 },{ -3,5 },{ -4,6 } },{ { 5,-5 },{ 4,-4 },{ 4,-4 },{ 3,-3 },{ 2,-2 },{ 1,-1 },{ 1,-1 },{ 0,0 },{ -1,1 },{ -1,1 },{ -2,2 },{ -3,3 },{ -4,4 },{ -4,4 },{ -5,5 } },{ { 6,-4 },{ 5,-3 },{ 4,-2 },{ 3,-2 },{ 3,-2 },{ 2,-1 },{ 1,0 },{ 0,0 },{ -1,0 },{ -2,1 },{ -3,2 },{ -3,2 },{ -4,2 },{ -5,3 },{ -6,4 } },{ { 7,-2 },{ 6,-2 },{ 5,-1 },{ 4,-1 },{ 3,-1 },{ 2,-1 },{ 1,0 },{ 0,0 },{ -1,0 },{ -2,1 },{ -3,1 },{ -4,1 },{ -5,1 },{ -6,2 },{ -7,2 } } };



//float L1[15][2] = { {-7, 0},{-6 ,0},{ -5, 0 },{ -4, 0 },{ -3, 0 },{ -2, 0 },{ -1, 0 },{ 0, 0 },{ 1, 0 },{ 2, 0 },{ 3, 0 },{ 4, 0 },{ 5, 0 },{ 6, 0 },{ 7, 0 } };
//float L2[15][2] = { { -7, -2 },{ -6, -2 },{ -5, -1 },{ -4, -1 },{ -3, -1 },{ -2, -1 },{ -1,0 },{ 0, 0 },{ 1, 0 },{ 2, 1 },{ 3, 1 },{ 4, 1 },{ 5, 1 },{ 6, 2 },{ 7, 2 } };
//float L3[15][3] = { { -6,-4 },{ -5,-3 },{ -4,-2 },{ -3,-2 },{ -3,-2 },{ -2,-1 },{ -1,0 },{ 0,0 },{ 1,0 },{ 2,1 },{ 3,2 },{ 3,2 },{ 4,2 },{ 5,3 },{ 6,4 } };
//float L4[15][4] = { { -5,-5 },{ -4,-4 },{ -4,-4 },{ -3,-3 },{ -2,-2 },{ -1,-1 },{ -1,-1 },{ 0,0 },{ 1,1 },{ 1,1 },{ 2,2 },{ 3,3 },{ 4,4 },{ 4,4 },{ 5,5 } };

void padImageForErosion(CFloatImage &image, CFloatImage &paddedImage, int structSize);
void padImageForDilation(CFloatImage &image, CFloatImage &paddedImage, int structSize);
CFloatImage ErosionStructSquare(CFloatImage &grayImage, int structSize);
CFloatImage Erosion(CFloatImage &grayImage, int structType);
CFloatImage Dilation(CFloatImage grayImage, int structType);
CFloatImage Opening(CFloatImage image, int structType);
CFloatImage Closing(CFloatImage image, int structType);
CFloatImage GeodesicOpening(CFloatImage marker, CFloatImage mask);
CFloatImage GeodesicClosing(CFloatImage marker, CFloatImage image);
CFloatImage GeodesicClosing(CFloatImage marker, CFloatImage image);
CFloatImage vesselSegmentation(CFloatImage &image);
#endif
