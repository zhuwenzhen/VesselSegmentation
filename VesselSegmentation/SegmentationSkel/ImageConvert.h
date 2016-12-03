#ifndef IMAGECONVERT_H
#define IMAGECONVERT_H

#include "ImageLib/ImageLib.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>

using namespace std;


class Fl_Image;

//5x5 Gaussian
const double gaussian5x5[25] = {0.003663, 0.014652,  0.025641,  0.014652,  0.003663, 
                                0.014652, 0.0586081, 0.0952381, 0.0586081, 0.014652, 
                                0.025641, 0.0952381, 0.150183,  0.0952381, 0.025641, 
                                0.014652, 0.0586081, 0.0952381, 0.0586081, 0.014652, 
                                0.003663, 0.014652,  0.025641,  0.014652,  0.003663 };


//7x7 Gaussian
const double gaussian7x7[49] = {0.000896861, 0.003587444, 0.006278027, 0.00896861,  0.006278027, 0.003587444, 0.000896861,
                                0.003587444, 0.010762332, 0.023318386, 0.029596413, 0.023318386, 0.010762332, 0.003587444, 
                                0.006278027, 0.023318386, 0.049327354, 0.06367713,  0.049327354, 0.023318386, 0.006278027,
                                0.00896861,  0.029596413, 0.06367713,  0.08161435,  0.06367713,  0.029596413, 0.00896861,  
                                0.006278027, 0.023318386, 0.049327354, 0.06367713,  0.049327354, 0.023318386, 0.006278027,
                                0.003587444, 0.010762332, 0.023318386, 0.029596413, 0.023318386, 0.010762332, 0.003587444,
                                0.000896861, 0.003587444, 0.006278027, 0.00896861,  0.006278027, 0.003587444, 0.000896861};



// Convert Fl_Image to CFloatImage.
bool convertImage(const Fl_Image *image, CFloatImage &convertedImage);

// Convert CFloatImage to CByteImage.
void convertToByteImage(CFloatImage &floatImage, CByteImage &byteImage);

// Compute SSD distance between two vectors.
double distanceSSD(const vector<double> &v1, const vector<double> &v2);


#endif
