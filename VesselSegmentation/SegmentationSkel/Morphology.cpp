#include "Morphology.h"
#include <assert.h>
#include <FL/Fl.H>
#include <FL/Fl_Image.H>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <functional>
#include <iostream>
#include "ImageLib/Convolve.h"

using namespace std;

//void testPixel(CFloatImage &image) {
//	for (int i = 0; i < 1e6; i++) {
//		image.Pixel()
//	}
//}

// 2. how to make the library works properly? 
// 3. how to geodesicDilation? 

// Because Erosion takes the minimum, so we pad 1
CFloatImage padImageForErosion(CFloatImage &image, int structSize) {
	CShape sh = image.Shape();
	int boarderSize = (structSize - 1) / 2;

	// w, h are the dimensions of the paddedImage
	int w = sh.width + 2 * boarderSize;
	int h = sh.height + 2 * boarderSize;
	CFloatImage paddedImage(w, h, 1);
	paddedImage.ClearPixels();
	// initialize it with all 1

	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			paddedImage.Pixel(x, y, 0) = 1;
		}
	}

	// copy the image to the "center" of paddedImage
	for (int y = 0; y < sh.height; y++) {
		for (int x = 0; x < sh.width; x++) {
			paddedImage.Pixel(x + boarderSize, y + boarderSize, 0) = image.Pixel(x, y, 0);
		}
	}
	return paddedImage;
}

// Because Dilation takes the maximum, so we pad 0
CFloatImage padImageForDilation(CFloatImage &image, int structSize) {
	CShape sh = image.Shape();
	int boarderSize = (structSize - 1) / 2;

	// w, h are the dimensions of the paddedImage
	int w = sh.width + 2 * boarderSize;
	int h = sh.height + 2 * boarderSize;
	CFloatImage paddedImage(w, h, 1);

	// initialize it with all zeros
	paddedImage.ClearPixels();

	// copy the image to the "center" of paddedImage
	for (int y = 0; y < sh.height; y++) {
		for (int x = 0; x < sh.width; x++) {
			paddedImage.Pixel(x + boarderSize, y + boarderSize, 0) = image.Pixel(x, y, 0);
		}
	}
	return paddedImage;
}

// paper used struct square to do erosion for geodedicDilation definition
CFloatImage ErosionStructSquare(CFloatImage &grayImage, int structSize) {
	// in the paper S_1 is defined using erosion (unit neighborhood) so d = 3
	int windowSize = structSize;	
	CShape sh = grayImage.Shape();
	int w = sh.width + 2; int h = sh.height + 2;
	CFloatImage paddedImage(w, h, 1);
	paddedImage = padImageForErosion(grayImage, 3);
	CFloatImage resultImage(sh);

	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; x < sh.height; y++) {
			int centerX = x + 1;
			int centerY = y + 1;
			vector<float> pixelValues;
			for (int i = 0; i < 8; i++) {
				pixelValues.push_back(paddedImage.Pixel(centerX + structSquare[i][0], centerY + structSquare[i][1], 0));
			}
			// pick the minimum
			float p_min = *min_element(pixelValues.begin(), pixelValues.end());
			resultImage.Pixel(x, y, 0) = p_min;
		}
	}
	return resultImage;
}

CFloatImage DilationStructSquare(CFloatImage &grayImage) {
	// in the paper S_1 is defined using erosion (unit neighborhood) so d = 3
	
	CShape sh = grayImage.Shape();
	int w = sh.width + 2; int h = sh.height + 2;

	CFloatImage paddedImage(w, h, 1);
	paddedImage = padImageForErosion(grayImage, 3);
	CFloatImage resultImage(sh);

	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; y < sh.height; y++) {
			int centerX = x + 1;
			int centerY = y + 1;
			vector<float> pixelValues;

			for (int i = 0; i < 9; i++) {
				pixelValues.push_back(paddedImage.Pixel(centerX + structSquare[i][0], centerY + structSquare[i][1], 0));
			}
			// pick the maximum
			float p_max = *max_element(pixelValues.begin(), pixelValues.end());
			resultImage.Pixel(x, y, 0) = p_max;
		}
	}
	return resultImage;
}

// in the paper, scale are 15 
// rotation Erosion
CFloatImage Erosion(CFloatImage &grayImage, int structType) {
	int scale = 15;
	size_t windowSize = 15;

	CFloatImage paddedImage;
	paddedImage = padImageForErosion(grayImage,  scale);

	CShape sh = paddedImage.Shape();
	CShape grayImage_sh = grayImage.Shape();
	CFloatImage resultImage(grayImage_sh);

	int w1 = grayImage_sh.width; int h1 = grayImage_sh.height;

	for (int x = 0; x < w1; x++) {
		for (int y = 0; y < h1; y++) {
			int centerX = x + 7;
			int centerY = y + 7;
			vector<float> pixelValues;
			for (int i = 0; i < 15; i++) {
				// centerx = 7
				// Pixel(10, 579+6, 0)
				float pixelValue = paddedImage.Pixel(centerX + structureElement[structType][i][0], centerY + structureElement[structType][i][1], 0);
				pixelValues.push_back(pixelValue);
			}
			// pick the minimum
			float p_min = *min_element(pixelValues.begin(), pixelValues.end());
			resultImage.Pixel(x, y, 0) = p_min;
		}
	}
	return resultImage;
}
CFloatImage Dilation(CFloatImage &grayImage, int structType) {
	int scale = 15;
	size_t windowSize = 15;

	CFloatImage paddedImage;
	paddedImage = padImageForErosion(grayImage, scale);

	CShape sh = paddedImage.Shape();
	CShape grayImage_sh = grayImage.Shape();
	CFloatImage resultImage(grayImage_sh);

	int w1 = grayImage_sh.width; int h1 = grayImage_sh.height;

	for (int x = 0; x < w1; x++) {
		for (int y = 0; y < h1; y++) {

			int centerX = x + 7;
			int centerY = y + 7;

			vector<float> pixelValues;

			for (int i = 0; i < 15; i++) {
				// centerx = 7
				// Pixel(10, 579+6, 0)
				float pixelValue = paddedImage.Pixel(centerX + structureElement[structType][i][0], centerY + structureElement[structType][i][1], 0);
				pixelValues.push_back(pixelValue);
			}

			// pick the maximum
			float p_max = *max_element(pixelValues.begin(), pixelValues.end());
			resultImage.Pixel(x, y, 0) = p_max;
		}
	}
	return resultImage;
}
CFloatImage Opening(CFloatImage image, int structType) {
	return Dilation(Erosion(image, structType), structType);
}
CFloatImage Closing(CFloatImage image, int structType) {
	return Erosion(Dilation(image, structType), structType);
}

// test this with Mathematica result
CFloatImage GeodesicDilation(CFloatImage marker, CFloatImage mask, int size) {
	// marker: Sm; f
	// mask: original image; g pg. 676
	int marker_w = marker.Shape().width;
	int marker_h = marker.Shape().height;

	CFloatImage resultImage(marker.Shape());

	// a neighborhood C of unit radius	
	if (size == 0) {
		return marker;
	}
	if (size == 1) {
		// paper: a neighborhood C of unit radius
		cout << "Entering base case size = 1" << endl;
		CFloatImage DilationImage = DilationStructSquare(marker);
		for (int x = 0; x < marker_w; x++) {
			for (int y = 0; y < marker_h; y++) {
				//In the paper, geodesic dilation used normal dilation in the base case;
				// mask[x, y] 
				// S1(M0) = inf({Dilation, S(M0)})
				resultImage.Pixel(x, y, 0) = min(DilationImage.Pixel(x, y, 0), mask.Pixel(x, y, 0));
			}
		} 
		// cout << resultImage.Pixel(100, 100, 0);
		return resultImage;
	}
	// recursion : very very unconfident 
	// defined using page 677 equation 9.6-15
	return GeodesicDilation(GeodesicDilation(marker, mask, size - 1), mask, 1);
 }
bool compareCFloatImageEqual(CFloatImage & img1, CFloatImage & img2) {
	CShape sh1 = img1.Shape();
	CShape sh2 = img2.Shape();
	float thresholdError = 0.001;
	if (sh1.height != sh2.height || sh1.width != sh2.width || sh1.nBands != sh2.nBands) {
		return false;
	}
	else {
		for (int x = 0; x < sh1.width; x++) {
			for (int y = 0; y < sh2.height; y++) {
				//No bands!
				if (abs(img1.Pixel(x, y, 0) - img2.Pixel(x, y, 0)) > thresholdError) {
					return false;
				}				
			}
		}
	}
	return true;
}
// iterate untill it doesn't change 
// different from Mathematica implementation
CFloatImage GeodesicOpening(CFloatImage marker, CFloatImage mask) {
	vector<CFloatImage> geodesicDilationList; // equaivalent to fixedPointList in mathematica
	CShape sh = marker.Shape();
	CFloatImage resultImage(sh);
	// apply geodesic dilation until it doesn't change
	CFloatImage baseDilatedImg = GeodesicDilation(marker, mask, 0);
	geodesicDilationList.push_back(baseDilatedImg);
	int i = 1;
	while (true) {
		CFloatImage dilatedImg = GeodesicDilation(marker, mask, i);
		if (compareCFloatImageEqual(geodesicDilationList[i - 1], dilatedImg)) {
			break;
		}
		else {
			geodesicDilationList.push_back(dilatedImg);
		}
		if (i > 10) {
			cout << "GeodesicOpening::Does't converge after 11 iterations" << endl;
			break;
		}
		i++;
	}
	
	// pick the maximum pixel-wise
	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; y < sh.width; y++) {
			float temp = 0;
			for (auto img : geodesicDilationList) {
				if (temp < img.Pixel(x, y, 0)) {
					temp = img.Pixel(x, y, 0);
				}
			}
			resultImage.Pixel(x, y, 0) = temp;
		}
	}
	return resultImage;
}

CFloatImage GeodesicClosing(CFloatImage marker, CFloatImage image) {
	CShape sh = marker.Shape();
	CFloatImage resultImage(sh);

	// Nmax is the maximum pixel value of the original image S == image
	float Nmax = 0;
	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; y < sh.height; y++) {
			if (Nmax < image.Pixel(x, y, 0)) {
				Nmax = image.Pixel(x, y, 0);
			}
		}
	}

	// marker = N_max - S_m (marker)
	// markerImage is the marker used for geodesic closing's opening
	CFloatImage markerImage(sh);
	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; y < sh.height; y++) {
			markerImage.Pixel(x, y, 0) = Nmax - marker.Pixel(x, y, 0);
		}
	}

	CFloatImage objectImage(sh);
	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; y < sh.height; y++) {
			objectImage.Pixel(x, y, 0) = Nmax - image.Pixel(x, y, 0);
		}
	}

	CFloatImage temp = GeodesicOpening(markerImage, objectImage);
	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; y < sh.height; y++) {
			resultImage.Pixel(x, y, 0) = Nmax - temp.Pixel(x, y, 0);
		}
	}

	return resultImage;
}

CFloatImage LaplacianFilter(CFloatImage &image) {
	CShape kernel_shape(3, 3, 1);
	CFloatImage laplacianKernel(kernel_shape);

	laplacianKernel.Pixel(0, 0, 0) = 0;
	laplacianKernel.Pixel(1, 0, 0) = 1;
	laplacianKernel.Pixel(2, 0, 0) = 0;
	laplacianKernel.Pixel(0, 1, 0) = 1;
	laplacianKernel.Pixel(1, 1, 0) = -4;
	laplacianKernel.Pixel(2, 1, 0) = 1;
	laplacianKernel.Pixel(0, 2, 0) = 0;
	laplacianKernel.Pixel(1, 2, 0) = 1;
	laplacianKernel.Pixel(2, 2, 0) = 0;

	CFloatImage resultImage(image.Shape());
	Convolve(image, resultImage, laplacianKernel);
	return resultImage;
}

CFloatImage GaussianFilter(CFloatImage &image) {
	CFloatImage kernel(15, 15, 1);
	for (int i = 0; i < 15; i++) {
		for (int j = 0; j < 15; j++) {
			kernel.Pixel(i, j, 0) = gaussian15x15[i][j];
		}
	}
	CFloatImage resultImage(image.Shape());
	Convolve(image, resultImage, kernel);
	return resultImage;
}

CFloatImage vesselSegmentation(CFloatImage &image) {
	// CFloatImage resultImage(sh.width, sh.height, 0); // Initialize a Image with  
	// step 1: S_op
	cout << "Entering Step 1: Computing S_op" << endl;
	CShape sh = image.Shape();
	int w = sh.width; int h = sh.height;

	CFloatImage maxOpening(w, h, 1);
	vector<CFloatImage> openingImageList;

	for (int i = 0; i < 12; i++) {
		CFloatImage tempImg = Opening(image, i);
		openingImageList.push_back(tempImg);
	}

	// pick the maximum pixel-wise
	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; y < sh.width; y++) {
			float temp = 0;
			for (auto img : openingImageList) {
				if (temp < img.Pixel(x, y, 0)) {
					temp = img.Pixel(x, y, 0);
				}
			}
			maxOpening.Pixel(x, y, 0) = temp;
		}
	}
	// apply geodesic opening with marker = S_0 (origimal image)
	CFloatImage Sop(w, h, 1);
	Sop = GeodesicOpening(image, maxOpening);
	return Sop;

	// Step 2: Compute sum of top hats
	// reduces small bright noise and improves the contrast of all linear parts. 
	// Vessels could be manually segmented with a simple threshold on S_sum
	// vector <CFloatImage> topHatImageList;
	cout << "Entering Step 2: Top hats" << endl;
	CFloatImage S_sum;

	for (int i = 0; i < 12; i++) {
		CFloatImage topHatImage(w, h, 1);
		for (int x = 0; x < w; x++) {
			for (int y = 0; y < h; y++) {
				topHatImage.Pixel(x, y, 0) = Sop.Pixel(x, y, 0) - openingImageList[i].Pixel(x, y, 0);
				S_sum.Pixel(x, y, 0) += topHatImage.Pixel(x, y, 0);
			}
		}
	}

	// Step 3: Denoise using Laplacian curvature on the image 
	// filtered by Gaussian Filter with kernel = GaussianMatrix(r = 7, sigma = 7/4)
	cout << "Entering Step 3: Laplacian Filter" << endl;
	CFloatImage S_lap = LaplacianFilter(GaussianFilter(S_sum));

	// Step 4: Final Result
	// 4.1 Compute S_1
	cout << "Entering Step 4.1: S_1" << endl;
	vector<CFloatImage> openingList;
	for (int i = 0; i < 12; i++) {
		openingList.push_back(Opening(S_lap, i));
	}
	CFloatImage maximumImage;
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			float temp = -10e10;
			for (CFloatImage & it : openingList) {
				if (temp < it.Pixel(x, y, 0)) {
					temp = it.Pixel(x, y, 0);					
				}
			}
			maximumImage.Pixel(x, y, 0) = temp;
		}
	}

	// marker S_m = S_lap, mask = maximum of the opening result
	CFloatImage S1 = GeodesicOpening(S_lap, maximumImage);

	// S2
	cout << "Entering Step 4.2: S_2" << endl;
	vector<CFloatImage> closingList;
	for (int i = 0; i < 12; i++) {
		closingList.push_back(Closing(S1, i));
	}


	CFloatImage minimumImage;
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			float temp = 10e10;
			for (CFloatImage & it : closingList) {
				if (temp > it.Pixel(x, y, 0)) {
					temp = it.Pixel(x, y, 0);
				}
			}
			minimumImage.Pixel(x, y, 0) = temp;
		}
	}

	// geodesic closing on min with marker = S1
	CFloatImage S2 = GeodesicClosing(S1, minimumImage);


	cout << "Entering Step 4.3: maximum" << endl;
	// Opening
	vector<CFloatImage> doubleOpeningList;
	for (int i = 0; i < 12; i++) {
		doubleOpeningList.push_back(Opening(S2, i));
	}

	const double threshold = 0.00392157; // 1 in 255

	CFloatImage doubleMaximumImage;
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			float temp = -10e10;
			for (CFloatImage & it : doubleOpeningList) {
				if (temp < it.Pixel(x, y, 0)) {
					temp = it.Pixel(x, y, 0);
				}
			}			
			if (temp > threshold) {
				doubleMaximumImage.Pixel(x, y, 0) = 1;
			}
			else {
				doubleMaximumImage.Pixel(x, y, 0) = 0;
			}
		}
	}
	// return doubleMaximumImage;
}

CFloatImage Erosion2(CFloatImage &grayImage, int structType) {
	int scale = 31;

	CFloatImage paddedImage;
	paddedImage = padImageForErosion(grayImage, scale);

	CShape sh = paddedImage.Shape();
	CShape grayImage_sh = grayImage.Shape();
	CFloatImage resultImage(grayImage_sh);

	int w1 = grayImage_sh.width; int h1 = grayImage_sh.height;

	for (int x = 0; x < w1; x++) {
		for (int y = 0; y < h1; y++) {
			int centerX = x + 15;
			int centerY = y + 15;
			vector<float> pixelValues;
			for (int i = 0; i < 31; i++) {
				// centerx = 7
				// Pixel(10, 579+6, 0)
				float pixelValue = paddedImage.Pixel(centerX + structureElement2[structType][i][0], centerY + structureElement2[structType][i][1], 0);
				pixelValues.push_back(pixelValue);
			}
			// pick the minimum
			float p_min = *min_element(pixelValues.begin(), pixelValues.end());
			resultImage.Pixel(x, y, 0) = p_min;
		}
	}
	return resultImage;
}

CFloatImage Dilation2(CFloatImage &grayImage, int structType) {
	int scale = 31;

	CFloatImage paddedImage;
	paddedImage = padImageForDilation(grayImage, scale);

	CShape sh = paddedImage.Shape();
	CShape grayImage_sh = grayImage.Shape();
	CFloatImage resultImage(grayImage_sh);

	int w1 = grayImage_sh.width; int h1 = grayImage_sh.height;

	for (int x = 0; x < w1; x++) {
		for (int y = 0; y < h1; y++) {
			int centerX = x + 15;
			int centerY = y + 15;
			vector<float> pixelValues;
			for (int i = 0; i < 31; i++) {
				float pixelValue = paddedImage.Pixel(centerX + structureElement2[structType][i][0], centerY + structureElement2[structType][i][1], 0);
				pixelValues.push_back(pixelValue);
			}
			// pick the maximum
			float p_max = *max_element(pixelValues.begin(), pixelValues.end());
			resultImage.Pixel(x, y, 0) = p_max;
		}
	}
	return resultImage;
}