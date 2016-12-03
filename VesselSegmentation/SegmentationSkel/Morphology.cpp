#include "Morphology.h"
#include <assert.h>
#include <FL/Fl.H>
#include <FL/Fl_Image.H>
#include "ImageLib/FileIO.h"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <functional>
#include <iostream>

using namespace std;

// Because Erosion takes the minimum, so we pad 1
void padImageForErosion(CFloatImage &image, CFloatImage &paddedImage, int structSize) {
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
}

// Because Dilation takes the maximum, so we pad 0
void padImageForDilation(CFloatImage &image, CFloatImage &paddedImage, int structSize) {
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
}

// paper used struct square to do erosion for geodedicDilation definition
CFloatImage ErosionStructSquare(CFloatImage &grayImage, int structSize) {
	// in the paper S_1 is defined using erosion (unit neighborhood) so d = 3

	int windowSize = structSize;
	
	CShape sh = grayImage.Shape();
	int w = sh.width + 2; int h = sh.height + 2;
	CFloatImage paddedImage(w, h, 1);
	padImageForErosion(grayImage, paddedImage, 3);

	CFloatImage resultImage(sh);

	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; x < sh.height; y++) {
			int centerX = x + 1;
			int centerY = y + 1;
			vector<float> pixelValues;

			for (int i = 0; i < 8; i++) {
				pixelValues.push_back(paddedImage.Pixel(centerX + structSquare[i][1], centerY + structSquare[i][2], 0));
			}
			// pick the minimum
			float p_min = *min_element(pixelValues.begin(), pixelValues.end());
			resultImage.Pixel(x, y, 0) = p_min;
		}
	}
	return resultImage;
}

// in the paper, scale are 15 
CFloatImage Erosion(CFloatImage &grayImage, int structType) {

	// [-7, -6, -5, ..., 0, 1, 2, ..., 6, 7]
	int scale = 15;
	size_t windowSize = 15;

	CFloatImage paddedImage;
	padImageForErosion(grayImage, paddedImage, scale);

	// {cos[t], sin[t]}, t = 0, Pi/12, ... 11 Pi/12
	CShape sh = paddedImage.Shape();
	CShape grayImage_sh = grayImage.Shape();
	CFloatImage resultImage(grayImage_sh);

	int w1 = grayImage_sh.width; int h1 = grayImage_sh.height;

	// window 15 * 15 or 31 * 31

	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; y < sh.width; y++) {

			int centerX = x + 7;
			int centerY = y + 7;

			vector<float> pixelValues;
		

			for (int i = 0; i < 15; i++) {
				float pixelValue = grayImage.Pixel(centerX + structureElement[structType][i][1], centerY + structureElement[structType][i][2], 0);
				pixelValues.push_back(pixelValue);
			}

			// pick the minimum
			float p_min = *min_element(pixelValues.begin(), pixelValues.end());

			resultImage.Pixel(x, y, 0) = p_min;

		}
	}
	return resultImage;

}

CFloatImage Dilation(CFloatImage grayImage, int structType) {
	int scale = 15;
	size_t windowSize = 15;

	CFloatImage paddedImage;
	padImageForErosion(grayImage, paddedImage, scale);

	// {cos[t], sin[t]}, t = 0, Pi/12, ... 11 Pi/12
	CShape sh = paddedImage.Shape();
	CShape grayImage_sh = grayImage.Shape();

	int w1 = grayImage_sh.width; int h1 = grayImage_sh.height;

	CFloatImage resultImage(grayImage_sh);

	// window 15 * 15 or 31 * 31

	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; y < sh.width; y++) {

			int centerX = x + 7;
			int centerY = y + 7;

			vector<float> pixelValues;

			for (int i = 0; i < 15; i++) {
				float pixelValue = grayImage.Pixel(centerX + structureElement[structType][i][1], centerY + structureElement[structType][i][2], 0);
				pixelValues.push_back(pixelValue);
			}

			// pick the minimum
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
// wait
//CFloatImage GeodesicDilation(CFloatImage marker, CFloatImage mask) {
//	// marker: Sm
//	// mask: original image
//	int marker_w = marker.Shape().width;
//	int marker_h = marker.Shape().height;
//
//	CFloatImage resultImage(marker.Shape);
//
//	CFloatImage erosionImage;
//
//
//	if (iterationNum == 1) {
//		for (int x = 0; x < marker_w; x++) {
//			for (int y = 0; y < marker_h; y++) {
//				resultImage.Pixel(x, y, 0) = max(resultImage.Pixel(x, y, 0), erosionImage.Pixel(x, y, 0));
//			}
//		}
//
//		return resultImage;
//	}
//	// recursion
//	return GeodesicDilation(GeodesicDilation(marker, mask, iterationNum - 1), mask, 1);
// }

CFloatImage GeodesicOpening(CFloatImage marker, CFloatImage mask) {
	vector<CFloatImage> geodesicDilationList;

	CShape sh = marker.Shape();
	CFloatImage resultImage(sh);
	
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

	float Nmax = 0;
	for (int x = 0; x < sh.width; x++) {
		for (int y = 0; y < sh.height; y++) {
			if (Nmax < image.Pixel(x, y, 0)) {
				Nmax = image.Pixel(x, y, 0);
			}
		}
	}

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

CFloatImage vesselSegmentation(CFloatImage &image) {

	// CFloatImage resultImage(sh.width, sh.height, 0); // Initialize a Image with  
	// step 1: S_op
	CShape sh = image.Shape();
	int w = sh.width; int h = sh.height;

	CFloatImage maxOpening(w, h, 0);
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

	// Step 2: Compute sum of top hats
	// reduces small bright noise and improves the contrast of all linear parts. 
	// Vessels could be manually segmented with a simple threshold on S_sum

	// vector <CFloatImage> topHatImageList;
	CFloatImage S_sum;

	for (int i = 0; i < 12; i++) {
		CFloatImage topHatImage(w, h, 0);
		for (int x = 0; x < w; x++) {
			for (int y = 0; y < h; y++) {
				topHatImage.Pixel(x, y, 0) = Sop.Pixel(x, y, 0) - openingImageList[i].Pixel(x, y, 0);
				S_sum.Pixel(x, y, 0) += topHatImage.Pixel(x, y, 0);
			}
		}
	}


	// Step 3: Denoise using Laplacian curvature on the image 
	// filtered by Gaussian Filter with kernel = GaussianMatrix(r = 7, sigma = 7/4)



	


}





	
