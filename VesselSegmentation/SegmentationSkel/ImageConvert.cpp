#include <assert.h>
#include <math.h>
#include <FL/Fl.H>
#include <FL/fl_types.H>
#include <FL/Fl_Image.H>
#include <FL/Fl_Shared_Image.H>
#include "ImageConvert.h" // not in ImageLib
#include "ImageLib/FileIO.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define PI 3.14159265358979323846

// Convert Fl_Image to CFloatImage.
bool convertImage(const Fl_Image *image, CFloatImage &convertedImage) {
	if (image == NULL) {
		return false;
	}

	// Let's not handle indexed color images.
	if (image->count() != 1) {
		return false;
	}

	int w = image->w();
	int h = image->h();
	int d = image->d();

	// Get the image data.
	const char *const *data = image->data();

	int index = 0;

	for (int y=0; y<h; y++) {
		for (int x=0; x<w; x++) {
			if (d < 3) {
				// If there are fewer than 3 channels, just use the
				// first one for all colors.
				convertedImage.Pixel(x,y,0) = ((uchar) data[0][index]) / 255.0f;
				convertedImage.Pixel(x,y,1) = ((uchar) data[0][index]) / 255.0f;
				convertedImage.Pixel(x,y,2) = ((uchar) data[0][index]) / 255.0f;
			}
			else {
				// Otherwise, use the first 3.
				convertedImage.Pixel(x,y,0) = ((uchar) data[0][index]) / 255.0f;
				convertedImage.Pixel(x,y,1) = ((uchar) data[0][index+1]) / 255.0f;
				convertedImage.Pixel(x,y,2) = ((uchar) data[0][index+2]) / 255.0f;
			}

			index += d;
		}
	}
	
	return true;
}

// Convert CFloatImage to CByteImage.
void convertToByteImage(CFloatImage &floatImage, CByteImage &byteImage) {
	CShape sh = floatImage.Shape();

    assert(floatImage.Shape().nBands == byteImage.Shape().nBands);
	for (int y=0; y<sh.height; y++) {
		for (int x=0; x<sh.width; x++) {
			for (int c=0; c<sh.nBands; c++) {
				float value = floor(255*floatImage.Pixel(x,y,c) + 0.5f);

				if (value < byteImage.MinVal()) {
					value = byteImage.MinVal();
				}
				else if (value > byteImage.MaxVal()) {
					value = byteImage.MaxVal();
				}

				// We have to flip the image and reverse the color
				// channels to get it to come out right.  How silly!
				byteImage.Pixel(x,sh.height-y-1,sh.nBands-c-1) = (uchar) value;
			}
		}
	}
}


void convertToGrayImage(const CFloatImage &image, const CFloatImage &grayImage) {
	
	//grayImage = ConvertToGray(floatImage);
	//return grayImage;
}

Fl_RGB_Image *convertToFLRGBImage( CByteImage & image) {
	CShape sh = image.Shape();
	const int imgSize = sh.width * sh.height *  sh.nBands;
	uchar *pixels = new uchar[imgSize];
	for (int b = 0; b < sh.nBands; b++) {
		for (int h = 0; h < sh.height; h++) {
			for (int w = 0; w < sh.width; w++) {
				pixels[b*sh.width *  sh.height + h *sh.width + w] = image.Pixel(w, h, b);
			}
		}
	}
	

	return new Fl_RGB_Image(pixels, sh.width, sh.height, sh.nBands);
}
int writePNG(CByteImage & image, const char *filename) {
	WriteFile(image, "tempimg.tga");
	int x, y, n;
	unsigned char *data = stbi_load("tempimg.tga", &x, &y, &n, 0);
	int result = stbi_write_png(filename,x, y, n, data, x);
	stbi_image_free(data);
	return result;
}