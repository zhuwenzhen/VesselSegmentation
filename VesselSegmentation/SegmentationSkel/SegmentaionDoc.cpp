#include <assert.h>
#include <FL/Fl.H>
#include <FL/Fl_Image.H>
#include <FL/Fl_Shared_Image.H>
#include <FL/fl_ask.H>
#include "ImageConvert.h"
#include "SegmentationUI.h"
#include "SegmentationDoc.h"
#include "Morphology.h"

// Create a new document.
SegmentationDoc::SegmentationDoc() {
	rgbImage = NULL;
	binaryImage = NULL;
	grayImage = NULL;
	ui = NULL;
}

// Load an image file for use as the rgb image.
void SegmentationDoc::load_rgb_image(const char *name) {
	ui->set_images(NULL, NULL, NULL);

	// Delete the current rgb image.
	if (rgbImage != NULL) {
		rgbImage->release();
		rgbImage = NULL;
	}

	// Delete the current gray-scale image.
	if (grayImage != NULL) {
		grayImage->release();
		grayImage = NULL;
	}

	// Delete the current binary image.
	if (binaryImage != NULL) {
		binaryImage->release();
		binaryImage = NULL;
	}

	// Load the image.
	rgbImage = Fl_Shared_Image::get(name);

	if (rgbImage == NULL) {
		fl_alert("couldn't load image file");
	}
	else {
		// Update the UI.
		ui->resize_windows(rgbImage->w(), 0, 0, rgbImage->h());
		ui->set_images(rgbImage, NULL, NULL);
	}

	ui->refresh();
}


void SegmentationDoc::convert_gray_image() {
	if (rgbImage == NULL) {
		fl_alert("couldn't load image file");
		return;
	}
	if (grayImage != NULL) {
		grayImage->release();
		grayImage = NULL;
	}
	CShape sh(rgbImage->w(), rgbImage->h(), 3);
	CFloatImage floatImage(sh);
	convertImage(rgbImage, floatImage);
	CShape graySh(rgbImage->w(), rgbImage->h(), 3);
	CFloatImage tempGrayImage(graySh);
	tempGrayImage = ConvertToGray(floatImage);
	CByteImage tempGrayByteImage = CByteImage(tempGrayImage.Shape());
	convertToByteImage(tempGrayImage, tempGrayByteImage);
	//writePNG(tempGrayByteImage, "tempgrayimg.png");
	grayImage = Fl_Shared_Image::get("gray.png");
	ui->resize_windows(rgbImage->w(), grayImage->w(), 0, rgbImage->h());
	ui->set_images(rgbImage, grayImage, NULL);
	ui->refresh();

}
void SegmentationDoc::vessel_segmentation() {
	if (grayImage == NULL) {
	}
	//Channel for convertImage is always 3!
	CShape sh(grayImage->w(), grayImage->h(), 3);
	CFloatImage floatImage(sh);
	// use float image to contain grayImage
	convertImage(grayImage, floatImage);

	// result of erosion
	CFloatImage resultImage = vesselSegmentation(floatImage);
	// convert to FL:
	// 1. convert to Byte
	CByteImage resultByteImage(resultImage.Shape());
	convertToByteImage(resultImage, resultByteImage);
	CShape singleChannel(grayImage->w(), grayImage->h(), 1);
	CByteImage resultByteImageSingleChannel(singleChannel);
	for (int x = 0; x < grayImage->w(); x++) {
		for (int y = 0; y < grayImage->h(); y++) {
			//convertToByteImage reversed channel number!!!
			// white pixel might result from here. // check later 
			resultByteImageSingleChannel.Pixel(x, y, 2) = resultByteImage.Pixel(x, y, 2);
		}
	}
	// 2. write to PNG
	writePNG(resultByteImageSingleChannel, "tempgrayimg.png");
	// 3. Read from PNG to Fl-share
	grayImage->release();
	grayImage = NULL;
	grayImage = Fl_Shared_Image::get("tempgrayimg.png");
	ui->resize_windows(rgbImage->w(), grayImage->w(), 0, rgbImage->h());
	ui->set_images(rgbImage, grayImage, NULL);
	ui->refresh();
}
void SegmentationDoc::morphological_operations() {
	fl_alert("Not implemented");
}
void SegmentationDoc::erosion() {
	if (grayImage== NULL) {
	}
	//Channel for convertImage is always 3!
	CShape sh(grayImage->w(), grayImage->h(), 3);
	CFloatImage floatImage(sh);
	// use float image to contain grayImage
	convertImage(grayImage, floatImage);

	// result of erosion

	CFloatImage resultImage = Erosion(floatImage, 2);
	// convert to FL:
	// 1. convert to Byte
	CByteImage resultByteImage(resultImage.Shape());
	convertToByteImage(resultImage, resultByteImage);
	CShape singleChannel(grayImage->w(), grayImage->h(), 1);
	CByteImage resultByteImageSingleChannel(singleChannel);
	for (int x = 0; x < grayImage->w(); x++) {
		for (int y = 0; y < grayImage->h(); y++) {
			//convertToByteImage reversed channel number!!!
			// white pixel might result from here. // check later 
			resultByteImageSingleChannel.Pixel(x, y, 2) = resultByteImage.Pixel(x, y, 2);
		}
	}
	// 2. write to PNG
	writePNG(resultByteImageSingleChannel, "tempgrayimg2.png");
	// 3. Read from PNG to Fl-share
	grayImage->release();
	grayImage = NULL;
	grayImage = Fl_Shared_Image::get("tempgrayimg2.png");
	ui->resize_windows(rgbImage->w(), grayImage->w(), 0, rgbImage->h());
	ui->set_images(rgbImage, grayImage, NULL);
	ui->refresh();
}
void SegmentationDoc::dilation() {
	if (grayImage == NULL) {
	}
	//Channel for convertImage is always 3!
	CShape sh(grayImage->w(), grayImage->h(), 3);
	CFloatImage floatImage(sh);
	// use float image to contain grayImage
	convertImage(grayImage, floatImage);

	// result of erosion

	CFloatImage resultImage = Dilation(floatImage, 3);
	// convert to FL:
	// 1. convert to Byte
	CByteImage resultByteImage(resultImage.Shape());
	convertToByteImage(resultImage, resultByteImage);
	CShape singleChannel(grayImage->w(), grayImage->h(), 1);
	CByteImage resultByteImageSingleChannel(singleChannel);
	for (int x = 0; x < grayImage->w(); x++) {
		for (int y = 0; y < grayImage->h(); y++) {
			//convertToByteImage reversed channel number!!!
			// white pixel might result from here. // check later 
			resultByteImageSingleChannel.Pixel(x, y, 2) = resultByteImage.Pixel(x, y, 2);
		}
	}
	// 2. write to PNG
	writePNG(resultByteImageSingleChannel, "tempgrayimg3.png");
	// 3. Read from PNG to Fl-share
	grayImage->release();
	grayImage = NULL;
	grayImage = Fl_Shared_Image::get("tempgrayimg3.png");
	ui->resize_windows(rgbImage->w(), grayImage->w(), 0, rgbImage->h());
	ui->set_images(rgbImage, grayImage, NULL);
	ui->refresh();
}
void SegmentationDoc::opening() {
	if (grayImage == NULL) {
	}
	//Channel for convertImage is always 3!
	CShape sh(grayImage->w(), grayImage->h(), 3);
	CFloatImage floatImage(sh);
	// use float image to contain grayImage
	convertImage(grayImage, floatImage);

	// result of erosion

	CFloatImage resultImage = Opening(floatImage, 3);
	// convert to FL:
	// 1. convert to Byte
	CByteImage resultByteImage(resultImage.Shape());
	convertToByteImage(resultImage, resultByteImage);
	CShape singleChannel(grayImage->w(), grayImage->h(), 1);
	CByteImage resultByteImageSingleChannel(singleChannel);
	for (int x = 0; x < grayImage->w(); x++) {
		for (int y = 0; y < grayImage->h(); y++) {
			//convertToByteImage reversed channel number!!!
			// white pixel might result from here. // check later 
			resultByteImageSingleChannel.Pixel(x, y, 2) = resultByteImage.Pixel(x, y, 2);
		}
	}
	// 2. write to PNG
	writePNG(resultByteImageSingleChannel, "tempgrayimg3.png");
	// 3. Read from PNG to Fl-share
	grayImage->release();
	grayImage = NULL;
	grayImage = Fl_Shared_Image::get("tempgrayimg3.png");
	ui->resize_windows(rgbImage->w(), grayImage->w(), 0, rgbImage->h());
	ui->set_images(rgbImage, grayImage, NULL);
	ui->refresh();
}
void SegmentationDoc::closing() {
	if (grayImage == NULL) {
	}
	//Channel for convertImage is always 3!
	CShape sh(grayImage->w(), grayImage->h(), 3);
	CFloatImage floatImage(sh);
	// use float image to contain grayImage
	convertImage(grayImage, floatImage);

	// result of erosion
	CFloatImage resultImage = Closing(floatImage, 3);
	// convert to FL:
	// 1. convert to Byte
	CByteImage resultByteImage(resultImage.Shape());
	convertToByteImage(resultImage, resultByteImage);
	CShape singleChannel(grayImage->w(), grayImage->h(), 1);
	CByteImage resultByteImageSingleChannel(singleChannel);
	for (int x = 0; x < grayImage->w(); x++) {
		for (int y = 0; y < grayImage->h(); y++) {
			//convertToByteImage reversed channel number!!!
			// white pixel might result from here. // check later 
			resultByteImageSingleChannel.Pixel(x, y, 2) = resultByteImage.Pixel(x, y, 2);
		}
	}
	// 2. write to PNG
	writePNG(resultByteImageSingleChannel, "tempgrayimg3.png");
	// 3. Read from PNG to Fl-share
	grayImage->release();
	grayImage = NULL;
	grayImage = Fl_Shared_Image::get("tempgrayimg3.png");
	ui->resize_windows(rgbImage->w(), grayImage->w(), 0, rgbImage->h());
	ui->set_images(rgbImage, grayImage, NULL);
	ui->refresh();
}
void SegmentationDoc::geodesic_erosion() {
	fl_alert("Not implemented");
}
void SegmentationDoc::geodesic_dilation() {
	if (grayImage == NULL) {
	}
	//Channel for convertImage is always 3!
	CShape sh(grayImage->w(), grayImage->h(), 3);
	CFloatImage floatImage(sh);
	// use float image to contain grayImage
	convertImage(grayImage, floatImage);

	// result of erosion
	CFloatImage resultImage = GeodesicDilation(floatImage, floatImage, 4);
	// convert to FL:
	// 1. convert to Byte
	CByteImage resultByteImage(resultImage.Shape());
	convertToByteImage(resultImage, resultByteImage);
	CShape singleChannel(grayImage->w(), grayImage->h(), 1);
	CByteImage resultByteImageSingleChannel(singleChannel);
	for (int x = 0; x < grayImage->w(); x++) {
		for (int y = 0; y < grayImage->h(); y++) {
			//convertToByteImage reversed channel number!!!
			// white pixel might result from here. // check later 
			resultByteImageSingleChannel.Pixel(x, y, 2) = resultByteImage.Pixel(x, y, 2);
		}
	}
	// 2. write to PNG
	writePNG(resultByteImageSingleChannel, "tempgrayimg3.png");
	// 3. Read from PNG to Fl-share
	grayImage->release();
	grayImage = NULL;
	grayImage = Fl_Shared_Image::get("tempgrayimg3.png");
	ui->resize_windows(rgbImage->w(), grayImage->w(), 0, rgbImage->h());
	ui->set_images(rgbImage, grayImage, NULL);
	ui->refresh();
	
}
void SegmentationDoc::geodesic_opening() {
	fl_alert("Not implemented");
}
void SegmentationDoc::geodesic_closing() {
	fl_alert("Not implemented");
}
// Set the UI pointer.
void SegmentationDoc::set_ui(SegmentationUI *ui) {
	this->ui = ui;
}
