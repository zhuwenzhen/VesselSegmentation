#include <assert.h>
#include <FL/Fl.H>
#include <FL/Fl_Shared_Image.H>
#include <FL/fl_ask.H>
#include "ImageConvert.h"
#include "SegmentationUI.h"
#include "SegmentationDoc.h"

// Create a new document.
SegmentationDoc::SegmentationDoc() {
	rgbImage = NULL;
	binaryImage = NULL;
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


// Set the UI pointer.
void SegmentationDoc::set_ui(SegmentationUI *ui) {
	this->ui = ui;
}
