#include <FL/Fl.H>
#include <FL/Fl_Image.H>
#include <FL/fl_draw.H>
#include "ImageView.h"

// Create a new ImageView object.
ImageView::ImageView(int x, int y, int w, int h, const char *l) : Fl_Double_Window(x,y,w,h,l) {
	image = NULL;

	// This call is necessary to prevent any additional UI widgets
	// from becoming subcomponents of this window.
	end();
}

void ImageView::draw() {
	if (image != NULL) {
		image->draw(0, 0);
	}
}

void ImageView::refresh() {
	redraw();
}

void ImageView::setImage(Fl_Image *image) {
	this->image = image;
}