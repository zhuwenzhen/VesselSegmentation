#ifndef IMAGEVIEW_H
#define IMAGEVIEW_H

#include <FL/Fl_Double_Window.H>

#include <iostream>
#include <fstream>

using namespace std;

class Fl_Image;

// The ImageView class is responsible for drawing an image and its
// associated features.
class ImageView : public Fl_Double_Window {
private:
	Fl_Image *image;

public:
	// Create an ImageView object.
	ImageView(int x, int y, int w, int h, const char *l);
	
	// Draw the image and features.
	void draw();

	// Refresh the window.
	void refresh();

	// Set the pointer to the image.
	void setImage(Fl_Image *image);

};
#endif
