#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Menu_Item.H>
#include <FL/Fl_File_Chooser.H>
#include <FL/fl_ask.H>
#include "ImageView.h"
#include "SegmentationDoc.h"
#include "SegmentationUI.h"

// Create and initialize the UI.
SegmentationUI::SegmentationUI() {
	// Create the main window.
	mainWindow = new Fl_Window(250, 250, "Vessel Segmentation");
	mainWindow->user_data((void *)this);

	// Create the menu bar.
	menuBar = new Fl_Menu_Bar(0, 0, 250, 25);
	menuBar->menu(menuItems);

	// Create the subwindows for viewing importing images.

	//1. RGB
	rgbView = new ImageView(0, 25, 250, 250, "RGB view");
	rgbView->box(FL_DOWN_FRAME);

	// 2.GrayScale
	grayView = new ImageView(250, 25, 250, 250, "Gray scale view");
	grayView->box(FL_DOWN_FRAME);

	// 3.Binary
	binaryView = new ImageView(500, 25, 250, 250, "Binary view");
	binaryView->box(FL_DOWN_FRAME);

	// Tell binaryView not to take any events.
	binaryView->set_output();

	mainWindow->end();
}

// Begin displaying the UI windows.
void SegmentationUI::show() {
	mainWindow->show();
	rgbView->show();
	grayView->show();
	binaryView->show();
}

// Refresh the window.
void SegmentationUI::refresh() {
	mainWindow->redraw();
	rgbView->redraw();
	grayView->show();
	binaryView->redraw();
}

// Resize the rgb and binary image windows.
void SegmentationUI::resize_windows(int w1, int w2, int w3, int h) {
	mainWindow->resize(mainWindow->x(), mainWindow->y(), w1+w2+w3, h+25);
	menuBar->resize(menuBar->x(), menuBar->y(), w1+w2+w3, 25);
	rgbView->resize(rgbView->x(), rgbView->y(), w1, h);
	grayView->resize(grayView->x(), grayView->y(), w2, h);
	binaryView->resize(rgbView->x()+w1, rgbView->y(), w3, h);
}

// Set the document pointer.
void SegmentationUI::set_document(SegmentationDoc *doc) {
	this->doc = doc;
}

// Set the rgb, gray, and binary image pointers.
void SegmentationUI::set_images(Fl_Image *rgbImage, Fl_Image *grayImage, Fl_Image *binaryImage) {
	rgbView->setImage(rgbImage);
	grayView->setImage(grayImage);
	binaryView->setImage(binaryImage);
}


// Identify the UI from a menu item.
SegmentationUI* SegmentationUI::who_am_i(Fl_Menu_ *o) {
	return (SegmentationUI *)(o->parent()->user_data());
}

// Called when the user chooses the "Load rgb Image" menu item.
void SegmentationUI::cb_load_rgb_image(Fl_Menu_ *o, void *v) {
	SegmentationDoc *doc = who_am_i(o)->doc;

    char *name = fl_file_chooser("Open File", "{*.p[gp]m,*.jpg}", NULL);

	if (name != NULL) {
		doc->load_rgb_image(name);
	}
}

// Called when the user chooses the "Perform rgb" menu item.
//void SegmentationUI::cb_perform_rgb(Fl_Menu_ *o, void *v) {
//	who_am_i(o)->doc->perform_rgb();
//}

void SegmentationUI::cb_convert_gray_image(Fl_Menu_ *o, void *v) {
	who_am_i(o)->doc->convert_gray_image();
}

void SegmentationUI::cb_vessel_segmentation(Fl_Menu_ *o, void *v) {
	who_am_i(o)->doc->vessel_segmentation();
}
void SegmentationUI::cb_morphological_operations(Fl_Menu_ *o, void *v) {
	who_am_i(o)->doc->morphological_operations();
}

// Called when the user selects erosion.
void SegmentationUI::cb_erosion(Fl_Menu_ *o, void *v) {
	who_am_i(o)->doc->erosion();
}

// Called when the user selects dilation
void SegmentationUI::cb_dilation(Fl_Menu_ *o, void *v) {
	who_am_i(o)->doc->dilation();
}


//void SegmentationUI::cb_closing(Fl_Menu_ *o, void *v) {
//	who_am_i(o)->doc->closing();
//}
//
//void SegmentationUI::cb_opening(Fl_Menu_ *o, void *v) {
//	who_am_i(o)->doc->opening();
//}

// Called when the user clicks the "About" menu item.
void SegmentationUI::cb_about(Fl_Menu_ *o, void *v) {
	fl_message("Project: Vessel Segmentation UI");
}

// Once again, you can add any extra menu items you like.
Fl_Menu_Item SegmentationUI::menuItems[] = {
	{"&File", 0, 0, 0, FL_SUBMENU},
		{"&Load rgb Image", 0, (Fl_Callback *)SegmentationUI::cb_load_rgb_image},	
		{"&Exit", 0, (Fl_Callback *)SegmentationUI::cb_exit},
		{0},
	{"&Image Operation", 0, 0, 0, FL_SUBMENU},
		{"&Convert to Gray Scale", 0, (Fl_Callback *)SegmentationUI::cb_convert_gray_image},
		{"&Vessel Segmentation", 0, (Fl_Callback *)SegmentationUI::cb_vessel_segmentation},
		{0},
	{"&Morphology Operation", 0, 0, 0, FL_SUBMENU},
		{"&Erosion", 0, (Fl_Callback *)SegmentationUI::cb_erosion},
		{"&Dilation", 0, (Fl_Callback *)SegmentationUI::cb_dilation},
		/*{"&Opening", 0, (Fl_Callback *)SegmentationUI:: cb_opening},
		{"&Closing", 0, (Fl_Callback *)SegmentationUI:: cb_closing},
		{"&geodesic_erosion", 0, (Fl_Callback *)SegmentationUI:: cb_geodesic_erosion},
		{"&Geodesic dilation",0, (Fl_Callback *)SegmentationUI:: cb_geodesic_dilation},
		{"&Geodesic opening", 0, (Fl_Callback *)SegmentationUI:: cb_geodesic_opening},
		{"&Geodesic closing", 0, (Fl_Callback *)SegmentationUI:: cb_geodesic_closing},*/
		{0},

	{"&Help", 0, 0, 0, FL_SUBMENU},
		{"&About", 0, (Fl_Callback *)SegmentationUI::cb_about},
		{0},
	{0}
};
