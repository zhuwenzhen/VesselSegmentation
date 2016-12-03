#ifndef SegmentationUI_H
#define SegmentationUI_H

class Fl_Window;
class Fl_Menu_;
class Fl_Menu_Bar;
class ImageView;
class SegmentationDoc;

// The SegmentationUI class controls the UI.  Feel free to play around with
// the UI if you'd like.

class SegmentationUI {
public:
	SegmentationDoc *doc;

	Fl_Window *mainWindow;
	Fl_Menu_Bar *menuBar;

	ImageView *rgbView;
	ImageView *grayView;
	ImageView *binaryView;

public:
	// Create the UI.
	SegmentationUI();

	// Begin displaying the UI.
	void show();

	// Refresh the window.
	void refresh();

	// Resize the image windows.
	void resize_windows(int w1, int w2, int w3, int h);

	// Set the document pointer.
	void set_document(SegmentationDoc *doc);

	// Set the pointers to the three images.
	void set_images(Fl_Image *rgbImage, Fl_Image *grayImage, Fl_Image *binaryImage);

	// Get the current match type.
	int get_match_type() const;

private:
	// Return the UI, given a menu item.
	static SegmentationUI *who_am_i(Fl_Menu_ *o);

	// Here are the callback functions.
	static void cb_load_rgb_image(Fl_Menu_ *o, void *v);
	static void cb_convert_gray_image(Fl_Menu_ *o, void *v);
	static void cb_vessel_segmentation(Fl_Menu_ *o, void *v);
	static void cb_morphological_operations(Fl_Menu_ *o, void *v);
	static void cb_erosion(Fl_Menu_ *o, void *v);
	static void cb_dilation(Fl_Menu_ *o, void *v);
	//static void cb_closing(Fl_Menu_ *o, void *v);
	//static void cb_opening(Fl_Menu_ *o, void *v);
	//static void cb_geodesic_erosion(Fl_Menu_ *o, void *v);
	//static void cb_geodesic_dilation(Fl_Menu_ *o, void *v);
	//static void cb_geodesic_opening(Fl_Menu_ *o, void *v);
	//static void cb_geodesic_closing(Fl_Menu_ *o, void *v);
	static void cb_exit(Fl_Menu_ *o, void *v);
	static void cb_about(Fl_Menu_ *o, void *v);

	//static void cb_load_rgb_features(Fl_Menu_ *o, void *v);
	//static void cb_load_rgb_features_sift(Fl_Menu_ *o, void *v);
	//static void cb_load_image_database(Fl_Menu_ *o, void *v);
	//static void cb_load_image_database_sift(Fl_Menu_ *o, void *v)
	//static void cb_select_all_features(Fl_Menu_ *o, void *v);
	//static void cb_deselect_all_features(Fl_Menu_ *o, void *v);
	//static void cb_toggle_features(Fl_Menu_ *o, void *v);


	// Here is the array of menu items.
	static Fl_Menu_Item menuItems[];
};

#endif