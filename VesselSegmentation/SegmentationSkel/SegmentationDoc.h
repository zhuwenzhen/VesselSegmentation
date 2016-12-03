#ifndef SegmentationDoc_H
#define SegmentationDoc_H

class Fl_Shared_Image;
class SegmentationUI;

// The SegmentationDoc class controls the functionality of the project, and
// has methods for all major operations, like loading image and
// and perfoming morphological operations

class SegmentationDoc {
private:
	Fl_Shared_Image *rgbImage;
	Fl_Shared_Image *grayImage;
	Fl_Shared_Image *binaryImage;

	int morphology_code; // should I hard-code them??

public:
	SegmentationUI *ui;

public:
	// Create a new document.
	SegmentationDoc();

	// Destroy the document.
	~SegmentationDoc();

	// Load an image
	void load_rgb_image(const char *name);
	

	// perform convert to grayscale on the currently loaded image
	void convert_gray_image();

	// perform segmentation on the gray image
	void vessel_segmentation();

	// morphological Operations
	void morphological_operations();

	// morphological operators
	void erosion();
	void dilation();
	void opening();
	void closing();
	void geodesic_erosion();
	void geodesic_dilation();
	void geodesic_opening();
	void geodesic_closing();


	// Set the pointer to the UI.
	void set_ui(SegmentationUI *ui);

	// Select or deselect all rgb features.
	void select_all_rgb_features();
	void deselect_all_rgb_features();

	// Set the match algorithm.
    int get_match_algorithm() { return morphology_code; }
    void set_match_algorithm(int type);
};

#endif