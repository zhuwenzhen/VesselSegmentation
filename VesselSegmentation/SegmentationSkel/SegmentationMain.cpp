#include <assert.h>
#include <fstream>
#include <FL/Fl.H>
#include <FL/Fl_Shared_Image.H>
#include "ImageConvert.h"
#include "SegmentationUI.h"
#include "SegmentationDoc.h"
#include <algorithm>
#include "Morphology.h"
SegmentationUI  *ui;
SegmentationDoc *doc;

void convertToFloatImage(CByteImage &byteImage, CFloatImage &floatImage) {
	CShape sh = byteImage.Shape();

    assert(floatImage.Shape().nBands == min(byteImage.Shape().nBands, 3));
	for (int y=0; y<sh.height; y++) {
		for (int x=0; x<sh.width; x++) {
			for (int c=0; c<min(3,sh.nBands); c++) {
				float value = byteImage.Pixel(x,y,c) / 255.0f;

				if (value < floatImage.MinVal()) {
					value = floatImage.MinVal();
				}
				else if (value > floatImage.MaxVal()) {
					value = floatImage.MaxVal();
				}

				// We have to flip the image and reverse the color
				// channels to get it to come out right.  How silly!
				floatImage.Pixel(x,sh.height-y-1,min(3,sh.nBands)-c-1) = value;
			}
		}
	}
}

bool LoadImageFile(const char *filename, CFloatImage &image) 
{
	// Load the rgb image.
	Fl_Shared_Image *fl_image = Fl_Shared_Image::get(filename);

	if (fl_image == NULL) {
		// printf("couldn't load rgb image\n");
        CByteImage byteImage;
		ReadFile(byteImage, filename);

        CShape sh = byteImage.Shape();
        sh.nBands = 3;

        image = CFloatImage(sh);
        convertToFloatImage(byteImage, image);  
        return true;

    } else {
	    CShape sh(fl_image->w(), fl_image->h(), 3);
	    image = CFloatImage(sh);
	    // Convert the image to the CImage format.
	    if (!convertImage(fl_image, image)) {
		    printf("couldn't convert image to RGB format\n");
		    return false;
	    }   
        return true;
    }
}

int mainErosion(int argc, char **argv) {
	// what is this doing?
	if ((argc < 4) || (argc > 5)) {
		printf("");
	}

	// specify a test image
	char* imageDirectory = "C:/Users/wenzhen/Dropbox/2_Society/WUSTL/cse559a/VesselSegmentation/DRIVE/training/images/21_training.tga";
	CFloatImage floatQueryImage;

	bool success = LoadImageFile(imageDirectory, floatQueryImage);
	//bool success = LoadImageFile(argv[2], floatQueryImage);
	if (!success) {
		printf("couldn't load query image\n");
		return -1;
	}

	// Perform erosions
	CFloatImage result;
	result = Erosion(floatQueryImage, 2);
	// convertToByteImage(CFloatImage &floatImage, CByteImage &byteImage)
	ofstream filestream("filename.txt");
	CShape sh = result.Shape();
	int w = sh.width; int h = sh.height;
	
	for (int x = 0; x < w; x++) {	
		for (int y = 0; y < h; y++) {
			filestream << result.Pixel(x, y, 0) << " ";
		}
		filestream << "\n";	
	}
	filestream.close();
}

int main(int argc, char **argv) {
	// This lets us load various image formats.
	fl_register_images();
	doc = new SegmentationDoc();
	ui = new SegmentationUI();

	ui->set_document(doc);
	doc->set_ui(ui);

	Fl::visual(FL_DOUBLE | FL_INDEX);

	ui->show();

	return Fl::run();
}

/*if (argc > 1) {
		if (strcmp(argv[1], "erosion") == 0) {
			return mainErosion(argc, argv);
		}
	}*/

	/*if (argc > 1) {
		if (strcmp(argv[1], "computeFeatures") == 0) {
			return mainComputeFeatures(argc, argv);
		}*/
	//	else if (strcmp(argv[1], "matchFeatures") == 0) {
	//		return mainMatchFeatures(argc, argv);
	//	}
	//	else if (strcmp(argv[1], "matchSIFTFeatures") == 0) {
	//		return mainMatchSIFTFeatures(argc, argv);
	//	}
	//	else if (strcmp(argv[1], "testMatch") == 0) {
	//		return mainTestMatch(argc, argv);
	//	}
	//	else if (strcmp(argv[1], "testSIFTMatch") == 0) {
	//		return mainTestSIFTMatch(argc, argv);
	//	}
	//	else if (strcmp(argv[1], "benchmark") == 0) {
	//		return mainBenchmark(argc, argv);
	//	}
	//	else if (strcmp(argv[1], "rocSIFT") == 0)
	//	{
	//		//return saveRoc(argc,argv);
	//		mainRocTestSIFTMatch(argc,argv);
	//	}
	//	else if (strcmp(argv[1], "roc") == 0)
	//	{
	//		//return saveRoc(argc,argv);
	//		mainRocTestMatch(argc,argv);
	//	}
	//	
	//	else {
	//		printf("usage:\n");
	//		printf("\t%s\n", argv[0]);
	//		printf("\t%s computeFeatures imagefile featurefile [featuretype]\n", argv[0]);
	//		printf("\t%s matchFeatures featurefile1 featurefile2 matchfile [matchtype]\n", argv[0]);
	//		printf("\t%s matchSIFTFeatures featurefile1 featurefile2 matchfile [matchtype]\n", argv[0]);
	//		// printf("\t%s testMatch featurefile1 featurefile2 homographyfile [matchtype]\n", argv[0]);
	//		// printf("\t%s testSIFTMatch featurefile1 featurefile2 homographyfile [matchtype]\n", argv[0]);
	//		// printf("\t%s benchmark imagedir [featuretype matchtype]\n", argv[0]);
	//		printf("\t%s rocSIFT featurefile1 featurefile2 homographyfile [matchtype] filename\n", argv[0]);
	//		printf("\t%s roc featurefile1 featurefile2 homographyfile [matchtype] filename\n", argv[0]);

	//		return -1;
	//	}
	//}
	//else {
	//	// Use the GUI.
	//	doc = new SegmentationDoc();
	//	ui = new SegmentationUI();

	//	ui->set_document(doc);
	//	doc->set_ui(ui);

	//	Fl::visual(FL_DOUBLE|FL_INDEX);

	//	ui->show();

	//	return Fl::run();
	//}
