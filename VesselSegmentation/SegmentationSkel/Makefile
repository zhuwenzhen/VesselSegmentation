# Makefile for project 1
include Makefile.FLTK

PROJ1=Features
PROJ1_OBJS=FeaturesDoc.o FeaturesUI.o FeatureSet.o ImageDatabase.o \
		ImageView.o FeaturesMain.o features.o

IMAGELIB=ImageLib/libImage.a

# Depending on where you installed FLTK, you may need to change -I and -L here.
# For example, I added "-L/home/research/furukawa/usr/local/lib" at the end of LIB_PATH.
CPPFLAGS=-Wall -O3 -I/usr/local/include
LIB_PATH=-L/usr/local/lib -L/usr/X11R6/lib
LIBS=-lfltk -lfltk_images -lpng -ljpeg -lX11 -lXext -lXi -lXmu

all: $(PROJ1)

$(IMAGELIB): 
	make -C ImageLib

$(PROJ1): $(PROJ1_OBJS) $(IMAGELIB)
	$(CXX) -o $@ ${LINKFLTK_ALL} $(PROJ1_OBJS) $(LIB_PATH) $(LIBS) $(IMAGELIB)

clean:
	rm -f *.o *~ $(PROJ1)
