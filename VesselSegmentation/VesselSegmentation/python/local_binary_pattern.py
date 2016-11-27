import numpy
import mahotas.features.lbp
def compute_lbp(img):
    #array2d = numpy.array(list(img.getdata()))
    #array2d = array2d.reshape(img.size())
    #features_array = mahotas.features.lbp.lbp(array2d)
    features_array = mahotas.features.lbp.lbp(img)
    return features_array
