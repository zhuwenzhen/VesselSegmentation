import numpy as np
import mahotas.features.lbp
def compute_lbp(img):
    #array2d = numpy.array(list(img.getdata()))
    #array2d = array2d.reshape(img.size())
    #features_array = mahotas.features.lbp.lbp(array2d)
    parameters = [(1, 8), (2, 16), (3, 24)]
    features_hist = [mahotas.features.lbp(img, r, p) for (r, p) in parameters]
    features_array = [i for hist in features_hist for i in hist]
    return features_array
def compute_entropy(lbp_img):
    return np.sum(-np.multiply(lbp_img, np.log(lbp_img)))
