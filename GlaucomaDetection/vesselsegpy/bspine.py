import scipy.interpolate
import sklearn.decomposition
import numpy as np
def extract_bspine(imgs):
    flattened_data = np.ndarray((len(imgs),imgs[0].shape[0] * imgs[0].shape[1]))
    for img in imgs:
        flattened_data = np.append(flattened_data, img.flatten())
    x = [i for i in range(imgs[0].shape[1])]
    y = [i for i in range(imgs[0].shape[0])]
    bspines = [scipy.interpolate.RectBivariateSpline(x, y, img, kx=4, ky=4).get_coeffs() for img in imgs]
    bspines = np.array([bspine.flatten() for bspine in bspines])
    pca = sklearn.decomposition.PCA(n_components=30)
    reduced = pca.fit(bspines).transform(bspines)
    return (reduced, pca)
def project_bspine(img, breducer):
    x = [i for i in range(img.shape[1])]
    y = [i for i in range(img.shape[0])]
    bspine = scipy.interpolate.RectBivariateSpline(x, y, img, kx=4, ky=4).get_coeffs()
    return breducer.transform(bspine)
