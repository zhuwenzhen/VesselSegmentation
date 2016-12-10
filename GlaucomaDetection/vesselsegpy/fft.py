import sklearn.decomposition
import numpy as np

def extract_FFT(imgs):
    fft = [np.fft.fft2(img, axes=(0,1)).flatten() for img in imgs]
    print(len(fft))
    fft = np.array(fft)
    pca = sklearn.decomposition.PCA(n_components=30)
    reduced = pca.fit(fft).transform(fft)
    return (reduced, pca)

def project_FFT(img, breducer):
    fft = np.fft.fft2(img, axes=(0,1))
    fft = fft.flatten()
    return breducer.transform(fft)
