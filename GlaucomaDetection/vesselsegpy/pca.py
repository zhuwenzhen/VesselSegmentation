import sklearn.decomposition
import numpy as np
def extract_PCA(imgs):
    flattened_data = np.ndarray((0,imgs[0].shape[0] * imgs[0].shape[1]))
    for img in imgs:
        flattened_data = np.append(flattened_data, np.array(img.flatten(), ndmin=2), axis=0)
    pca = sklearn.decomposition.PCA(n_components=30)
    reduced = pca.fit(flattened_data).transform(flattened_data)
    #print(reduced)
    return (reduced, pca)
def project_PCA(img, preducer):
    return preducer.transform(img.flatten())
