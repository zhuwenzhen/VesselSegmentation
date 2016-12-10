import scipy.ndimage
import argparse
import sklearn.svm
import sklearn.externals
import logging
import vesselsegpy
import numpy as np
import os.path
import cv2
import PIL
logging.basicConfig(format="%(levelname)s:%(asctime)s:%(module)s:%(funcName)s:%(lineno)d:%(message)s")
logger = logging.getLogger("main_logger")
logger.setLevel(logging.INFO)
def compute_classify(img, preducer, freducer, breducer):
    logger.info("equalizing...")
    #img = cv2.equalizeHist(img)
    logger.info("computing pca...")
    pca = vesselsegpy.pca.project_PCA(img, preducer)
    logger.info("computing fft...")
    fft = vesselsegpy.fft.project_FFT(img, freducer)
    logger.info("computing bspines...")
    bspines = vesselsegpy.bspine.project_bspine(img, breducer)
    logger.info("concanating features...")
    #features = np.concatenate(( np.array(pca).flatten(), np.array(fft).flatten(), np.array(bspines).flatten()))
    #features =np.array(lme_fts).flatten()
    #print(features.shape)
    return (np.array(pca), np.array(fft), np.array(bspines))

def compute_features(imgs):
    logger.info("equalizing...")
    #img = [cv2.equalizeHist(img) for img in imgs]
    logger.info("computing pca...")
    (pca, preducer) = vesselsegpy.pca.extract_PCA(imgs)
    logger.info("computing fft...")
    (fft, freducer) = vesselsegpy.fft.extract_FFT(imgs)
    logger.info("computing bspines...")
    (bspines, breducer) = vesselsegpy.bspine.extract_bspine(imgs)
    logger.info("concanating features...")

    #features = np.concatenate(( np.array(pca), np.array(fft), np.array(bspines)), axis=1)
    #print(np.array(fft).flatten().shape)
    #features =np.array(lme_fts).flatten()
    #print(features.shape)
    return (np.array(pca), np.array(fft), np.array(bspines), preducer, freducer, breducer)
def compute_features_for_manifest(glaucoma_prefix, glaucom_manifest, health_prefix, health_manifest):
    imgs = []
    Xs = []
    glaucoma = 0
    health = 0
    with open(glaucom_manifest, "rt") as manifest_f:
        for line in manifest_f:
            glaucoma+=1
            line = line.rstrip('\n')
            img = scipy.ndimage.imread(os.path.join(glaucoma_prefix, line), mode = "L")
            imgs.append(img)
    with open(health_manifest, "rt") as manifest_f:
        for line in manifest_f:
            health +=1
            line = line.rstrip('\n')
            img = scipy.ndimage.imread(os.path.join(health_prefix, line), mode = "L")
            imgs.append(img)
    pca, fft, bspines, preducer, freducer, breducer = compute_features(imgs)
    return (pca, fft, bspines, preducer, freducer, breducer, glaucoma, health)
def main(args):
    pass

def train(glaucoma_img_manifest, healthy_img_manifest):
    glaucoma_prefix = os.path.dirname(glaucoma_img_manifest)
    health_prefix = os.path.dirname(healthy_img_manifest)
    clf = sklearn.svm.SVC()
    clf_p = sklearn.svm.SVC(probability=True)
    clf_f = sklearn.svm.SVC(probability=True)
    clf_g = sklearn.svm.SVC(probability=True)
    Ys = []
    pca, fft, bspines, preducer, freducer, breducer, glaucoma_num, health_num = compute_features_for_manifest(glaucoma_prefix, glaucoma_img_manifest, health_prefix, healthy_img_manifest)
    Ys.extend([1] * glaucoma_num)
    Ys.extend([-1] * health_num)
    print(Ys)
    logger.info("training classifier...")
    clf_p.fit(pca, np.array(Ys))
    clf_f.fit(fft, np.array(Ys))
    clf_g.fit(bspines, np.array(Ys))
    clf.fit(np.concatenate((clf_p.predict_proba(pca)[:,0].reshape(-1, 1), clf_f.predict_proba(fft)[:,0].reshape(-1, 1), clf_g.predict_proba(bspines)[:,0].reshape(-1, 1)), axis=1), Ys)
    logger.info("saving model...")
    sklearn.externals.joblib.dump(clf, 'trained.pkl')
    sklearn.externals.joblib.dump(clf_p, 'trainedp.pkl')
    sklearn.externals.joblib.dump(clf_f, 'trainedf.pkl')
    sklearn.externals.joblib.dump(clf_g, 'trainedg.pkl')
    sklearn.externals.joblib.dump(preducer, 'preducer.pkl')
    sklearn.externals.joblib.dump(freducer, 'freducer.pkl')
    sklearn.externals.joblib.dump(breducer, 'breducer.pkl')
def classify(manifest):
    clf = sklearn.externals.joblib.load('trained.pkl')
    clf_p = sklearn.externals.joblib.load('trainedp.pkl')
    clf_f = sklearn.externals.joblib.load('trainedf.pkl')
    clf_g = sklearn.externals.joblib.load('trainedg.pkl')
    preducer = sklearn.externals.joblib.load('preducer.pkl')
    freducer = sklearn.externals.joblib.load('freducer.pkl')
    breducer = sklearn.externals.joblib.load('breducer.pkl')
    prefix = os.path.dirname(manifest)
    with open(manifest, "rt") as manifest_f:
        for line in manifest_f:
            logger.info("predicting for %s", line)
            line = line.rstrip('\n')
            img = scipy.ndimage.imread(os.path.join(prefix, line), mode = "L")
            pca, fft, bspines = compute_classify(img, preducer, freducer, breducer)
            print(clf_p.predict(pca))
            #print(clf_p.predict_proba(pca)[:,1])
            #print(clf.predict(np.concatenate((clf_p.predict_proba(pca)[:,0].reshape(-1, 1), clf_f.predict_proba(fft)[:,0].reshape(-1, 1), clf_g.predict_proba(bspines)[:,0].reshape(-1, 1)), axis=1)))
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="classify images")
train("pics/glaucoma_cropped/manifest_train.txt", "pics/healthy_cropped/manifest_train.txt")
#"pics/normal/healthy_test.txt" "pics/glaucoma/glaucoma_test.txt"
classify("pics/glaucoma_cropped/manifest_test.txt")
classify("pics/healthy_cropped/manifest_test.txt")
