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
def compute_features(img):
    # logger.info("computing lbp...")
    # lbp_fts = vesselsegpy.lbp.compute_lbp(img)
    img = cv2.equalizeHist(img)
    # logger.info("computing lme...")
    # lme_fts = vesselsegpy.lme.compute_laws_mask_energy(img)
    # lte_energies = [vesselsegpy.lme.compute_energy(ft) for ft in lme_fts]
    # logger.info("computing lme...%s",lte_energies)
    logger.info("computing fglcm...")
    fglcm_mats = vesselsegpy.fglcm.fglcm(img, 20)
    logger.info("computing fglcm_energy...")
    fglcm_energy = vesselsegpy.fglcm.fglcm_energy(fglcm_mats)
    logger.info("computing fglcm_entropy...")
    fglcm_entropy = vesselsegpy.fglcm.fglcm_entropy(fglcm_mats)
    logger.info("computing frlm_fts...")
    frlm_fts = vesselsegpy.frlm.frlm(img)
    frlm_sre = vesselsegpy.frlm.compute_sre(frlm_fts[3])
    logger.info("concanating features...")
    #print(np.array(lbp_fts).flatten())
    #np.array(lte_energies).flatten(),
    #np.array(lbp_fts).flatten(),
    #, np.array(frlm_sre).flatten()
    features = np.concatenate(( np.array(fglcm_energy).flatten(), np.array(fglcm_entropy).flatten()))
    #features =np.array(lme_fts).flatten()
    print(features.shape)
    return features
def compute_features_for_manifest(path_prefix, manifest):
    Xs = []
    with open(manifest, "rt") as manifest_f:
        for line in manifest_f:
            line = line.rstrip('\n')
            logger.info("starting to train %s", line)
            img = scipy.ndimage.imread(os.path.join(path_prefix, line), mode = "L")
            features = compute_features(img)
            Xs.append(features)
    return Xs
def main(args):
    pass

def train(glaucoma_img_manifest, healthy_img_manifest):
    glaucoma_prefix = os.path.dirname(glaucoma_img_manifest)
    health_prefix = os.path.dirname(healthy_img_manifest)
    clf = sklearn.svm.SVC()
    Xs = []
    Ys = []
    glaucoma_features = compute_features_for_manifest(glaucoma_prefix, glaucoma_img_manifest)
    Xs.extend(glaucoma_features)
    Ys.extend([1] * len(glaucoma_features))
    healthy_features = compute_features_for_manifest(health_prefix, healthy_img_manifest)
    Xs.extend(healthy_features)
    Ys.extend([-1] * len(healthy_features))
    logger.info("training classifier...")
    clf.fit(np.array(Xs), np.array(Ys))
    logger.info("saving model...")
    sklearn.externals.joblib.dump(clf, 'trained.pkl')

def classify(manifest):
    clf = sklearn.externals.joblib.load('trained.pkl')
    prefix = os.path.dirname(manifest)
    with open(manifest, "rt") as manifest_f:
        for line in manifest_f:
            logger.info("predicting for %s", line)
            line = line.rstrip('\n')
            img = scipy.ndimage.imread(os.path.join(prefix, line), mode = "L")
            features = compute_features(img).reshape(1, -1)
            print(clf.predict(features))
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="classify images")
train("pics/glaucoma/glaucoma_train.txt", "pics/normal/healthy_train.txt")
#"pics/normal/healthy_test.txt" "pics/glaucoma/glaucoma_test.txt"
classify("pics/normal/healthy_test.txt")
classify("pics/glaucoma/glaucoma_test.txt")
