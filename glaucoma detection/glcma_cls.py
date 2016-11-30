import scipy.ndimage
import argparse
import sklearn
import sklearn.externals
import logging
import vesselsegpy
import numpy as np

logging.basicConfig(format="%(levelname)s:%(asctime)s:%(module)s:%(funcName)s:%(lineno)d:%(message)s")
logger = logging.getLogger("main_logger")
logger.setLevel(logging.INFO)
def main(args):
    pass

def train(manifest_filename):
    clf = svm.SVC()
    Xs = []
    Ys = []
    with open(manifest_filename, "rt") as manifest_f:
        for line in manifest_f:
            logger.info("starting to train %s", line)
            filename, label = line.strip(" ").split(" ")
            img = scipy.ndimage.imread(filename, mode = "L")
            logger.info("computing lbp...")
            lbp_fts = vesselsegpy.lbp.compute_lbp(img)
            logger.info("computing lme...")
            lme_fts = vesselsegpy.lme.compute_laws_mask_energy(img)
            logger.info("computing fglcm...")
            fglcm_mats = vesselsegpy.fglcm.fglcm(img, 20)
            logger.info("computing fglcm_energy...")
            fglcm_energy = vesselsegpy.fglcm.fglcm_energy(fglcm_mats)
            logger.info("computing fglcm_entropy...")
            fglcm_entropy = vesselsegpy.fglcm.fglcm_entropy(fglcm_mats)
            frlm_fts = frlm(img)
            logger.info("concanating features...")
            features = np.array(lbp_fts.flatten(), lme_fts.flatten(), fglcm_energy.flatten(), fglcm_entropy.flatten(), frlm_fts.flatten())
            Xs.append(features)
            Ys.append(label)
    logger.info("training classifier...")
    clf.train(np.array(Xs), np.array(Ys))
    logger.info("saving model...")
    sklearn.externals.joblib.dump(clf, 'trained.pkl')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="classify images")