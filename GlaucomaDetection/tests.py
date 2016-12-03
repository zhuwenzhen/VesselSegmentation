import vesselsegpy
import scipy.ndimage
import logging

import sklearn.externals
import numpy as np
import PIL
import cv2

np.set_printoptions(threshold=np.nan)

logging.basicConfig(format="%(levelname)s:%(asctime)s:%(module)s:%(funcName)s:%(lineno)d:%(message)s")
logger = logging.getLogger("main_logger")
logger.setLevel(logging.DEBUG)

# img = scipy.ndimage.imread("02_g.jpg", mode = "L")
# img = cv2.equalizeHist(img)
#img = img.astype(dtype=float, casting="safe")
#debug_img = PIL.Image.fromarray(img.astype(dtype="uint8"))
#debug_img.save("grey_scale.jpg", mode="L")
# logger.info("computing fglcm...")
# fglcm_mats = vesselsegpy.fglcm.fglcm(img, 20)
# print(fglcm_mats)
# logger.info("computing fglcm_energy...")
# fglcm_energy = vesselsegpy.fglcm.fglcm_energy(fglcm_mats)
# print(fglcm_energy)
# logger.info("computing fglcm_entropy...")
# fglcm_entropy = vesselsegpy.fglcm.fglcm_entropy(fglcm_mats)
# print(fglcm_entropy)
# logger.info("computing frlm_fts...")
# frlm_fts = vesselsegpy.frlm.frlm(img)
#print(list(map(vesselsegpy.frlm.compute_sre, frlm_fts)))
#vesselsegpy.lme.compute_laws_mask_energy(img)

def compute_avgforfglcm(manifest_filename):
    counter = 0
    fglcm_energy_t = 0
    fglcm_entropy_t = 0
    with open(manifest_filename, "rt") as manifest_f:
        for line in manifest_f:
            line = line.strip("\n")
            counter += 1
            logger.info("starting to compute for %s", line)
            img = scipy.ndimage.imread(line, mode = "L")
            logger.info("computing fglcm...")
            fglcm_mats = vesselsegpy.fglcm.fglcm(img, 20)
            logger.info("computing fglcm_energy...")
            fglcm_energy = vesselsegpy.fglcm.fglcm_energy(fglcm_mats)
            logger.info("fglcm energy...: %s", fglcm_energy)
            fglcm_entropy = vesselsegpy.fglcm.fglcm_entropy(fglcm_mats)
            logger.info("fglcm entropy...: %s", fglcm_entropy)
            fglcm_energy_t +=fglcm_energy
            fglcm_entropy_t+=fglcm_entropy
    logger.info("computing avg...")
    logger.info("energy...")
    print(fglcm_energy_t/counter)
    logger.info("entropy...")
    print(fglcm_entropy_t/counter)
compute_avgforfglcm("healthy.txt")
def compute_avg(manifest_filename):
    counter = 0
    frlm_135 = 0
    all_fglcm = []
    with open(manifest_filename, "rt") as manifest_f:
        for line in manifest_f:
            line = line.strip("\n")
            counter += 1
            logger.info("starting to compute for %s", line)
            img = scipy.ndimage.imread(line, mode = "L")
            #logger.info("computing fglcm...")
            #fglcm_mats = vesselsegpy.fglcm.fglcm(img, 20)
            #logger.info("computing fglcm_energy...")
            #fglcm_energy = vesselsegpy.fglcm.fglcm_energy(fglcm_mats)
            logger.info("computing fglcm_entropy...")
            lme_fts = vesselsegpy.lme.compute_laws_mask_energy(img)
            #fglcm_entropy = vesselsegpy.fglcm.fglcm_entropy(fglcm_mats)
            frlm_fts = vesselsegpy.frlm.frlm(img)
            frlm_135 += vesselsegpy.frlm.compute_sre(frlm_fts[3])
            logger.info("fglcm_entropy: %s", vesselsegpy.frlm.compute_sre(frlm_fts[3]))
            all_fglcm.append(str(vesselsegpy.frlm.compute_sre(frlm_fts[3])) + "\n")
    logger.info("computing avg...")
    print(frlm_135/counter)
    with open("avg.txt", mode="wt") as f:
        f.writelines(all_fglcm)
def compute_lte(manifest_filename):
    counter = 0
    all_fglcm = []
    with open(manifest_filename, "rt") as manifest_f:
        for line in manifest_f:
            line = line.strip("\n")
            counter += 1
            logger.info("starting to compute for %s", line)
            img = scipy.ndimage.imread(line, mode = "L")
            lme_fts = vesselsegpy.lme.compute_laws_mask_energy(img)
            lte_energies = [vesselsegpy.lme.compute_energy(ft) for ft in lme_fts]
            print(lte_energies)

#compute_avg("normal.txt")
#compute_lte("healthy.txt")
