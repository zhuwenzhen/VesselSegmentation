import vesselsegpy
import scipy.ndimage
import logging

import sklearn.externals
import numpy as np

np.set_printoptions(threshold=np.nan)

logging.basicConfig(format="%(levelname)s:%(asctime)s:%(module)s:%(funcName)s:%(lineno)d:%(message)s")
logger = logging.getLogger("main_logger")
logger.setLevel(logging.DEBUG)

img = scipy.ndimage.imread("test2_scale.jpg", mode = "L")
logger.info("computing fglcm...")
#fglcm_mats = vesselsegpy.fglcm.fglcm(img, 20)
#print(fglcm_mats)
logger.info("computing fglcm_energy...")
#fglcm_energy = vesselsegpy.fglcm.fglcm_energy(fglcm_mats)
#print(fglcm_energy)
logger.info("computing fglcm_entropy...")
#fglcm_entropy = vesselsegpy.fglcm.fglcm_entropy(fglcm_mats)
#print(fglcm_entropy)
logger.info("computing frlm_fts...")
frlm_fts = vesselsegpy.frlm.frlm(img)




print(frlm_fts)
