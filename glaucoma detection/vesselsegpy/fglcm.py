import scipy.interpolate
import math
import numpy as np
import logging
import numba

logger = logging.getLogger("main_logger")
#scipy.interpolate.interp2d(x_coords, y_coords, img)
#degree = [[1, 0], [1, 1], [0, 1], [-1, 1]]
# @numba.jit
# def fglcm_check_point(img, x, y, d, degree):
#     #if(angle_mode == "deg"):
#     #    degree = math.radians(degree)
#     if(round(img[y][x]) == m):
#         new_x = x + (d * degree)[0]
#         new_y = y + (d * degree)[1]
#         if(round(img[new_y, new_x]) == n):
#             return True
#     return False
def _map_group(v):
    return v//32
def glcm(img, degree, d):
    glc_mat = np.zeros((8,8))
    if degree[0] == -1 and degree[1] == 1 :
        for x in range((d*degree)[0], img.shape[0]):
            for y in range(img.shape[1]- (d * degree)[1]):
                #print("x,y: ", (x, y))
                new_x = x + (d * degree)[0]
                new_y = y + (d * degree)[1]
                #print("new_x,new_y: ", (new_x, new_y))
                glc_mat[img[x, y]//32][img[new_x, new_y]//32] += 1
    else:
        for x in range(img.shape[0] - (d * degree)[0]):
            for y in range(img.shape[1] - (d * degree)[1]):
                #print("x,y: ", (x, y))
                new_x = x + (d * degree)[0]
                new_y = y + (d * degree)[1]
                #print("new_x,new_y: ", (new_x, new_y))
                glc_mat[img[x, y]//32][img[new_x, new_y]//32] += 1

    return glc_mat
def fglcm(img, d):
    #x_coords = list(range(0, img.shape[0]))
    #y_coords = list(range(0, img.shape[1]))
    degrees = np.array([[1, 0], [1, 1], [0, 1], [-1, 1]])
    #
    glc_mats = []
    for deg in degrees:
        logger.debug("computing deg: %s", deg)
        glc_mat = glcm(img, deg, d)
        glc_mats.append(glc_mat)
    fglcm = sum(glc_mats) / len(glc_mats)
    #max_ele = max(max(row) for row in fglcm)
    #min_ele = max(max(row) for row in fglcm)
    #fglcm_sum = sum(sum(row) for row in fglcm)
    #TODO: change the way to normalize
    fglcm_norm = fglcm/(fglcm.shape[0] * fglcm.shape[1])
    return fglcm_norm[np.nonzero(fglcm_norm)]
def fglcm_energy(fglcm_norm):
    return np.sum(list(np.sum(row ** 2) for row in fglcm_norm))
def fglcm_entropy(fglcm_norm):
    return np.sum(list(np.sum(-np.multiply(row, np.log(row))) for row in fglcm_norm))
