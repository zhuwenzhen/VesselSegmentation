import scipy.interpolate
import math
import numpy as np
import logging
import numba
import PIL
import skimage.feature

logger = logging.getLogger("main_logger")

"""
Wasted 10+ Hours. Assumed manhattan distance for d rather than euclidean distance.
I assumed manhattan distance makes more sense for an image. Actually euclidean distance is the correct_solution
"""

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
@numba.jit
def triangular(x, n, a):
    return max(0, 1 - abs(x-n)/a)
@numba.jit
def glcm_helper(img, glc_mat, x, y, d, degree):
    # new_x = x + (d * degree)[0]
    # new_y = y + (d * degree)[1]
    #print("new_x,new_y: ", (new_x, new_y))
    new_x = x + round(math.sin(degree))
    new_y = y + round(math.cos(degree))
    if new_x < 0 or new_x >= img.shape[1] or new_y < 0 or new_y >= img.shape[0]:
        return
    for i in range(-5, 6):
        for j in range(-5, 6):
            if img[y, x] + i < 0 or img[y, x] + i >= 256 or img[new_y, new_x] + j < 0 or img[new_y, new_x] + j >= 256:
                continue
            glc_mat[img[y, x] + i][img[new_y, new_x] + j] += min(triangular(img[y, x], img[y, x] + i, 6), triangular(img[new_y, new_x], img[new_y, new_x] + j, 6))
            #glc_mat[img[y, x] + i][img[new_y2, new_x] + j] += min(triangular(img[y, x], img[y, x] + i, 6), triangular(img[new_y2, new_x], img[new_y2, new_x] + j, 6))
@numba.jit
def glcm(img, degree, d):
    glc_mat = np.zeros((256, 256))
    # if degree[0] == 1 and degree[1] == 0:
    #     for x in range(img.shape[1] - (d * degree)[0]):
    #         for y in range(img.shape[0]):
    #             #print("x,y: ", (x, y))
    #             #print("new_x,new_y: ", (new_x, new_y))
    #             glcm_helper(img, glc_mat, x, y,d,degree)
    # elif degree[0] == 1 and degree[1] == 1:
    #     for x in range(img.shape[1] - (d * degree)[0]):
    #         for y in range(img.shape[0]  - (d * degree)[1]):
    #             glcm_helper(img, glc_mat, x, y,d,degree)
    # elif degree[0] == 0 and degree[1] == 1:
    #     for x in range(img.shape[1]):
    #         for y in range(img.shape[0]  - (d * degree)[1]):
    #             glcm_helper(img, glc_mat, x, y,d,degree)
    # elif degree[0] == -1 and degree[1] == 1:
    #     for x in range((d*degree)[0], img.shape[1]):
    #         for y in range(img.shape[0]- (d * degree)[1]):
    #             glcm_helper(img, glc_mat, x, y,d,degree)
    # elif degree[0] == -1 and degree[1] == 0:
    #     for x in range((d*degree)[0], img.shape[1]):
    #         for y in range(img.shape[0]):
    #             glcm_helper(img, glc_mat, x, y,d,degree)
    # elif degree[0] == -1 and degree[1] == -1:
    #     for x in range((d*degree)[0], img.shape[1]):
    #         for y in range((d*degree)[0], img.shape[0]):
    #             glcm_helper(img, glc_mat, x, y,d,degree)
    # elif degree[0] == 0 and degree[1] == -1:
    #     for x in range(img.shape[1]):
    #         for y in range((d*degree)[0], img.shape[0]):
    #             glcm_helper(img, glc_mat, x, y,d,degree)
    # elif degree[0] == 1 and degree[1] == -1:
    #     for x in range(img.shape[1] - (d * degree)[1]):
    #         for y in range((d*degree)[0], img.shape[0]):
    #             glcm_helper(img, glc_mat, x, y,d,degree)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            glcm_helper(img, glc_mat, x, y, d, degree)
    return glc_mat

def fglcm(img, d):
    #x_coords = list(range(0, img.shape[0]))
    #y_coords = list(range(0, img.shape[1]))
    #[1, 0], [1, 1], [0, 1], [-1, 1]
    #[1, 0], [1, 1], [0, 1], [-1, 1]
    #degrees = np.array([[0,1], [1, 1], [1, 0], [1, -1]])
    #, [-1, 0], [-1, -1], [0, -1], [1, -1]
    degrees = [0, 45, 90, 135]
    degrees_rad = list(map(lambda deg: math.radians(deg), degrees))

    glc_mats = []
    for deg in degrees_rad:
        logger.debug("computing deg: %s", deg)
        glc_mat = glcm(img, deg, d)
        glc_mats.append(glc_mat)
    #print(glc_mats[0])
    #PIL.Image.fromarray(glc_mats[0].astype(dtype="uint8")).save("glc_mat2.jpg")
    fglcm = sum(glc_mats) / len(glc_mats)
    #max_ele = max(max(row) for row in fglcm)
    #min_ele = max(max(row) for row in fglcm)
    #fglcm_sum = sum(sum(row) for row in fglcm)
    #TODO: change the way to normalize
    fglcm_norm = fglcm/np.sum(fglcm)#(fglcm.shape[0] * fglcm.shape[1])

    return fglcm_norm
def fglcm_energy(fglcm_norm):
    return np.sum(fglcm_norm ** 2)
def fglcm_entropy(fglcm_norm):
    result = 0
    for row in fglcm_norm:
        log_of_row = np.log(row)
        fixed_log_of_row = np.nan_to_num(log_of_row)
        row_entropy = np.sum(-np.multiply(row, fixed_log_of_row))
        result += row_entropy
    return result
"""
Due to my wrong intepretation before I have to use an external library
"""

# def fglcm(img, d):
#     degrees = [0, 45, 90, 135]
#     degrees_rad = list(map(lambda deg: math.radians(deg), degrees))
#     colms = skimage.feature.greycomatrix(img, [20], degrees_rad, symmetric=True, normed=True)
#     colm_mats = []
#     for deg_i in range(len(degrees)):
#         colm_mats.append(colms[:,:,0,deg_i])
#     fglc_mat = sum(colm_mats)/len(colm_mats)
#     return fglc_mat
