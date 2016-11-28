import scipy.interpolate
import math
import numpy as np
#scipy.interpolate.interp2d(x_coords, y_coords, img)
#degree = [[1, 0], [1, 1], [0, 1], [-1, 1]]
def fglcm_check_point(img, x, y, d, degree, m, n):
    #if(angle_mode == "deg"):
    #    degree = math.radians(degree)
    if(round(img[y][x]) == m):
        new_x = x + (d * degree)[0]
        new_y = y + (d * degree)[1]
        if(round(img[new_y, new_x]) == n):
            return True
    return False
def glcm(img, degree, d):
    glc_mat = np.zeros(img.shape())
    for m in range(256):
        for n in range(256):
            for x in range(img.shape()[1] - d):
                for y in range(img.shape()[0] - d):
                    glc_mat[m][n] += fglcm_check_point(img, x, y, d, degree, m, n)
    return glc_mat

def fglcm(img, d):
    x_coords = list(range(0, img.shape()[1]))
    y_coords = list(range(0, img.shape()[0]))
    degrees = np.array([[1, 0], [1, 1], [0, 1], [-1, 1]])
    glc_mats = []
    for deg in degrees:
        glc_mat = glcm(img, deg, d)
        glc_mats.append(glc_mat)
    fglcm = sum(glc_mats) / len(glc_mats)
    #max_ele = max(max(row) for row in fglcm)
    #min_ele = max(max(row) for row in fglcm)
    #fglcm_sum = sum(sum(row) for row in fglcm)
    #TODO: change the way to normalize
    fglcm_norm = fglcm/(img.shape()[0] * img.shape()[1])
def fglcm_energy(fglcm_norm):
    return sum(sum(row ** 2) for row in fglcm_norm)
def fglcm_entropy(fglcm_norm):
    return sum(sum(-np.multiply(row, np.log(row))) for row in fglcm_norm)
