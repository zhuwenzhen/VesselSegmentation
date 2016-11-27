import scipy.interpolate
import math
import numpy as np
#scipy.interpolate.interp2d(x_coords, y_coords, img)
def fglcm_check_point(img, x, y, d, degree, m, n, interp_func, angle_mode="rad"):
    if(angle_mode == "deg"):
        degree = math.radians(degree)
    if(round(img[y][x]) == m):
        new_x = x + d * cos(degree)
        new_y = y + d * sin(degree)
        if(round(interp_func(new_x, new_y)) == n):
            return True
    return False
def glcm(img, degree, d, interp_func, angle_mode="rad"):
    glc_mat = np.zeros(img.shape())
    for m in range(256):
        for n in range(256):
            for x in range(img.shape()[1] - d):
                for y in range(img.shape()[0] - d):
                    glc_mat[m][n] += fglcm_check_point(img, x, y, d, degree, m, n, interp_func, angle_mode)
    return glc_mat

def fglcm(img, d):
    x_coords = list(range(0, img.shape()[1]))
    y_coords = list(range(0, img.shape()[0]))
    f = scipy.interpolate.interp2d(x_coords, y_coords, img)
    degrees = [0, 45, 90, 135]
    glc_mats = []
    for deg in degrees:
        glc_mat = glcm(img, deg, d, f, angle_mode="deg")
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
