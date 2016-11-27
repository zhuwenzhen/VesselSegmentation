import numpy as np
def frlm(img, degree, angle_mode="rad"):
    frl_mat = np.zeros(img.shape())
    if(angle_mode == "deg"):
        degree = math.radians(degree)
    x = 0
    y = 0
    new_x = 0
    new_y = 0
    run = 0
    if img[new_y][new_x] == img[y][x]:
        run+=1
    #TODO: figure out the loop
    frl_mat[img[y][x]][run] += 1

def compute_sre():
    pass
