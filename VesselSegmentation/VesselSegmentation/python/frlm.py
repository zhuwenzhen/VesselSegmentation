import numpy as np
#degree = [[1, 0], [1, 1], [0, 1], [-1, 1]]

def frlm(img):
    x_sz = img.shape()[1]
    y_sz = img.shape()[0]
    coord_mat = array([(x, y) for y in range(x_sz) for x in range(y_sz)]).reshape(img.shape())
    runs_135 = np.array([coord_mat.diagonal(i) for i in range(-y_sz+1, x_sz)])
    runs_45 = np.array([np.flipud(coord_mat).diagonal(i) for i in range(-y_sz+1, x_sz)])
    runs_0 = coord_mat.copy()
    runs_90 = coord_mat.transpose().copy()
    all_runs = [runs_135, runs_45, runs_0, runs_90]
    all_rlms = map(lambda runs:rlm(img, runs),all_runs)
    return all_rlms

def rlm(img, runs):
    frl_mat = np.zeros(img.shape())
    #if(angle_mode == "deg"):
    #    degree = math.radians(degree)
    #TODO: figure out the loop
    # idx_y = 0
    # while idx_y < img.shape()[0]:
    #     new_y = idx_y
    #     while new_x >= 0 && new_x < img.shape()[1] && new_y >= 0 && new_y < img.shape()[0]:
    #         #compute new_x new_y
    #         x = new_x
    #         y = new_y
    #         new_x = x + degree[0]
    #         new_y = y + degree[1]
    #         if img[new_y][new_x] == img[y][x]:
    #             run+=1
    #         else:
    #             frl_mat[img[y][x]][run] += 1
    #             run = 1
    #     idx_y += 1
    for single_run in runs:
        x, y = single_run[0]
        run = 0
        for new_x, new_y in single_run:
            if img[new_y][new_x] == img[y][x]:
                run+=1
            else:
                frl_mat[img[y][x]][run] += 1
                run = 1
            x = new_x
            y = new_y
        frl_mat[img[y, x][run] += 1
    return frl_mat
def compute_sre(frl_mat):
    sre_up = sum(sum(col/i**2) for i, col in enumerate(frl_mat.transpose()))
    sre_down = sum(sum(row) for row in frl_mat)
    return sre_up / sre_down
