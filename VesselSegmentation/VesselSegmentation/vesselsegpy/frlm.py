import numpy as np
#degree = [[1, 0], [1, 1], [0, 1], [-1, 1]]

class XYHelper:
    def __init__(self, x, y):
        self.x = x
        self.y = y
def frlm(img):
    x_sz = img.shape[0]
    y_sz = img.shape[1]
    coord_mat = np.array([XYHelper(x, y) for y in range(y_sz) for x in range(x_sz)]).reshape(img.shape[0], img.shape[1])
    runs_135 = np.array([coord_mat.diagonal(i) for i in range(-y_sz+1, x_sz)])
    runs_45 = np.array([np.flipud(coord_mat).diagonal(i) for i in range(-y_sz+1, x_sz)])
    ru = filter(lambda r: len(r[1]) == 0, enumerate(runs_45))
    print(coord_mat)
    runs_0 = coord_mat.copy()
    runs_90 = coord_mat.transpose().copy()
    all_runs = [runs_45]
    #runs_0, runs_45, runs_90, runs_135
    all_rlms = map(lambda runs:rlm(img, runs),all_runs)
    return list(all_rlms)

def rlm(img, runs):
    frl_mat = np.zeros(img.shape)
    #if(angle_mode == "deg"):
    #    degree = math.radians(degree)
    #TODO: figure out the loop
    # idx_y = 0
    # while idx_y < img.shape[0]:
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
        x, y = (single_run[0].x,  single_run[0].y)
        run = 0
        for coord in single_run:
            if img[coord.y][coord.x] == img[y][x]:
                run+=1
            else:
                frl_mat[img[y][x]][run] += 1
                run = 1
            x = coord.x
            y = coord.y
        frl_mat[img[y, x]][run] += 1
    return frl_mat
def compute_sre(frl_mat):
    sre_up = np.sum(list(np.sum(col/i**2) for i, col in enumerate(frl_mat.transpose())))
    sre_down = np.sum(frl_mat)
    return sre_up / sre_down
