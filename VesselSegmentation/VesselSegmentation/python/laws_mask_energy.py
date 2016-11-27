import numpy as np
import numpy.ndarray
import scipy.ndimage
import scipy.signal
import itertools
import PIL
l3 = np.array([[1, 2, 1]])
E3 = np.array([[-1, 0, 1]])
S3 = np.array([[-1, 2, -1]])

elementary_masks = [l3 E3 S3]
elementary_masks_pair = itertools.permutations(elementary_masks, 2)
laws_masks = map(lambda k1, k2: scipy.signal.convolve2d(k1, k2), elementary_masks_pair)
#pillow_kernels = map(lambda mask: PIL.ImageFilter.Kernel((3,3), numpy.ndarray.flatten(mask)), laws_masks)
uniform_kernel7x7 = np.ones((7,7))/(7*7)

L3L3 = scipy.signal.convolve2d(l3, l3)
#L3L3_kernel = PIL.ImageFilter.Kernel((3,3), numpy.ndarray.flatten(L3L3))


def compute_laws_mask_energy(img):
    #tis = map(lambda kernel: img.filter(kernel),pillow_kernels)
    tis = map(lambda mask: scipy.ndimage.convolve(img, mask, mode="constant"),laws_masks)
    ti_l3l3 = scipy.ndimage.convolve(img, L3L3_kernel, mode="constant")
    tis_normalized = map(lambda ti: ti/ti_l3l3, tis)
    tis_normalized_abs = map(lambda ti:np.absolute(ti), tis_normalized)
    TEMs = map(lambda ti: scipy.ndimage.convolve(ti, uniform_kernel7x7, mode="constant"), tis_normalized_abs)
    return TEMs
