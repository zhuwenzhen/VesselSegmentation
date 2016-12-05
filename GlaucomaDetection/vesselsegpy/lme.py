import numpy as np
import numpy
import scipy.ndimage
import scipy.signal
import itertools
import PIL
l3 = np.array([[1, 2, 1]])
E3 = np.array([[-1, 0, 1]])
S3 = np.array([[-1, 2, -1]])

elementary_masks = [l3, E3, S3]
elementary_masks_pair = list(itertools.permutations(elementary_masks, 2))
#scipy.signal.convolve2d(k1, k2)
laws_masks = list(map(lambda k:scipy.signal.convolve2d(k[0].transpose(), k[1]), elementary_masks_pair))
#pillow_kernels = map(lambda mask: PIL.ImageFilter.Kernel((3,3), numpy.ndarray.flatten(mask)), laws_masks)
uniform_kernel7x7 = np.ones((7,7))/(7*7)

L3L3 = scipy.signal.convolve2d(l3.transpose(), l3)
#L3L3_kernel = PIL.ImageFilter.Kernel((3,3), numpy.ndarray.flatten(L3L3))
def compute_laws_mask_energy(img):
    img = img.astype(dtype=float)
    #tis = map(lambda kernel: img.filter(kernel),pillow_kernels)
    tis = list(map(lambda mask: scipy.ndimage.convolve(img, mask, mode="constant"),laws_masks))
    #debug_img = PIL.Image.fromarray(tis[0], mode="L")
    #debug_img.save("tis_0.jpg")
    ti_l3l3 = scipy.ndimage.convolve(img, L3L3, mode="constant")
    #debug_img = PIL.Image.fromarray(ti_l3l3.astype(dtype="uint8"), mode="L")
    #debug_img.save("ti_l3l3.jpg")
    tis_normalized = list(map(lambda ti: ti/ti_l3l3, tis))
    tis_normalized = np.nan_to_num(tis_normalized)
    #for i,ti in enumerate(tis):
    #    print(i," : ", np.isfinite(ti).all())
    #debug_img = PIL.Image.fromarray(tis_normalized[0].astype(dtype="uint8"), mode="L")
    #debug_img.save("tis_normalized_0.jpg")
    tis_normalized_abs = map(lambda ti:np.absolute(ti), tis_normalized)
    TEMs = list(map(lambda ti: scipy.ndimage.convolve(ti, uniform_kernel7x7, mode="constant"), tis_normalized_abs))
    #TODO: FINISH
    #debug_img = PIL.Image.fromarray(TEMs[0].astype(dtype="uint8"), mode="L")
    #debug_img.save("debug_tem_0.jpg")
    return TEMs
def compute_energy(img):
    return np.sum(img)
