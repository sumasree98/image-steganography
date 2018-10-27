import math
import cv2
import numpy as np 
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim

#from encryption import img
#from decryption import decr_img

img = cv2.imread("dataset/plane.tiff", 0)
decr_img = cv2.imread("decrypted_img.bmp", 0)

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err
def psnr(mean_se):
    psn_ratio = 10*(math.log10((255**2)/mean_se))
    return psn_ratio

m = mse(img, decr_img)
s = ssim(img, decr_img)
psn_ratio = psnr(m)

print "Mean square error: %.2f" % round(m,2)
print "SSIM: %.2f" % round(s,2)
print "PSNR: %.2f" % round(psn_ratio,2)
