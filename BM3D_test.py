
import numpy as np 
import os
import cv2
#from skimage.io import imread
import matplotlib.pyplot as plt
import pybm3d
from skimage.measure import compare_psnr, compare_ssim

#def clip(X):
#    X = np.maximum(X,0)
#    X = np.minimum(X,255.0)
#    return X

ratio = 1
sigma = 12
list_psnr = []
list_ssim = []
test_dir = 'data/Test/Set68'
for im in os.listdir(test_dir):
    if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
        x = cv2.imread(os.path.join(test_dir,im),0)
        x = x.astype('float32')
        y = np.random.poisson(ratio*x)/(ratio)
        y=y.astype('float32')
        out = np.array(pybm3d.bm3d.bm3d(y, sigma))
        out = out
        list_psnr.append(compare_psnr(x/255.0,out/255.0))
        list_ssim.append(compare_ssim(x/255.0,out/255.0))
        plt.imshow(out,cmap='gray')
        print('%s psnr: %s, ssim: %s'%(im,compare_psnr(x/255.0,out/255.0),compare_ssim(x/255.0,out/255.0)))
mean_psnr = np.mean(list_psnr)
mean_ssim = np.mean(list_ssim)
print('psnr:{0}, ssim:{1}'.format(mean_psnr,mean_ssim))


#gnoise = np.random.normal(0, 25/255.0, x.shape)
#z = x+gnoise

#flag = 'p'
#if flag == 'p':
#    plt.imshow(y,cmap='gray')
#else:    
#    plt.imshow(z,cmap='gray')
#    
#
#print('normal_psnr: {0}'.format(compare_psnr(x,z)))
#print('poisson_psnr: {0}'.format(compare_psnr(x,y)))
#print('normal_ssim: {0}'.format(compare_ssim(x,z)))
#print('poisson_ssim: {0}'.format(compare_ssim(x,y)))

#outy = pybm3d.bm3d.bm3d(y, 25)





