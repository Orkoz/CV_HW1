# importing libraries
import os
from matplotlib import pyplot as plt
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from PIL import Image
import torch.optim
import cv2
from sklearn.decomposition import PCA
from skimage import feature
from scipy.ndimage import filters
import imageio



def sectionA():
    images = [imageio.imread('edges_images_GT/Church.jpg'), imageio.imread('edges_images_GT/Golf.jpg'), imageio.imread('edges_images_GT/Nuns.jpg')]

    sobel_ksize = [1, 3, 5, 0]
    canny_thresholds = [50, 100, 150, 200]
    laplace_parms = [1, 3, 5, 0]

    parms = [canny_thresholds, laplace_parms, sobel_ksize]
    opertors = ['canny', 'gaussian laplace', 'sobel']

    for image in images:
        for j in range(0, 3):
            plt.figure(), plt.tight_layout()
            plt.subplot(2, 3, 2), plt.imshow(image, cmap='gray')
            plt.title('original image'), plt.xticks([]), plt.yticks([])
            for i in range(0, 3):
                im = apply_filter(image, opertors[j], [parms[j][i], parms[j][i+1]], True)
                plt.subplot(2, 3, i+4), plt.imshow(im, cmap='gray')
                plt.title(opertors[j]+' with '+str(parms[j][i])), plt.xticks([]), plt.yticks([])
            plt.suptitle(opertors[j], fontsize=20)
            plt.show()


def apply_filter(image, filter, parms, adaptive=False):
    if filter == 'canny':
        # return feature.canny(image, parms[2], parms[0], parms[1])
        return cv2.Canny(image, parms[0], parms[1])
    elif filter == 'gaussian laplace':
        image_edges = filters.gaussian_laplace(image, parms[0])
        if adaptive:
            return cv2.adaptiveThreshold(image_edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
        else:
            _, th = cv2.threshold(image_edges, parms[1], 255, cv2.THRESH_BINARY_INV)
            return th
    elif filter == 'sobel':
        image_edges = cv2.Sobel(image, -1, 1, 1, ksize=parms[0])
        if adaptive:
            return cv2.adaptiveThreshold(image_edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
        else:
            _, th = cv2.threshold(image_edges, parms[1], 255, cv2.THRESH_BINARY_INV)
            return th


def main():
    sectionA()


if __name__ == '__main__':
    main()