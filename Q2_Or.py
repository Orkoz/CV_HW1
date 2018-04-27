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


def sectionA():
    image1 = Image.open('cat_10.jpg')
    image2 = Image.open('wolf.jpg')
    image3 = Image.open('tiger.jpg')

    sobel_ksize = [3, 5, 7]
    canny_thresholds = [50, 100, 150, 200]
    laplace_parms = [1, 3, 5]

    opertors = ['canny', 'gaussian laplace', 'sobel']

    for j in range(0, 4):
        plt.figure(figsize=(10, 10)), plt.tight_layout()
        for j in range(0, 4):
            plt.subplot(131), plt.imshow(apply_filter(image1,opertors[i],[]), plt.title('Internet Cat'), plt.xticks([]), plt.yticks([])
            plt.subplot(132), plt.imshow(dog_and_cat[1]), plt.title('Internet Dog'), plt.xticks([]), plt.yticks([])
        plt.suptitle(opertors[i])


def apply_filter(image, filter, parms, adaptive=False):
    if filter == 'canny':
        return cv2.Canny(image, parms[1], parms[2])
    elif filter == 'gaussian laplace':
        image_edges = filters.gaussian_laplace(img_gray, ksize=parms[1])
        if adaptive:
            return cv.adaptiveThreshold(image_edges, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 11, 2)
        else:
            _, th = cv.threshold(image_edges, parms[2], 255, cv.THRESH_BINARY)
            return th
    elif filter == 'sobel':
        image_edges = cv2.Sobel(img_gray, -1, 1, 1, ksize=parms[1])
        if adaptive:
            return cv.adaptiveThreshold(image_edges, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 11, 2)
        else:
            _, th = cv.threshold(image_edges, parms[2], 255, cv.THRESH_BINARY)
            return th

def main():

if __name__ == '__main__':
    main()