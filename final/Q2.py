# importing libraries
import copy
import numpy as np
from skimage.filters import sobel
from skimage.feature import canny
import matplotlib.image as mpimg
from sklearn.metrics import precision_score, recall_score
from matplotlib import pyplot as plt
import cv2
from scipy.ndimage import filters
import imageio

def sectionA():
    images = [imageio.imread('edges_images_GT/Church.jpg'), imageio.imread('edges_images_GT/Golf.jpg'), imageio.imread('edges_images_GT/Nuns.jpg')]

    sobel_ksize = [1, 3, 5, 0]
    canny_thresholds = [50, 100, 150, 200]
    laplace_params = [1, 3, 5, 0]

    sobel__th = [50, 100, 150]
    canny__th = [50, 100, 150]
    laplace_th = [50, 100, 150]

    th = [canny__th, laplace_th, sobel__th]
    params = [canny_thresholds, laplace_params, sobel_ksize]
    operators = ['canny', 'LoG', 'sobel']

    for image in images:
        for j in range(0, 3):
            plt.figure(figsize=(15, 15)), plt.tight_layout()
            plt.subplot(5, 3, 2), plt.imshow(image, cmap='gray')
            plt.title('original image'), plt.xticks([]), plt.yticks([])
            for i in range(0, 3):
                im = apply_filter(image, operators[j], [params[j][i], params[j][i+1]], True)
                plt.subplot(5, 3, i+4), plt.imshow(im, cmap='gray')
                plt.title('Kernel size = ' + str(params[j][i]) + ' Threshold = adaptive'), plt.xticks([]), plt.yticks([])
                for z in range(0, 3):
                    im = apply_filter(image, operators[j], [params[j][i], th[j][z]])
                    plt.subplot(5, 3, i + 7 + z*3), plt.imshow(im, cmap='gray')
                    plt.title('Kernel size = ' + str(params[j][i]) + ' Threshold = ' + str(th[j][z])), plt.xticks([]), plt.yticks([])
            plt.suptitle(operators[j], fontsize=30)
            plt.show(block=False)


def apply_filter(image, filter, params, adaptive=False):
    if filter == 'canny':
        # return feature.canny(image, parms[2], parms[0], parms[1])
        # Canny(image, min_threshold, max_threshold) default ksize is 3
        return cv2.Canny(image, params[0], params[1])
    elif filter == 'LoG':
        # gaussian_laplace(image, sigma)
        image_edges = filters.gaussian_laplace(image, params[0])
        if adaptive:
            return cv2.adaptiveThreshold(image_edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
        else:
            _, th = cv2.threshold(image_edges, params[1], 255, cv2.THRESH_BINARY)
            return th
    elif filter == 'sobel':
        # Sobel(image, -1, 1, 1, ksize)
        image_edges = cv2.Sobel(image, -1, 1, 1, ksize=params[0])
        if adaptive:
            return cv2.adaptiveThreshold(image_edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
        else:
            _, th = cv2.threshold(image_edges, params[1], 255, cv2.THRESH_BINARY)
            return th

def my_filter(filt_name, thres, image):
    if filt_name == 'canny':
        sigma = 1.5
        low_thres = 0
        high_thres = thres
        binary_edges = canny(image, sigma=sigma, low_threshold=low_thres, high_threshold=high_thres)
    elif filt_name == 'sobel':
        edges = sobel(image)
        # edges = cv2.Sobel(image, -1, 1, 1, ksize=3)
        binary_edges = edges > thres
    elif filt_name == 'LoG':
        sigma = 1.5
        edges = filters.gaussian_laplace(image, sigma=sigma)
        binary_edges = edges > thres
    else:
        print("Illegal filter name!")
        exit(1)
    return binary_edges

def sectionB():
    images = [mpimg.imread('edges_images_GT/Church.jpg')/255, mpimg.imread('edges_images_GT/Golf.jpg')/255,
              mpimg.imread('edges_images_GT/Nuns.jpg')/255]
    gt_images = [mpimg.imread('edges_images_GT/Church_GT.bmp'), mpimg.imread('edges_images_GT/Golf_GT.bmp'),
                 mpimg.imread('edges_images_GT/Nuns_GT.bmp')]

    filts = ['LoG', 'sobel', 'canny']
    thresholds = np.arange(0, 1, 0.01)

    # list of filters, each filter contains a list of thresholds, each threshold contains 3 images
    precision = [[], [], []]
    recall = [[], [], []]

    for i, filt in enumerate(filts):
        for j, thres in enumerate(thresholds):
            precision[i].append([])
            recall[i].append([])
            for k, im in enumerate(images):
                bool_binary = my_filter(filt, thres, im)
                if (bool_binary.flatten().astype(int)).any():
                    prec = precision_score(y_true=gt_images[k].flatten(), y_pred=bool_binary.flatten().astype(int))
                    rec = recall_score(y_true=gt_images[k].flatten(), y_pred=bool_binary.flatten().astype(int))
                else:
                    prec = 0
                    rec = 0
                precision[i][j].append(prec)
                recall[i][j].append(rec)

    P = [[], [], []]
    R = [[], [], []]
    F = [[], [], []]
    for i, filt in enumerate(filts):
        for j, thres in enumerate(thresholds):
            p = np.mean(precision[i][j])
            P[i].append(p)
            r = np.mean(recall[i][j])
            R[i].append(r)
            if r == 0 or p == 0:
                F[i].append(0)
            else:
                F[i].append(2*((p*r)/(p+r)))

    plt.figure()
    for i, f in enumerate(F):
        plt.plot(thresholds, f, label=filts[i])
    plt.legend()
    plt.xlabel('Threshold'), plt.ylabel('F')
    plt.title('F vs. threshold')
    plt.show(block=False)


if __name__ == '__main__':
    sectionA()
    sectionB()
