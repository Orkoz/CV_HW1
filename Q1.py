# importing libraries

import imageio
from matplotlib import pyplot as plt
import os
from PIL import Image
from skimage import io

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2


def tranforamImageToFitVGG(im):
    # dynamic_range = np.power(2,im.bits)
    # R, G, B = cv2.split(np.array(im))
    #
    # R_mean = np.mean(R)/dynamic_range
    # G_mean = np.mean(G)/dynamic_range
    # B_mean = np.mean(B)/dynamic_range
    #
    # R_std = np.std(R)/dynamic_range
    # G_std = np.std(G)/dynamic_range
    # B_std = np.std(B)/dynamic_range
    # normalize = transforms.Normalize(mean=[R_mean, G_mean, B_mean], std=[R_std, G_std, B_std])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    unloader = transforms.ToPILImage()
    compose = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    return unloader(compose(im))

# ################# section 2 ##########################
birds = []

birds.append(Image.open('birds/bird_0.jpg'))
birds.append(Image.open('birds/bird_1.jpg'))

plt.figure(figsize=(15, 15)),plt.tight_layout()
plt.subplot(121), plt.imshow(birds[0]), plt.title('bird #0'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(birds[1]), plt.title('bird #1'), plt.xticks([]), plt.yticks([])
plt.suptitle('Original Birds Images', fontsize=40)
plt.show()

plt.figure(figsize=(15, 15)), plt.tight_layout()
plt.subplot(121), plt.imshow(tranforamImageToFitVGG(birds[0])), plt.title('bird #0'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(tranforamImageToFitVGG(birds[1])), plt.title('bird #1'), plt.xticks([]), plt.yticks([])
plt.suptitle('Original Birds Images', fontsize=40)
plt.show()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.getcwd(), transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])))
model = models.vgg16(pretrained=True)
model.features = torch.nn.DataParallel(model.features)
model.eval()

for i, (input, target) in enumerate(val_loader):
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)
    output = model(input_var)

# ################# section 4 ##########################

lizard = Image.open('lizard_wet_task1_section3.jpg')

# transformation operators
rotate = transforms.RandomRotation(90, resample=Image.BICUBIC)  # geometric transformation
colors = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
gaussian = cv2.getGaussianKernel(9, -1)


# applying transformation to lizard image
rotate_lizard = rotate(lizard)
colored_lizard = colors(lizard)
blurred_lizard = cv2.filter2D(np.array(lizard), -1, gaussian)
# blurred_lizard = cv2.GaussianBlur(lizard, 9, 3)

plt.figure(figsize=(15, 15)), plt.tight_layout()
plt.subplot(141), plt.imshow(lizard), plt.title('Lizard'), plt.xticks([]), plt.yticks([])
plt.subplot(142), plt.imshow(rotate_lizard), plt.title('Rotated Lizard'), plt.xticks([]), plt.yticks([])
plt.subplot(143), plt.imshow(colored_lizard), plt.title('Jitter Colored Lizard'), plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(blurred_lizard), plt.title('Gaussian Blurred Lizard'), plt.xticks([]), plt.yticks([])
# plt.suptitle('Lizard transformation', fontsize=40)
plt.show()
