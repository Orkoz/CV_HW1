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

###############
###Functions###
###############
def imshow(inp, i, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(i)
    plt.imshow(inp), plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show(block=False)

def load_data(folder, label):
    """
    This function load images from the specified folder, which are under the specified label
    :param folder: the root dir where the image folders are located
    :param label: a string, the label of the images we want to use
    :return: data_loader (the data loader), label_idx (the label index)
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(folder, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])))
    label_idx = data_loader.dataset.class_to_idx[label]

    return data_loader, label_idx

def classify_vgg16(folder, label):
    # Create a dictionary with all the posibble classifications
    classes_dict = dict()  # prepare dictionary
    with open("classes.txt", "r") as f:
        for line in f:  # for each line in your file
            (key, val) = line.strip().split(':')
            classes_dict[key] = val

    # create the pretrained nn
    net = models.vgg16(pretrained=True)

    data_loader, label_idx = load_data(folder, label)

    # iterate over the input images and display them with the classification vgg16 gave them
    for i, (inputs, labels) in enumerate(data_loader):
        # running the inout image through the network
        input_var = torch.autograd.Variable(inputs, volatile=True)
        label_var = torch.autograd.Variable(labels, volatile=True)
        if int(label_var) is not label_idx:
            continue
        out = net(input_var)
        # out is a vector of 1000 energies. take the max of them to get the class index
        _, out_index = torch.max(out, 1)
        # convert the index into a string so we can find the classification in the classes_dictionary
        idx = [str(int(out_index)) for s in (str(out_index).split()) if s.isdigit()]
        output_classification = classes_dict[idx[0]][2:-2]
        # display the input image with its classification
        imshow(inputs[0], i, title=output_classification)

def transform_image_to_fit_vgg(im):
    """
    This function normalizes and resizes the input image to the format vgg16 expects
    :param im: input RGB image
    :return: the normalised and resized image
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    unloader = transforms.ToPILImage()
    compose = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    return unloader(compose(im))

def section2():
    birds = [Image.open('birds/bird_0.jpg'), Image.open('birds/bird_1.jpg')]
    plt.figure(figsize=(15, 15)), plt.tight_layout()
    plt.subplot(121), plt.imshow(birds[0]), plt.title('bird #0'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(birds[1]), plt.title('bird #1'), plt.xticks([]), plt.yticks([])
    plt.suptitle('Original Birds Images', fontsize=40)
    plt.show()
    plt.figure(figsize=(15, 15)), plt.tight_layout()
    plt.subplot(121), plt.imshow(transform_image_to_fit_vgg(birds[0])), plt.title('bird #0'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(transform_image_to_fit_vgg(birds[1])), plt.title('bird #1'), plt.xticks([]), plt.yticks([])
    plt.suptitle('Normalized and Resized Birds Images', fontsize=40)
    plt.show()

def section4():
    lizard = Image.open('lizards/lizard.jpg')

    # transformation operators
    rotate = transforms.RandomRotation(90, resample=Image.BICUBIC)  # geometric transformation
    colors = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
    gaussian = cv2.getGaussianKernel(9, -1)

    # applying transformation to lizard image
    rotate_lizard = rotate(lizard)
    colored_lizard = colors(lizard)
    blurred_lizard = cv2.filter2D(np.array(lizard), -1, gaussian)

    # saving transformed images to new folder
    my_path = os.path.join(os.getcwd(), 'transformed_imgs')
    if not os.path.exists(my_path):
        os.makedirs(my_path)
    rotate_lizard.save('transformed_imgs/rotate_lizard.jpg')
    colored_lizard.save('transformed_imgs/colored_lizard.jpg')
    blurred_lizard.save('transformed_imgs/blurred_lizard.jpg')

    # displaying the image and the transforms
    plt.figure(figsize=(15, 15)), plt.tight_layout()
    plt.subplot(141), plt.imshow(lizard), plt.title('Lizard'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(rotate_lizard), plt.title('Rotated Lizard'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(colored_lizard), plt.title('Jitter Colored Lizard'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(blurred_lizard), plt.title('Gaussian Blurred Lizard'), plt.xticks([]), plt.yticks([])
    plt.show()

def main():
    # Q1.2
    # section2()
    # classify_vgg16(os.getcwd(), 'birds')
    # # Q1.3
    # classify_vgg16(os.getcwd(), 'lizards')
    # Q1.4
    section4()
    classify_vgg16(os.getcwd(), 'transformed_imgs')

if __name__ == '__main__':
    main()
