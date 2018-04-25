# importing libraries
import os
import imageio
import PIL
from matplotlib import pyplot as plt
import torch
import torchvision
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from skimage import io
from PIL import Image
import torch.optim

###############
###Functions###
###############
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure()
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def main():
    # Create a dictionary with all the posibble classifications
    classes_dict = dict()  # prepare dictionary
    with open("classes.txt", "r") as f:
        for line in f:  # for each line in your file
            (key, val) = line.strip().split(':')
            classes_dict[key] = val

    # create the pretrained nn
    net = models.vgg16(pretrained=True)
    # load the images, resize and normalize them to the form vgg16 expects
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.getcwd(), transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])))

    # iterate over the input images and display them with the classification vgg16 gave them
    for i, (inputs, _) in enumerate(data_loader):
        # running the inout image through the network
        input_var = torch.autograd.Variable(inputs, volatile=True)
        out = net(input_var)
        # out is a vector of 1000 energies. take the max of them to get the class index
        _, out_index = torch.max(out, 1)
        # convert the index into a string so we can find the classification in the classes_dictionary
        idx = [str(int(out_index)) for s in (str(out_index).split()) if s.isdigit()]
        output_classification = classes_dict[idx[0]][2:-2]
        # display the input image with its classification
        imshow(inputs[0], title=output_classification)


if __name__ == '__main__':
    main()
