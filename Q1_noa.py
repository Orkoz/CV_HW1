# importing libraries
import copy
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

def classify_vgg16(net, folder, label):
    # Create a dictionary with all the posibble classifications
    classes_dict = dict()  # prepare dictionary
    with open("classes.txt", "r") as f:
        for line in f:  # for each line in your file
            (key, val) = line.strip().split(':')
            classes_dict[key] = val

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
    # scaler = transforms.Resize((224, 224))
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # to_tensor = transforms.ToTensor()
    # return torch.autograd.Variable(normalize(to_tensor(scaler(im))).unsqueeze(0))

def get_features_vector(model, modules_key, layer_idx, image, out_features_size):
    """
    This func. returns the output features vector of the specified layer for the input image
    :param model: the pretrained net
    :param modules_key: 'features' or 'classifier'
    :param layer_idx: index of the layer within modules_key
    :param image: the image to extract net features on (already in torch.autograd.Variable form, which is what
     the nn expects)
    :param out_features_size: tuple, the size of the features the layer outputs
    :return: the features vector
    """
    # Use the model object to select the desired layer
    layer = model._modules.get(modules_key).__getitem__(layer_idx)
    # Create a vector of zeros that will hold our feature vector
    my_embedding = torch.zeros(out_features_size)

    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.view_as(my_embedding))

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # Run the model on our transformed image
    model(image)
    # Detach our copy function from the layer
    h.remove()
    # Return the feature vector
    return my_embedding

def section2():
    birds = [Image.open('birds/bird_0.jpg'), Image.open('birds/bird_1.jpg')]
    plt.figure(figsize=(15, 15)), plt.tight_layout()
    plt.subplot(121), plt.imshow(birds[0]), plt.title('bird #0'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(birds[1]), plt.title('bird #1'), plt.xticks([]), plt.yticks([])
    plt.suptitle('Original Birds Images', fontsize=40)
    plt.show(block=False)
    plt.figure(figsize=(15, 15)), plt.tight_layout()
    plt.subplot(121), plt.imshow(transform_image_to_fit_vgg(birds[0])), plt.title('bird #0'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(transform_image_to_fit_vgg(birds[1])), plt.title('bird #1'), plt.xticks([]), plt.yticks([])
    plt.suptitle('Normalized and Resized Birds Images', fontsize=40)
    plt.show(block=False)

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
    blurred_lizard_PIL = Image.fromarray(blurred_lizard)

    # saving transformed images to new folder
    my_path = os.path.join(os.getcwd(), 'transformed_imgs')
    if not os.path.exists(my_path):
        os.makedirs(my_path)
    rotate_lizard.save('transformed_imgs/rotate_lizard.jpg')
    colored_lizard.save('transformed_imgs/colored_lizard.jpg')
    blurred_lizard_PIL.save('transformed_imgs/blurred_lizard.jpg')

    # displaying the image and the transforms
    plt.figure(figsize=(15, 15)), plt.tight_layout()
    plt.subplot(141), plt.imshow(lizard), plt.title('Lizard'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(rotate_lizard), plt.title('Rotated Lizard'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(colored_lizard), plt.title('Jitter Colored Lizard'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(blurred_lizard), plt.title('Gaussian Blurred Lizard'), plt.xticks([]), plt.yticks([])
    plt.show(block=False)

def section6(net):
    """

    :param net: the nn
    :return:
        fc7_vecs (20x4096 matrix, each line is a features vector of a different image)
        reduced_features (20x2 matrix, each line is the 2 primary pca components of the features vector)
        tags (vector containing tags of the images, 0 for cat, 1 for dog)
    """
    fc7_vecs = []

    # iterate over the cats images and extract the FC7 features vector
    data_loader, label_idx = load_data(os.getcwd(), 'cats')
    for i, (inputs, labels) in enumerate(data_loader):
        # running the inout image through the network
        input_var = torch.autograd.Variable(inputs, volatile=True)
        label_var = torch.autograd.Variable(labels, volatile=True)
        if int(label_var) is not label_idx:
            continue
        features = torch.autograd.Variable(get_features_vector(net, 'classifier', 3, input_var,
                                                               ([4096]))).data.cpu().numpy()
        fc7_vecs.append(features)

    # iterate over the dogs images and extract the FC7 features vector
    data_loader, label_idx = load_data(os.getcwd(), 'dogs')
    for i, (inputs, labels) in enumerate(data_loader):
        # running the inout image through the network
        input_var = torch.autograd.Variable(inputs, volatile=True)
        label_var = torch.autograd.Variable(labels, volatile=True)
        if int(label_var) is not label_idx:
            continue
        features = torch.autograd.Variable(get_features_vector(net, 'classifier', 3, input_var,
                                                               ([4096]))).data.cpu().numpy()
        fc7_vecs.append(features)
    # getting 2 components using PCA
    tags = np.zeros(20)
    tags[10:19] = 1
    reduced_features = PCA(n_components=2).fit(fc7_vecs, tags).transform(fc7_vecs)

    # displaying the components for cats and dogs
    figure = plt.figure()
    plt.title('FC7 Feature Vec. of Cats and Dogs')
    plt.scatter(reduced_features[tags == 0, 0], reduced_features[tags == 0, 1], marker=".", c="r", label='cats')
    plt.scatter(reduced_features[tags == 1, 0], reduced_features[tags == 1, 1], marker=".", c="b", label='dogs')
    figure.legend()
    plt.show()

    return fc7_vecs, reduced_features, tags


def main():
    # create the pretrained nn
    net = models.vgg16(pretrained=True)
    net.eval()

    # Q1.2
    # section2()
    # classify_vgg16(os.getcwd(), 'birds')
    # # # Q1.3
    # classify_vgg16(os.getcwd(), 'lizards')
    # # Q1.4
    # section4()
    # classify_vgg16(os.getcwd(), 'transformed_imgs')
    # Q1.6
    section6(net)


if __name__ == '__main__':
    main()
