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
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn import svm

#############
# Functions #
#############
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure()
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
        imshow(inputs[0], title=output_classification)

def transform_image_to_fit_vgg(im):
    """
    This function normalizes and resizes the input image to the format vgg16 expects
    :param im: input RGB image
    :return: the normalised and resized image
    """
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # unloader = transforms.ToPILImage()
    # compose = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # return unloader(compose(im))
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    return torch.autograd.Variable(normalize(to_tensor(scaler(im))).unsqueeze(0))

def transform_image(im):
    """
    Normalize and Resize image
    :param im: PIL image
    :return: PIL image after normalization and resizing
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    unloader = transforms.ToPILImage()
    compose = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    return unloader(compose(im))

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
    plt.subplot(121), plt.imshow(transform_image(birds[0])), plt.title('bird #0'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(transform_image(birds[1])), plt.title('bird #1'), plt.xticks([]), plt.yticks([])
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

def section5(net):
    # extract the first layer from the net and the two first filers kernels
    filter_layer1 = net.features[0].weight.data
    first_kernel = filter_layer1[0]
    second_kernel = filter_layer1[1]

    # display the image
    plt.figure(figsize=(15, 15)), plt.tight_layout()
    plt.subplot(121), plt.imshow(get_kernels_display(first_kernel)), plt.title('First Filter'), plt.xticks(
        []), plt.yticks([])
    plt.subplot(122), plt.imshow(get_kernels_display(second_kernel)), plt.title('Rotated Lizard'), plt.xticks(
        []), plt.yticks([])
    plt.suptitle('First Layer Filter', fontsize=40)
    plt.show(block=False)

    # going trough all images, identifying the images we want and analyzing them
    plt.figure(figsize=(15, 15))
    data_loader, label_idx = load_data(os.getcwd(), 'transformed_imgs')
    j = 1
    # iterate over the transformed lizards images and extract the FC7 features vector
    for i, (inputs, labels) in enumerate(data_loader):
        # running the inout image through the network
        input_var = torch.autograd.Variable(inputs, volatile=True)
        label_var = torch.autograd.Variable(labels, volatile=True)
        if int(label_var) is not label_idx:
            continue

        # get the response feature vector of layer 1 to the input image and from that vector take the response of the
        #  first and the second filter
        layer1_response = get_features_vector(net, 'features', 0, input_var, ([1, 64, 224, 224]))
        layer1_response = torch.autograd.Variable(
            layer1_response[0]).data.cpu().numpy()  # casting the FloatTensor to numpy for displaying the images
        first_response = layer1_response[0]
        second_response = layer1_response[1]

        # display the responses
        plt.subplot(3, 2, j), plt.imshow(first_response, cmap='gray'), plt.title('First Filter Response'), plt.xticks(
            []), plt.yticks([])
        j = j + 1
        plt.subplot(3, 2, j), plt.imshow(second_response, cmap='gray'), plt.title('Second Filter Response'), plt.xticks(
            []), plt.yticks([])
        j = j + 1

    plt.suptitle('First Layer Response', fontsize=40)
    plt.show(block=False)


def get_kernels_display(kernel):
    """
    :param kernel: the kernel to get ready for display (a FloatTensor)
    :return: uint8 numpy.array in range [0 255]
    """
    kernel = kernel.numpy()
    kernel = ((kernel - kernel.min()) * 255) / (kernel.max() - kernel.min())
    return kernel.astype(np.uint8)

def extract_fc7_cats_dogs(net):
    """
    For every image in dogs and for every image in cats extract the feature vector of layer FC7.
    Plot all the vectors on the same graph.
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
    plt.show(block=False)

    return fc7_vecs, reduced_features, tags

def svm_classify_cats_dogs(net, fc7_vecs, tags, image1, image2):
    """
    Returns 0 if image is classified as cat by SVM classifier, 1 if dog.
    Also displayes the image and its classification
    :param net: the nn
    :param fc7_vecs: training set, matrix of features vectors to train the SVM classifier with
    :param tags: labels of the training set
    :param image1: PIL image
    :param image2: PIL image
    :return: nothing, display images and their classifications
    """

    # train SVM classifier on cats and dogs database
    clf = svm.SVC()
    clf.fit(fc7_vecs, tags)

    # test classifier on new image1
    im1 = transform_image_to_fit_vgg(image1)
    features_vec1 = torch.autograd.Variable(get_features_vector(net, 'classifier', 3, im1, 4096)).data.cpu().numpy()
    classification1 = clf.predict(features_vec1.reshape(1, -1))

    # test classifier on new image2
    im2 = transform_image_to_fit_vgg(image2)
    features_vec2 = torch.autograd.Variable(get_features_vector(net, 'classifier', 3, im2, 4096)).data.cpu().numpy()
    classification2 = clf.predict(features_vec2.reshape(1, -1))

    # display classifications
    plt.figure()
    plt.suptitle("SVM Classification")
    # display class. of first image
    plt.subplot(1, 2, 1), plt.imshow(image1), plt.xticks([]), plt.yticks([])
    if classification1 == 0:
        plt.title("Cat")
    elif classification1 == 1:
        plt.title("Dog")
    # display class. of second image
    plt.subplot(1, 2, 2), plt.imshow(image2), plt.xticks([]), plt.yticks([])
    if classification2 == 0:
        plt.title("Cat")
    elif classification2 == 1:
        plt.title("Dog")
    plt.show(block=False)

def section7_8(net, features_mat):
    """
    :param features_mat:
    :param net: the VGG16 network.
    :param dogs_features_mat: the features of the 10 dogs from section 6 (10x4096 ndarray)
    :param cats_features_mat: the features of the 10 cats from section 6 (10x4096 ndarray)
    :return: none
    """
    ################### section 7 #######################3
    # load and display the image from the internet
    dog_and_cat = [Image.open('cat.jpg'), Image.open('dog.jpg')]

    plt.figure(figsize=(10, 10)), plt.tight_layout()
    plt.subplot(221), plt.imshow(dog_and_cat[0]), plt.title('Internet Cat'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(dog_and_cat[1]), plt.title('Internet Dog'), plt.xticks([]), plt.yticks([])

    # finding the nearest image from the evaluation set (represented as a row index from the feature matrix)
    nearest_cat = get_nearest_image_from_dataset(net, features_mat, dog_and_cat[0])
    nearest_dog = get_nearest_image_from_dataset(net, features_mat, dog_and_cat[0])

    nearest_dog_and_cat = [Image.open('cats/cat_' + str(nearest_cat) + '.jpg'),
                           Image.open('dogs/dog_' + str(nearest_dog) + '.jpg')]

    plt.subplot(223), plt.imshow(nearest_dog_and_cat[0]), plt.title(
        'nearest Cat image from the evaluation set'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(nearest_dog_and_cat[1]), plt.title(
        'nearest DOG image from the evaluation set'), plt.xticks([]), plt.yticks([])
    plt.suptitle('Images Nearest Neighbor')
    plt.show(block=False)

    ################### section 8 #######################3
    wolf = Image.open('wolf.jpg')
    tiger = Image.open('tiger.jpg')

    plt.figure(figsize=(10, 10)), plt.tight_layout()
    plt.subplot(221), plt.imshow(tiger), plt.title('Internet Wolf'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(wolf), plt.title('Internet Tiger'), plt.xticks([]), plt.yticks([])

    # finding the nearest image from the evaluation set (represented as a row index from the feature matrix)
    nearest_cat = get_nearest_image_from_dataset(net, features_mat, tiger)
    nearest_dog = get_nearest_image_from_dataset(net, features_mat, wolf)

    nearest_dog_and_cat = [Image.open('cats/cat_' + str(nearest_cat) + '.jpg'),
                           Image.open('dogs/dog_' + str(nearest_dog) + '.jpg')]

    plt.subplot(223), plt.imshow(nearest_dog_and_cat[0]), plt.title(
        'nearest Cat image from the evaluation set'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(nearest_dog_and_cat[1]), plt.title(
        'nearest DOG image from the evaluation set'), plt.xticks([]), plt.yticks([])
    plt.suptitle('Images Nearest Neighbor')
    plt.show(block=False)


def get_nearest_image_from_dataset(net, features_mat, src_image):
    # modify the internet images to fit the input format of the VGG net.
    vgg_image = transform_image_to_fit_vgg(src_image)

    # insert the images to the net and extract the second fully connected layer (while converting the result to ndarray)
    image_features = torch.autograd.Variable(
        get_features_vector(net, 'classifier', 3, vgg_image, ([4096]))).data.cpu().numpy()

    # finding the nearest image from the evaluation set (represented as a row index from the feature matrix)
    return find_nearest_neighbor(features_mat, image_features) % 9


def find_nearest_neighbor(src_mat, ref_vac):
    """
    :param src_mat: the vector space.
    :param ref_vac: the reference vectors
    :return: the index (row number) of the nearest vector to ref_vec from src_mat
    """
    norm = [distance.euclidean(x, ref_vac) for x in
            src_mat]  # calculate the L2 of the ref_vec from each of the vectors in src_mat
    min_list = min(norm)
    return norm.index(min_list)  # return the index (row number) of the nearest vector


def main():
    # create the pretrained nn
    net = models.vgg16(pretrained=True)
    net.eval()
    cwd = os.getcwd()

    # Q1.2
    section2()
    classify_vgg16(net, cwd, 'birds')
    # Q1.3
    classify_vgg16(net, cwd, 'lizards')
    # Q1.4
    section4()
    classify_vgg16(net, cwd, 'transformed_imgs')
    # Q1.5
    section5(net)
    classify_vgg16(net, cwd, 'transformed_imgs')
    # Q1.6
    fc7_vecs, _, tags = extract_fc7_cats_dogs(net)
    # Q1.7 & 8
    fc7_mat = np.asarray(fc7_vecs)
    section7_8(net, fc7_mat)
    # Q1.9
    im1 = Image.open('cat.jpg')
    im2 = Image.open('dog.jpg')
    svm_classify_cats_dogs(net, fc7_vecs, tags, im1, im2)
    # Q1.10
    im1 = Image.open('wolf.jpg')
    im2 = Image.open('tiger.jpg')
    svm_classify_cats_dogs(net, fc7_vecs, tags, im1, im2)

if __name__ == '__main__':
    main()
