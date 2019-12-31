#
# SciNet's DAT112, Neural Network Programming.
# Lecture 12, 6 June 2019.
# Erik Spence.
#
# This file, my_style_transfer.py, contains code used for lecture 12.
# It is a script which performs 'style transfer' between the two input
# images, and outputs the resulting image.
#
# This code is heavily based on the Keras example:
# https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
#

########################################################################    


import imageio as io
import numpy as np
import scipy.optimize as sco
import time
import argparse

import keras.preprocessing.image as kpi
import keras.applications.vgg19 as kav19
from keras import backend as K


########################################################################


parser = argparse.ArgumentParser(description = 'Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar = 'base', type = str,
                    help = 'Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar = 'ref', type = str,
                    help = 'Path to the style reference image.')
parser.add_argument('result_prefix', metavar = 'res_prefix', type = str,
                    help = 'Prefix for the saved results.')
parser.add_argument('--iter', type = int, default = 200, required=False,
                    help = 'Number of iterations to run.')
parser.add_argument('--content_weight', type = float, default = 0.025,
                    required = False,
                    help = 'Content weight.')
parser.add_argument('--style_weight', type = float, default = 1.0,
                    required = False,
                    help = 'Style weight.')
parser.add_argument('--tv_weight', type = float, default = 1.0,
                    required = False,
                    help = 'Total Variation weight.')


########################################################################    


# Function to open, resize and format pictures into appropriate
# tensors
def preprocess_image(image_path, nrows, ncols):

    img = kpi.load_img(image_path, target_size = (nrows, ncols))
    img = kpi.img_to_array(img)

    # The image comes as (height, width, 3).  This next line changes
    # it to (1, height, width, 3).
    img = np.expand_dims(img, axis = 0)

    # Image preprocessing is needed, to keep the image consistent with
    # input requirements of the specific model.
    img = kav19.preprocess_input(img)
    return img


########################################################################


# Function to convert a tensor into a valid image.  This is used to
# create images which are in a format which can be saved.
def deprocess_image(x, nrows, ncols):

    x = x.reshape((nrows, ncols, 3))

    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')

    return x


########################################################################


# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):

    # Move the channels to the first index, then flatten the 2D
    # feature maps.  Meaning if the original size is (10, 10, 64), the
    # indices are switched to (64, 10, 10), and then flattened to
    # become size (64, 100).
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

    # We then calculate the correlation between all the individual
    # feature maps.
    gram = K.dot(features, K.transpose(features))
    return gram


########################################################################


# the "style loss" is designed to maintain the style of the reference
# image in the generated image.  It is based on the gram matrices
# (which capture style) of feature maps from the style reference image
# and from the generated image.
def style_loss(style, combination):

    nrows, ncols, channels = style.shape
    
    S = gram_matrix(style)
    C = gram_matrix(combination)

    size = nrows * ncols
    return K.sum(K.square(S - C)) / ((2. * channels * size) ** 2)


########################################################################


# The loss function designed to maintain the "content" of the base
# image in the generated image
def content_loss(base, combination):
    return K.sum(K.square(combination - base))


########################################################################


def eval_loss_and_grads(x):

    # x comes in flattened, so put it back.
    x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


########################################################################    

# This Evaluator class makes it possible to compute loss and gradients
# in one pass while retrieving them via two separate functions, "loss"
# and "grads". This is done because scipy.optimize requires separate
# functions for loss and gradients, but computing them separately
# would be inefficient.

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        return self.grad_values


########################################################################    


# Get the arguments.
args = parser.parse_args()

# Assign the arguments to values.
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter

# these are the weights of the different loss components
style_weight = args.style_weight
content_weight = args.content_weight

print("Content weight is ", content_weight)

# dimensions of the generated picture.
width, height = kpi.load_img(base_image_path).size
img_nrows = height
img_ncols = width


# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path,
                                         img_nrows, img_ncols))
style_image = K.variable(preprocess_image(style_reference_image_path,
                                          img_nrows, img_ncols))

# this will contain our generated image
combo_image = K.placeholder((1, img_nrows, img_ncols, 3))

# Combine the 3 images into a single Keras tensor, of size
# (3, nrows, ncols, 3).
input_tensor = K.concatenate([base_image,
                              style_image,
                              combo_image], axis = 0)

# build the VGG19 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
model = kav19.VGG19(input_tensor = input_tensor,
                    weights = 'imagenet', include_top = False)
print('Model loaded.')

# Get the symbolic outputs of each "key" layer (they have unique
# names).
outputs_dict = {layer.name: layer.output for layer in model.layers}

# combine these loss functions into a single scalar
layer_features = outputs_dict['block5_conv2']
base_features = layer_features[0, :, :, :]
combo_features = layer_features[2, :, :, :]

loss = content_weight * content_loss(base_features,
                                     combo_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

for layer in feature_layers:
    layer_features = outputs_dict[layer]
    style_features = layer_features[1, :, :, :]
    combo_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combo_features)
    loss += style_weight * sl


# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combo_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

# Create a Keras function that will be used to evaluate the graph, as
# needed by scipy.
f_outputs = K.function([combo_image], outputs)


# Create an instance of the evaluator.
evaluator = Evaluator()

# Run scipy-based optimization (L-BFGS) over the pixels of the
# generated image so as to minimize the neural style loss.

# Start with the original image as the output.
x = preprocess_image(base_image_path, img_nrows, img_ncols)

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = sco.fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)

    # save current generated image
    img = deprocess_image(x.copy(), img_nrows, img_ncols)
    fname = result_prefix + '_at_iteration_%d.png' % i
    io.imwrite(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    
