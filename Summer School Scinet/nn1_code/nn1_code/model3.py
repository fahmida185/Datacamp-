#
# Compute Ontario Summer School
# Neural Networks with Python I
# 26 June 2019
# Erik Spence
#
# This file, model3.py, contains a routine to generate the model for
# class' regularized  Keras example.
#

#######################################################################



"""
model3.py contains a routine which generates the model for the class'
regularized MNIST Keras example.

"""


#######################################################################


import keras.models as km
import keras.layers as kl
import keras.regularizers as kr


#######################################################################


def get_model(numnodes, input_size = 784, output_size = 10,
              lam = 0.0):

    """
    This function returns a simple Keras model, consisting of a
    re-implementation of the second_network.py neural network, with
    numnodes in the hidden layer, and regularization implemented.

    Inputs:
    - numnodes: int, the number of nodes in the hidden layer.

    - intput_size: int, the size of the input data, default = 784.

    - output_size: int, the number of nodes in the output layer,
      default = 10.

    - lam: float, the value of the regularization parameter.

    Output: the constructed Keras model.

    """

    # Initialize the model.
    model = km.Sequential()

    # Add a hidden fully-connected layer.
    model.add(kl.Dense(numnodes, name = 'hidden',
                       input_dim = input_size,
                       activation = 'sigmoid',
                       kernel_regularizer = kr.l2(lam)))

    # Add the output layer.
    model.add(kl.Dense(output_size, activation = 'sigmoid',
                       name = 'output',
                       kernel_regularizer = kr.l2(lam)))

    # Return the model.
    return model


#######################################################################
