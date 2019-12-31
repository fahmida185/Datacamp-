#
# Compute Ontario Summer School
# Neural Networks with Python I
# 26 June 2019
# Erik Spence
#
# This file, model1.py, contains a routine to generate the model for
# the first Keras example.
#

#######################################################################


"""
model1.py contains a routine which generates the model for the class'
first Keras example.

"""


#######################################################################


import keras.models as km
import keras.layers as kl


#######################################################################


def get_model(numnodes, input_size = 784, output_size = 10):

    """
    This function returns a simple Keras model, consisting of a
    re-implementation of the second_network.py neural network, with
    numnodes in the hidden layer.

    Inputs:
    - numnodes: int, the number of nodes in the hidden layer.

    - intput_size: int, the size of the input data, default = 784.

    - output_size: int, the number of nodes in the output layer,
      default = 10.

    Output: the constructed Keras model.

    """

    # Initialize the model.
    model = km.Sequential()

    # Add a hidden fully-connected layer.
    model.add(kl.Dense(numnodes,
                       input_dim = input_size,
                       activation = 'sigmoid'))

    # Add the output layer.
    model.add(kl.Dense(output_size, activation = 'sigmoid'))

    # Return the model.
    return model


#######################################################################
