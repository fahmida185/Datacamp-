#
# Compute Ontario Summer School
# Neural Networks with Python I
# 26 June 2019
# Erik Spence
#
# This file, model4.py, contains a routine to generate the model for
# our MNIST Keras example, using dropout.
#

#######################################################################


"""
model4.py contains a routine which generates the model for the class'
MNIST Keras example, using dropout.

"""


#######################################################################


import keras.models as km
import keras.layers as kl


#######################################################################


def get_model(numnodes, input_size = 784, output_size = 10,
              d_rate = 0.4):

    """
    This function returns a simple Keras model, consisting of a
    re-implementation of the second_network.py neural network, with
    numnodes in the hidden layer, and dropout applied to the hidden
    layer.

    Inputs:
    - numnodes: int, the number of nodes in the hidden layer.

    - intput_size: int, the size of the input data, default = 784.

    - output_size: int, the number of nodes in the output layer,
      default = 10.

    - d_rate: float, fraction of nodes to be dropped out by the
      dropout procedure.

    Output: the constructed Keras model.

    """

    # Initialize the model.
    model = km.Sequential()

    # Add a hidden fully-connected layer.
    model.add(kl.Dense(numnodes, name = 'hidden',
                       input_dim = input_size,
                       activation = 'sigmoid'))

    # Add dropout to the hidden layer.
    model.add(kl.Dropout(d_rate, name = 'dropout'))

    # Add the output layer.
    model.add(kl.Dense(output_size, name = 'output',
                       activation = 'sigmoid'))

    # Return the model.
    return model


#######################################################################
