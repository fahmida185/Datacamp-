#
# Compute Ontario Summer School
# Neural Networks with Python I
# 26 June 2019
# Erik Spence
#
# This file, example2.py, contains a routine to generate the data
# for the second example.
#

#######################################################################


"""
example2.py contains a routine for generating the data for the class'
second example.

"""


#######################################################################


import sklearn.datasets as skd
import sklearn.model_selection as skms


#######################################################################


def get_data(n):

    """This functon generates n data points generated using
    scikit-learn's make_circles function.  The data is then split into
    training and testing data sets, and returned.

    Inputs:
    - n: int, the number of points to return.

    Outputs: two arrays of size 0.8 * n, 0.2 * n, randomly generated
    using scikit-learn's make_circles routine.

    """

    # Generate the data.
    pos, value = skd.make_circles(n, noise = 0.1)

    # Split the data into training and testing data sets, and return.
    return skms.train_test_split(pos, value,
                                test_size = 0.2)


#######################################################################
