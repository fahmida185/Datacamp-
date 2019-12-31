#
# Compute Ontario Summer School
# Machine Learning in Python
# 13 June 2018
# Erik Spence
#
# This file, regression.py, contains functions which generate the
# artificial data used for the regression part of the class.  It also
# contains a function which generates the error as a function of
# polynomial order.
#

#######################################################################


import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


#######################################################################


# The base function to which noise is added.
def my_tanh(x):
    return np.tanh(8 * x) - x


#######################################################################


# Function which generates a noisy version of the base function.
def noisy_data(num = 50):

    x = np.linspace(-1, 1, num)
    x += 0.1 * npr.rand(num)

    y = my_tanh(x)
    y += 0.125 * npr.randn(num)

    return x, y


#######################################################################


def calc_fit_errors():

    orders = np.arange(1, 21)
    errors = np.zeros(20)

    x2 = np.linspace(-1, 1, 50)
    
    for order in orders:

        for i in range(10):
            
            x, y = noisy_data()        
            p = np.polyfit(x, y, order)
            fit = np.poly1d(p)

            errors[order - 1] += np.sum((my_tanh(x2) - fit(x2))**2)

    plt.semilogy(orders, errors, 'ko-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Fit Error')
    plt.tight_layout()
    plt.savefig('images/error_vs_degree.pdf')
        
    return orders, errors


#######################################################################
