#
# Compute Ontario Summer School
# Machine Learning in Python
# 13 June 2018
# Erik Spence
#
# This file, crossvalidation.py, contains functions which run
# crossvalidation on a polynomial fit to some noisy data.  The
# functions plot the best result and save the figure.
#
# This code was stolen, almost wholesale, from L. Dursi.
#
#######################################################################


import numpy as np
import numpy.random as npr
import matplotlib.pylab as plt

import sklearn.model_selection as skms

# The module containing the function which generates the data.
import regression as reg


#######################################################################


def estimateError(x, y, d, kfolds = 10):

    """Estimate the error in fitting data (x,y) with a polynomial of
       degree d via cross-validation.  Returns RMS error."""

    # Hold the total error.
    err = 0.0

    # Loop over the indices of the KFold splits.
    for train, test in skms.KFold(n_splits = kfolds,
                                  shuffle = True).split(x):

        # Split the data into training and testing.
        test_x  = x[test]
        test_y  = y[test]
        train_x = x[train]
        train_y = y[train]

        # Fit to the data.
        p = np.polyfit(train_x, train_y, d)

        # Create the polynomial function.
        fit = np.poly1d(p)

        # Add up the squared difference.
        err += sum((test_y - fit(test_x))**2)

    # Return the RMS error.
    return np.sqrt(err)


#######################################################################


def chooseDegree(num, maxdegree = 20, filename = None):
    
    """Gets noisy data, uses cross validation to estimate error, and fits
    new data with best model.

    """

    # Generate some noisy data.
    x, y = reg.noisy_data(num)

    # The degrees to be examined.
    degrees = np.arange(maxdegree+1)

    # The variable to hold the errors.
    errs = np.zeros_like(degrees, dtype = np.float)

    # Loop over the degrees, and gather the errors.
    for i, d in enumerate(degrees):
        errs[i] = estimateError(x, y, d)

    # Get the best error value.
    bestindex = np.argmin(errs)
    bestdegree = degrees[bestindex]

    # Plot the errors versus degrees.
    plt.subplot(1,2,1)
    plt.plot(degrees, errs,'ko-')
    plt.xlabel("Degree")
    plt.ylabel("CV Error")
    
    plt.subplot(1,2,2)

    # Plot the data.
    plt.plot(x, y, 'ro')

    # Create the x values needed for the line.
    xs = np.linspace(min(x), max(x), 100)

    # Calculate the values for the line.
    fit = np.poly1d(np.polyfit(x, y, bestdegree))

    # And plot and save.
    plt.plot(xs, fit(xs),'g-', lw = 2)
    plt.xlim(-1, 1)
    plt.suptitle('Selected Degree ' + str(bestdegree))
    plt.tight_layout()
    plt.savefig(filename)

    
#######################################################################


if __name__ == "__main__":

    chooseDegree(50, filename = 'images/CV_polynomial.pdf')
