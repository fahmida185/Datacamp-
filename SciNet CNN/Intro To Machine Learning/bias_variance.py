#
# Compute Ontario Summer School
# Machine Learning in Python
# 13 June 2018
# Erik Spence
#
# This file, bias_variance.py, contains a script which will generate
# polynomial fits of a variety of orders, calculates the corresponding
# bias and variance for a particular point, plots the results and
# saves the figures.
#

#######################################################################


import numpy as np
import matplotlib.pylab as plt

# The module containing the function which generates the data.
import regression as reg


# Limits for the plots.
MIN = -1.0
MAX = 1.0

# The number of polynomial orders to consider.
num_degrees = 21

# The number of points in the plots.
n = 40

# The number of times we'll repeated generate the data and fit, to get
# statistics.
num_repeats = 1000

# The total errors.
tot_error = np.zeros(num_degrees)

# The location of x to use in the bias-variance analysis.
x_eval = 0.

# The exact
zerotrue = reg.my_tanh(x_eval)

# x values used for plotting the fit.
x2 = np.linspace(MIN, MAX, 100)


for degree in range(num_degrees):
 
    print("Processing degree", degree)

    # Create the file name for the image.
    f = 'images/bias_variance_' + str(degree) + '.pdf'
    
    plt.figure(figsize = (6, 3))
    plt.subplot(2,1,1)

    plt.title("Degree" + str(degree))

    # The vector of error values.
    errors = np.zeros(num_repeats)
    
    # repeat 1000 times
    for j in range(num_repeats):

        # Get new data.
        x, y = reg.noisy_data(n)

        # Fit to the new data.
        p = np.polyfit(x, y, degree)

        # Create a polynomial representation.
        fit = np.poly1d(p)
        
        # Calculate the bias.  How far are we from the correct answer?
        errors[j] = fit(0.) - zerotrue

        # Plot this particular fit.
        plt.plot(x2, fit(x2), 'g')

        # Add the error to the total.
        tot_error[degree] += sum((y - fit(x))**2)

        
    # Plot an example of the data.
    plt.plot(x, y, 'ko')
    plt.xlim((MIN, MAX))        

    # Switch to the lower plot.
    plt.subplot(2,1,2)

    # Plot the histogram of the errors, 20 bins.
    num, bins, patches = plt.hist(errors, int(20))

    # Calculate a good max height of the histogram.
    lheight = max(num) * 5 / 4

    # Add a line to the plot at the zerotrue location.
    line = plt.plot([zerotrue, zerotrue], [0, lheight], 'r-')
    plt.setp(line, linewidth = 2)

    # Get the mean, std and height of the histogram.
    mean    = np.mean(errors)
    sd      = np.sqrt(np.var(errors))
    datahi  = max(num)

    # Figure out where to put the line for the bias.
    if mean < 0:
        txtpos = mean - 2.4 * sd
        balign = 'left'
        valign = 'right'
    else:
        txtpos = mean + 2.4 * sd
        balign = 'right'
        valign = 'left'

    # Place the line.
    plt.annotate('Bias', xy = (mean, 0.9 * lheight), 
                 xytext = (zerotrue, 0.9 * lheight),
                 xycoords = 'data', ha = balign, va = 'center',
                 arrowprops = {'facecolor':'red', 'shrink':0.05})
    line = plt.plot([mean - 2 * sd, mean + 2 * sd],
                    [datahi / 3., datahi / 3.], 'g-')

    # Add the line for the variance.
    plt.setp(line, linewidth = 6, color = 'purple')
    plt.text(txtpos, datahi * 9./24., 'Variance', ha = valign, va = 'bottom')

    # Reset the x limits.
    plt.xlim((-0.3, 0.3))

    # Clean up and save the figure.
    plt.tight_layout(pad = 0.1)
    plt.savefig(f, transparent = True)
    plt.close()

    
#######################################################################


if __name__ == "__main__":

    # Plot the total error, and save the figure.
    plt.plot(np.arange(num_degrees), tot_error, 'ko-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('In-Sample Error')
    plt.tight_layout()
    plt.savefig('images/in_sample_error_vs_degree.pdf')
