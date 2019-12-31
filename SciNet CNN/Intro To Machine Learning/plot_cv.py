
import numpy as np
from sklearn.base import BaseEstimator

class PolyRegression(BaseEstimator):

    def __init__(self, deg = 1):

        self.degree = deg
        print('the degree is ', self.degree)
        print('here0', type(self.degree))        
        #        self.fit(range(10), range(10))
        self.p = np.zeros(self.degree + 1)

        
    def fit(self, x, y):

        print('here', type(self.degree))
        self.p = np.polyfit(x, y, self.degree)
        self.fitfunc = np.poly1d(self.p)


    def predict(self, x):
        return [self.fitfunc(x0) for x0 in x]
    
    
    def score(self, x, y):
        return sum((self.fitfunc(x) - y)**2)

    def get_params(self, deep = False):
        return self.p        
    
    @property
    def coef_(self):
        return self.p
    
