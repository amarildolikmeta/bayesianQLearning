import numpy as np

class DensityFunction(object):
    
    def __init__(self,h=1, K=None):
        self.h=h
        if K is None:
            self.K=lambda x:0.5*(1/np.pi)*np.exp(-0.5*x**2)
        else:
            self.K=K
    def q(self, x):
        return (1/self.h)*self.K(x/self.h)
    
