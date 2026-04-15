"""
This file reuses most of the code from https://github.com/mlu355/MetadataNorm  
"""

import numpy as np
import scipy.stats as st


def generate_data(N, seed=0, scale=0):    
    np.random.seed(seed)

    # confounding effects between 2 groups
    cf = np.zeros((N*2,))
    cf[:N] = np.random.uniform(3, 5,size=N)
    cf[N:] = np.random.uniform(4, 6,size=N)
    cf[:N] -=  scale
    cf[N:] += scale

    # major effects between 2 groups
    np.random.seed(seed+1)
    mf = np.zeros((N*2,))
    mf[:N] = np.random.uniform(3, 5,size=N) 
    mf[N:] = np.random.uniform(4, 6,size=N)
    mf[:N] +=  scale
    mf[N:] -=  scale
    
    # simulate images
    d = int(32)
    dh = d//2
    y = np.zeros((N*2,)) 
    y[N:] = 1
    
    x = np.zeros((N*2,d,d,1))
    for i in range(N*2):
        x[i,:dh,:dh,0] = gkern(kernlen=d//2, nsig=5) * mf[i]
        x[i,dh:,:dh,0] = gkern(kernlen=d//2, nsig=5) * cf[i]
        x[i,dh:,dh:,0] = gkern(kernlen=d//2, nsig=5) * mf[i]
        x[i] = x[i] + np.random.normal(0,0.01,size=(d,d,1)) # random noise
    
    return cf, mf, x, y


def gkern(kernlen=21, nsig=5):
    """Returns a 2D Gaussian kernel array."""
    
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
