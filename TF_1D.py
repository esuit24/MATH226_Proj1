'''
This module simulates the Thomas-Fermi approximation of the GPE in 1 dimension using unitless parameters
consistent with those defined in Bao, et. al, 2003 and Pethick & Smith. This module is to be used to compare
to the results of gpeclass.py
'''
# imports
import numpy as np
import math
from sklearn.preprocessing import normalize


# function definitions

# define mu as a function of delta in 1D
def mu(delta, eps = 1):
    '''
    Defines mu as a function of the interaction constant delta

    Parameters:
    delta - interaction strength constant (unitless)

    Returns:
    mu - the chemical potential (unitless)
    '''
    delta = delta / np.sqrt(eps)
    mu = 0.5 * (3/2 * delta)**(2/3)

    return mu

# define the thomas fermi wavefunction in 1D
def tf_wf(x, delta, eps, dx):
    '''
    Defines the Thomas Fermi wavefunction in 1D

    Parameters:
    x - the x range on which the wavefunction will be evaluated on
    delta - the interaction strength constant
    a0 - the harmonic oscillator length scale

    Returns:
    wf - the Thomas-Fermi wavefunction in 1D
    '''
    delta = delta * 1/np.sqrt(eps)

    num = (3/2 * delta)**(2/3) - (x/np.sqrt(eps))**2
    #print(num)
    den = 2*delta*np.sqrt(eps) * np.ones_like(num)
    #print(den)
    num[num<0] = 0

    wf = np.sqrt(num/den)

    # normalize
    wf_sqdx = wf * np.power(dx, 1/2)
    norm = np.sum(np.power(wf_sqdx, 2))
    sqdx_norm = wf_sqdx / norm
    wf_norm = sqdx_norm / np.power(dx, 1/2)
    return wf_norm

def tf_rad(delta, eps):
    '''
    Computes the Thomas-Fermi Radius
    '''
    rad = np.power(3*delta*1/np.sqrt(eps)/2, 1/3) * np.sqrt(eps)
    return rad

# def normalize_tf(wf, x):
#     '''
#     Parameters:
#     wf - TF wavefunction
#     x - x grid to evaluate the TF model on
#
#     Returns:
#     dens - the TF density
#     '''
#     dx = abs(x[1] - x[0])
#     wf_sqdx = wf * np.sqrt(dx)
#     sqdx_norm = normalize(wf_sqdx.reshape(1,-1)).ravel()
#     wf_norm = sqdx_norm/np.sqrt(dx)
#     dens = wf_norm**2
#
#     return dens
