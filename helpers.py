"""
Description:

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from numba import jit

@jit
def get_pkj(k_radpm, j, latitude, J, jstar, Hj_norm, BN0, E0=4.0):
    """
    Garret-Munk variance (equation 5 in notes from Richard Evans)
    latitude in degrees
    j is mode number 
    k is horizontal mode number (in radians / m)
    E0 - energy density factor (4.0 seems typical but can vary)
    """
    F_I = 1/12 * np.sin(latitude * np.pi / 180.) # in cph

    kj = (np.pi * F_I / (BN0)) *j
    Bkj  = 4 / np.pi * (kj*k_radpm**2) / ((kj**2 + k_radpm**2))**2 # this function integrates to 0 from k = -infty to infty

    Hj = (1/(j**2 + jstar**2)) 
    Hj /= Hj_norm

    pkj = E0 * Hj * Bkj
    return pkj
