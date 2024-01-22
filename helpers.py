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
    Hj_norm - normalization factor for mode weighting
    """
    F_I = 1/12 * np.sin(latitude * np.pi / 180.) # in cph

    kj = (np.pi * F_I / (BN0)) *j
    Bkj  = 4 / np.pi * (kj*k_radpm**2) / ((kj**2 + k_radpm**2))**2 # this function integrates to 0 from k = -infty to infty

    Hj = (1/(j**2 + jstar**2)) 
    Hj /= Hj_norm

    pkj = E0 * Hj * Bkj
    return pkj


def get_isopycnal_zpert(delta_z, z, zeta_xzt):
    """
    delta_z - vertical distance between isopycnals (m)
    z - mesh grid at which the displacement field has been computed
    zeta_xzt - displacement field (m)
        first axis is range (or x), second is depth, third is time

    z_pert - depth of isopycnals in the displacement field
        same size as zeta_xzt
    """
    N = int((z[-1] - z[0])/delta_z) + 1
    isopycnal_mesh = np.linspace(z[0], z[-1], N)
    delta_z = isopycnal_mesh[1] - isopycnal_mesh[0]
    z_pert = np.zeros((zeta_xzt.shape[0], isopycnal_mesh.shape[0], zeta_xzt.shape[2]))
    num_x = zeta_xzt.shape[0]
    num_t = zeta_xzt.shape[2]
    for i in range(num_x):
        for j in range(num_t):
            zeta = zeta_xzt[i,:,j]
            disps = np.interp(isopycnal_mesh, z, zeta)
            z_pert[i,:,j] = isopycnal_mesh + disps
    return isopycnal_mesh, z_pert
    
