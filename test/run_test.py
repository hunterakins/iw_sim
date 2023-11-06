"""
Description:
    Test solver on prof336.txt

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from iw_sim import iw_dispersion as iw_disp

fnames = ['prof336.txt', 'medsea.txt'][::-1]
for fname in fnames:
    x = np.loadtxt(fname, skiprows=2)

    #x = np.loadtxt('medsea.txt', skiprows=1)
    z = x[:,0]
    temp = x[:,1]
    sal = x[:,2]
    cond = x[:,3]
    bv_cph = x[:,4]




    latitude = 50.0

    plt.figure()
    plt.plot(bv_cph, z)
    plt.gca().invert_yaxis()
    plt.xlabel('Brunt-Vaisala Frequency (cph)')
    plt.ylabel('Depth (m)')

    disp = iw_disp.IWDispersion(z, bv_cph, latitude)
    kmin = 1e-3
    kmax = 0.5
    Nk = 20
    kgrid_cpkm=np.linspace(kmin, kmax, Nk)
    J = 20
    mesh_dz = 2.0
    disp.set_sim_params(kgrid_cpkm, J, mesh_dz)
    disp.compute_omega_phi_arr()

    print(disp.omega_arr / (2*np.pi) * 3600)
    plt.figure()
    plt.suptitle(fname + ' dispersion relation')
    plt.plot(kgrid_cpkm, disp.omega_arr.T / (2*np.pi) * 3600, 'ko')
    plt.xlabel('k (cpkm)')
    plt.ylabel('Frequency (cph)')

    dzeta_dz = np.gradient(disp.phi_arr, disp.zgrid, axis=0)
    dzeta_dz = np.max(np.abs(dzeta_dz), axis=0)
    plt.figure()
    plt.plot(kgrid_cpkm, dzeta_dz.T, 'ko')
    plt.xlabel('k (cpkm)')
    plt.ylabel('max |dzeta/dz|')
plt.show()

