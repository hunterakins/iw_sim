"""
Description:
    Test solver on prof336.txt

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from iw_sim import iw_dispersion as iw_disper
from iw_sim import iw_field as iw_field

fnames = ['mfn7.txt', 'prof336.txt', 'medsea.txt']
for fname in fnames:
    x = np.loadtxt(fname, skiprows=2)

    #x = np.loadtxt('medsea.txt', skiprows=1)
    z = x[:,0]
    temp = x[:,1]
    sal = x[:,2]
    c = x[:,3]
    bv_cph = x[:,4]




    latitude = 50.0

    plt.figure()
    plt.plot(bv_cph, z)
    plt.gca().invert_yaxis()
    plt.xlabel('Brunt-Vaisala Frequency (cph)')
    plt.ylabel('Depth (m)')

    disper = iw_disper.IWDispersion(z, bv_cph, latitude)
    kmin = 1e-3
    kmax = 0.7
    Nk = 10
    kgrid_cpkm=np.linspace(kmin, kmax, Nk)
    J = 20
    mesh_dz = 1.0
    sav_dz = 1.0
    disper.set_sim_params(kgrid_cpkm, J, mesh_dz, sav_dz)
    disper.compute_omega_phi_arr()
    plt.figure()
    plt.suptitle(fname + ' dispersion relation')
    plt.plot(kgrid_cpkm, disper.omega_arr.T / (2*np.pi) * 3600, 'ko')
    plt.xlabel('k (cpkm)')
    plt.ylabel('Frequency (cph)')

    dzeta_dz = np.gradient(disper.phi_arr, disper.zgrid_sav, axis=0)
    dzeta_dz = np.max(np.abs(dzeta_dz), axis=0)
    plt.figure()
    plt.plot(kgrid_cpkm, dzeta_dz.T, 'ko')
    plt.xlabel('k (cpkm)')
    plt.ylabel('max |dzeta/dz| (mode)')


    """ now compute displacement"""
    Nkx = 50
    Nky = Nkx
    dk_cpkm = 0.007
    kmax = Nkx * dk_cpkm #
    dt = 100
    Nt = 1
    xmax = 50*1e3
    dx_des = 500
    J = 20
    jstar = 1.8
    E0 = 4.0
    field = iw_field.IWField(disper)
    field.add_GM_params(jstar, E0)
    field.add_field_params(dk_cpkm, Nkx, Nky, dt, Nt, xmax, dx_des, J)
    #x, zeta_xzt = field.gen_zeta_field()
    x, zeta_xzt, u_xzt, v_xzt, w_xzt = field.gen_zuvw_field()
    zeta_xzt = np.squeeze(zeta_xzt).T
    u_xzt = np.squeeze(u_xzt).T
    v_xzt = np.squeeze(v_xzt).T
    w_xzt = np.squeeze(w_xzt).T
    objs = [zeta_xzt, u_xzt, v_xzt, w_xzt]
    labels = ['zeta', 'u', 'v', 'w']


    for i in range(4):
        obj = objs[i]
        label = labels[i]
        plt.figure()
        plt.pcolormesh(x, disper.zgrid_sav, obj)
        plt.colorbar()
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.gca().invert_yaxis()
        plt.suptitle(label)

    plt.show()


    dzeta_dz = np.gradient(zeta_xzt, disper.zgrid_sav, axis=0)
    plt.figure()
    plt.pcolormesh(x, disper.zgrid_sav, dzeta_dz)
    plt.colorbar()
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.gca().invert_yaxis()
    #x, incoh_zeta_xzt = displ.get_disp_corr_field()
    #incoh_zeta_xzt = np.squeeze(incoh_zeta_xzt)
    #for i in range(3):
    #    zi = 10*i
    #    plt.figure()
    #    plt.pcolormesh(x, disper.zgrid_sav, incoh_zeta_xzt[:,zi,:].T)
    #    plt.colorbar()
    #    plt.xlabel('x (m)')
    #    plt.ylabel('z (m)')
    #    plt.gca().invert_yaxis()
    plt.show()

plt.show()


