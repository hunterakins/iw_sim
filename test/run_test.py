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
from iw_sim import helpers


def get_poisson_seg_deltas(num_els, DeltaZ, kappa0):
    """
    Divide segment of size DeltaZ into num_els
    Poisson elements
    """
    deltaz = DeltaZ / num_els
    deltas = np.zeros(num_els+1)
    for i in range(num_els-1):
        deltas[i+1] = deltaz*np.random.gamma(kappa0, 1/kappa0)
    while np.sum(deltas) > DeltaZ:
        for i in range(num_els-1):
            deltas[i+1] = deltaz*np.random.gamma(kappa0, 1/kappa0)
    deltas[-1] = DeltaZ - np.sum(deltas)
    return deltas



DeltaZ = 20.0
kappa0 = 2.0
num_els = 10
deltas = get_poisson_seg_deltas(num_els, DeltaZ, kappa0)
zmin = 100.0
zmax = 300.0

def add_poisson_finestructure(z_pert, DeltaZ, kappa0, num_els, zmin, zmax):
    """
    z_pert is the perturbation of a background mesh
    unperturbed isopycnal mesh with spacing DeltaZ
    due to internal waves
    make a new mesh with finer spacing and poisson structure
    in the range zmin, zmax
    """
    z_fine = np.zeros((z_pert.shape[0], (z_pert.shape[1] - 1)*num_els + 1, z_pert.shape[2]))
    for i in range(z_pert.shape[1]-1): # for each depth
        if z_pert[0, i, 0] >= zmin and z_pert[0, i+1, 0] <= zmax:
            dZ  = z_pert[:, i+1, :] - z_pert[:, i, :]
            deltas = get_poisson_seg_deltas(num_els, DeltaZ, kappa0)
            strain_rescale = dZ / DeltaZ
            deltas = deltas[None, :, None] * strain_rescale[:, None, :]
            z_unif = z_pert[:, i, :][:, None, :] + np.cumsum(deltas, axis=1)[:, :-1, :]
            z_fine[:, i*num_els:(i+1)*num_els, :] = z_unif
        else:
            dZ  = z_pert[:, i+1, :] - z_pert[:, i, :]
            dz = dZ / num_els
            for xi in range(z_pert.shape[0]):
                for ti in range(z_pert.shape[2]):
                    z_fine[xi, i*num_els:(i+1)*num_els, ti] = z_pert[xi, i, ti] + np.linspace(0.0, dZ[xi, ti], num_els+1)[:-1]
    z_fine[:, -1, :] = z_pert[:, -1, :]
    return z_fine
        

#z_pert = np.arange(0, 100, 20.0)
#z_pert = z_pert[None, :, None]
#DeltaZ = z_pert[0, 1, 0] - z_pert[0, 0, 0]
#z_fine = add_poisson_finestructure(z_pert, DeltaZ, kappa0, num_els, 40, 60)

fnames = ['mfn7.txt', 'prof336.txt', 'medsea.txt']
deltaZ = 20.0
deltaz = 2.0 # for poisson segment
for fname in fnames:
    x = np.loadtxt(fname, skiprows=2)

    #x = np.loadtxt('medsea.txt', skiprows=1)
    z = x[:,0]
    temp = x[:,1]
    sal = x[:,2]
    c = x[:,3]
    bv_cph = x[:,4]




    latitude = 14.0

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
    Nkx = 5
    Nky = Nkx
    dk_cpkm = 0.007
    kmax = Nkx * dk_cpkm #
    dt = 100
    Nt = 1
    xmax = 100*1e3
    dx_des = 1000
    J = 20
    jstar = 3.0
    E0 = 4.0
    field = iw_field.IWField(disper)
    field.add_GM_params(jstar, E0)
    field.add_field_params(dk_cpkm, Nkx, Nky, dt, Nt, xmax, dx_des, J)
    #x, zeta_xzt = field.gen_zeta_field()
    x, z, t, zeta_xzt, u_xzt, v_xzt, w_xzt = field.gen_zuvw_field()

    #iso_mesh, pert_mesh = helpers.get_isopycnal_zpert(deltaZ, z, zeta_xzt)
    #z_fine = add_poisson_finestructure(pert_mesh, DeltaZ, kappa0, num_els, zmin, zmax)

    plt.figure()
    tmp = zeta_xzt[0,:,0]
    kz_tmp = np.fft.fftshift(np.fft.fftfreq(tmp.size, z[1] - z[0]))

    """
    plt.figure()
    for l in range(x.size):
        for kk in range(iso_mesh.size):
            plt.plot(x, pert_mesh[:,kk, 0], 'k')
            plt.plot(x, iso_mesh[kk]*np.ones(x.size), 'r')
        for kk in range(z_fine.shape[1]):
            plt.plot(x, z_fine[:,kk, 0], 'b', alpha=0.5)
    plt.gca().invert_yaxis()
    plt.ylabel('z (m)')
    plt.xlabel('x (m)')
    """

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
        plt.pcolormesh(x, z, obj)
        plt.colorbar()
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.gca().invert_yaxis()
        plt.suptitle(label)



    dzeta_dz = np.gradient(zeta_xzt, z, axis=0)
    plt.figure()
    plt.pcolormesh(x, z, dzeta_dz)
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

    plt.figure()
    plt.suptitle('Compare zeta depth variance')
    zz, zzeta_sq = field.get_zeta_depth_variance()
    plt.plot(np.sqrt(zzeta_sq), zz)
    plt.plot(zeta_xzt[:,0], z)
    plt.gca().invert_yaxis()

    plt.figure()
    plt.suptitle('Compare w depth variance')
    zz, ww_sq = field.get_w_depth_variance()
    plt.plot(np.sqrt(ww_sq), zz)
    plt.plot(w_xzt[:,0], z)
    plt.gca().invert_yaxis()

    t_lags = np.array([0, 1000, 2000])
    zz, ww_cov = field.get_w_depth_cov(t_lags)
    plt.figure()
    for lag_i in range(t_lags.size):
        plt.plot(ww_cov[:,lag_i], zz, label='t = ' + str(t_lags[lag_i]))

    plt.legend()
    plt.gca().invert_yaxis()
    plt.ylabel('z (m)')


    plt.show()



