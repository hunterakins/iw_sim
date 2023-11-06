"""
Description:

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from iw_sim.iw_dispersion import IWDispersion
from numba import njit, jit
from iw_sim.helpers import get_pkj

def load_iw_disp(key):
    with open(npy_folder + 'med_iw_disp_{}.pickle'.format(key), 'rb') as f:
        iw_disp = pickle.load(f)
    return iw_disp

def add_m_contrib(hlzt, k, l, Nt, tgrid, p_kj, omega_jlm, phi_jlm):
    """
    The kernel value at a fixed horizontal wavenumber l is comprised of a sum over m and over omega
    This function adds the contribution from a single m to the kernel value at a fixed l

    It adds a value for both positive m and negative m
    It adds a value for plus and minus omega for each of these...
    """
    variance = p_kj / 2 / (2*np.pi*k) # first 2 is for real and imag., 2pi k is for cartesian kx, ky
    Gplus_jlm = np.random.randn()*np.sqrt(variance) + 1j*np.random.randn()*np.sqrt(variance)
    Gminus_jlm = np.random.randn()*np.sqrt(variance) + 1j*np.random.randn()*np.sqrt(variance)
    Gplus_jlminus_m = np.random.randn()*np.sqrt(variance) + 1j*np.random.randn()*np.sqrt(variance)
    Gminus_jlminus_m = np.random.randn()*np.sqrt(variance) + 1j*np.random.randn()*np.sqrt(variance)
    for t_index in range(Nt):
        pos_m_contrib = (Gplus_jlm*np.exp(1j*omega_jlm*tgrid[t_index]) + Gminus_jlm*np.exp(-1j*omega_jlm*tgrid[t_index]))*phi_jlm
        neg_m_contrib = (Gplus_jlminus_m*np.exp(1j*omega_jlm*tgrid[t_index]) + Gminus_jlminus_m*np.exp(-1j*omega_jlm*tgrid[t_index]))*phi_jlm
        hlzt[l-1,:,t_index] += pos_m_contrib + neg_m_contrib
    return

def get_h_lzt(iw_disp_func, dk_radpm, dt, Nx, Ny, Nz, Nt,J, latitude, BN0):
    """
    Get the Fourier transform of a displacement field realization on the plane y=0
    evaluated at discrete wavenumbers kx = l \Delta k
    l = 1, \dots, N_x
    Input:
        dkx: float, wavenumber spacing in x (same in y) (cph)
        dt : desired time spacing (in seconds)
        Nx: int, number of grid points in positive x direction
        Ny: int, number of grid points in positive y direction
        Nt: int, number of time steps
        J: int, maximum mode number
        latitude: float, latitude in degrees

    NOTE: I DO NOT FACTOR IN THE APPROPRIATE SCALING FOR THE VARIANCES BASED ON DK,
    I HANDLE THAT IN AT THE END WHEN I DO THE FOURIER TRANSFORM OVER KX
    AS A RESULT THE FUNCTION RETURNED IS OFF FROM IT'S TRUE VALUE BY A FACTOR OF \SQRT(DK)
    (real h(l,z,t) = \sqrt(DK) h(l,z,t) returned here)
    """
    tgrid = np.linspace(0, dt*(Nt-1), Nt)

    jstar = 3
    Hj_norm = np.sum(1 / (np.linspace(1, J, J)**2 + jstar**2))

    hlzt = np.zeros((Nx, Nz, Nt), dtype=np.complex_)

    """
    A note on the summation:
        Here I sum over the upper triangle of the 2D wavenumber space, along with the diagonal
        Since the modes and frequencies depend only on the norm of the wavenumber
        For each l, I need a contribution from m= 1, \dots, N_y
        However, the contribution to (l, m) uses the same wavenumbers as the contribution to (m,l)
        (and therefore the same frequencies, modes, and variances)
    """
    for l in range(1, Nx+1):
        print('l', l)
        kx = dk_radpm*l
        for m in range(l, Ny+1):
            ky = dk_radpm*m
            k_radpm = np.sqrt(kx**2 + ky**2)
            k_cpkm = k_radpm * 1e3 /(2*np.pi) # convert to cpm
            omegas, phi_arr = iw_disp_func(k_cpkm)
            for j in range(J):
                mode_j = j+1
                p_kj = get_pkj(k_radpm, mode_j, latitude, J, jstar, Hj_norm, BN0) # variance of coeffs.
                omega_jlm = omegas[j]
                phi_jlm = phi_arr[:,j]
                add_m_contrib(hlzt, k_radpm, l, Nt, tgrid, p_kj, omega_jlm, phi_jlm)
                """
                Add another contribution for m and l switched (since modes and omega depend only on the norm...)
                """
                if l != m:
                    add_m_contrib(hlzt, k_radpm, m, Nt, tgrid, p_kj, omega_jlm, phi_jlm) 
    return hlzt

def get_zeta_xzt(dk, h_lzt, xmax, dx_des):
    """
    Take inverse Fourier transform of h_lzt
    Zero pad to get the desired resolution dx_des
    Truncate IFFT output to only include values up to xmax
    """
    Nkx = h_lzt.shape[0] # not counting l=0
    # add on the l= 0  term...
    h_lzt = np.concatenate((np.zeros((1, h_lzt.shape[1], h_lzt.shape[2]), dtype=np.complex_), h_lzt), axis=0)

    """ 
    zeros pad for finer r resolution
    """
    Xmax = 2*np.pi / dk
    print('Xmax', Xmax)
    kmax = h_lzt.shape[0] * dk
    dx = np.pi / kmax
    print('dx', dx)
    kmax_des = np.pi / dx_des
    factor = kmax_des / kmax
    print('factor', factor)
    factor = max(1, factor)
    des_Nkx = int(Nkx * factor)
    print('des Nkx', des_Nkx)
    zeros_to_add = des_Nkx - Nkx
    h_lzt = np.concatenate((h_lzt, (np.zeros((zeros_to_add, h_lzt.shape[1], h_lzt.shape[2]), dtype=np.complex_))), axis=0)
    Nx = 2*des_Nkx + 1
    x = np.linspace(0, Xmax, Nx)
    # take the IRFFT
    zeta_xzt = dk*Nx*np.fft.irfft(h_lzt, n=Nx,axis=0) # get rid of scaling
    trunc_inds = x <= xmax
    x = x[trunc_inds]
    zeta_xzt = zeta_xzt[trunc_inds,:,:]
    return x, zeta_xzt

def advect_ssp(z_arr, c, temp_C, sal, lat, lon, zeta_xz):
    """
    Given displacement field zeta_xz
    background profiles in z, c, temp_C and sal
    advect the parcels to new places
    """
    pressure=z_arr
    delta_c = np.zeros_like(zeta_xz)

    abs_sal = gsw.conversions.SA_from_SP(sal, pressure, lon, lat)
    cons_temp = gsw.conversions.CT_from_t(abs_sal, temp_C, pressure)
    atg = gsw.adiabatic_lapse_rate_from_CT(abs_sal, cons_temp, pressure)

    zprime = z_arr[:,None] + zeta_xz
    delta_t = atg[:,None] * zeta_xz # change in temperature due to adiabatic expansion
    for i in range(zeta_xz.shape[1]): # for each range

        plt.figure()
        plt.plot(z_arr, 'o')
        plt.plot(z_arr + zeta_xz[:,i], 'o')
        plt.show()

        temp_C_prime = np.interp(zprime[:,i], z_arr, temp_C)
        sal_prime = np.interp(zprime[:,i], z_arr, sal)
        temp_C_prime -= delta_t[:,i]
        c_prime = gsw.sound_speed_t_exact(sal_prime, temp_C_prime, pressure)
        delta_c [:,i] = c_prime - c
    return delta_c

def get_dc_realiz(key, realiz_ind):
    _,_,_,_, lat, lon = get_profiles.load_ctd(key)
    z_arr, b_sq, h, sigma0, omegaI = get_bsq(key)
    bv = np.sqrt(b_sq - omegaI**2)
    bv_cph = bv * 3600 / (2*np.pi)
    BN0 = np.trapz(bv_cph, z_arr)
    print('BN0', BN0)


    iw_disp = load_iw_disp(key)
    disp_func = iw_disp.make_disp_func()
    Nz = iw_disp.Nz
    dk_cpkm = 0.007
    dk_radpm = dk_cpkm * 2*np.pi / 1e3
    dt = 10
    Nx = 60
    Ny = 60
    Nt = 1
    J = iw_disp.J


    h_lzt = get_h_lzt(disp_func, dk_radpm, dt, Nx, Ny, Nz, Nt, J, lat, BN0)
    xmax = 50*1e3
    dx_des = 500
    x, zeta_xzt = get_zeta_xzt(dk_radpm, h_lzt, xmax, dx_des)
    zeta_xzt = np.squeeze(zeta_xzt)
    zeta_xzt = zeta_xzt.T

    depth_avg_E = np.mean(np.trapz(zeta_xzt**2 * b_sq[:,None], z_arr, axis=0))
    print('depth avg E', depth_avg_E)

    z, c, temp_C, sal, lat, lon = get_profiles.load_ctd(key)
    c = np.interp(z_arr, z, c)
    temp_C = np.interp(z_arr, z, temp_C)
    sal = np.interp(z_arr, z, sal)

    plt.figure()
    plt.pcolormesh(x, z_arr, zeta_xzt)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.savefig(pic_folder + 'zeta_xt_{0}_{1}.png'.format(key, realiz_ind))

    plt.figure()
    plt.pcolormesh(x, z_arr, np.gradient(zeta_xzt, z_arr, axis=0))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.savefig(pic_folder + 'dzetadz_xt_{0}_{1}.png'.format(key, realiz_ind))

    print(z_arr)
    delta_c = advect_ssp(z_arr, c, temp_C, sal, lat, lon, zeta_xzt)
    c = c[:,None] + delta_c
    plt.figure()
    plt.pcolormesh(x, z_arr, delta_c)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.savefig(pic_folder + 'dc_{0}_{1}.png'.format(key, realiz_ind))

    plt.figure()
    plt.pcolormesh(x, z_arr, c)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.savefig(pic_folder + 'c_{0}_{1}.png'.format(key, realiz_ind))


    save_dict = {'realiz_ind': realiz_ind, 'key': key, 'x': x, 'z_arr': z_arr, 'zeta_xzt': zeta_xzt, 'delta_c': delta_c, 'c': c}
    file_name = npy_folder + 'ssp_realiz_{0}_{1}.mat'.format(key, realiz_ind)
    io.savemat(file_name, save_dict)



    plt.show()

class IWDisplacement:
    def __init__(self, iw_disp):
        self.iw_disp = iw_disp

    def add_displacement_params(self, dk_cpkm,
                                      Nkx, Nky,
                                      dt, Nt, 
                                      xmax, dx_des, 
                                      J):
        """
        dk_cpkm - float
            spacing of the wavenumbers in cpkm
        Nkx - int
            number of grid points in kx
        Nky - int 
            number of grid points in ky
        dt - float
            time step in secodns
        Nt - int
            number of time steps
        xmax - float
            maximum distance in meters
        dx_des - float
            desired grid spacing in meters
        J - int
            number of modes to use
        """
        self.dk_cpkm = dk_cpkm
        self.Nkx = Nkx
        self.Nky = Nky
        self.dt = dt
        self.Nt = Nt
        self.xmax = xmax
        self.dx_des = dx_des
        self.J = J
        if self.J > self.iw_disp.J:
            raise ValueError('J must be less than or equal to {0}'.format(self.iw_disp.J))
        return

    def gen_disp_field(self):
        iw_disp = self.iw_disp
        disp_func = iw_disp.make_disp_func()
        dk_radpm = self.dk_cpkm * 2*np.pi / 1e3
        zgrid = iw_disp.zgrid
        bv_cph = np.interp(zgrid, iw_disp.bv_zgrid, iw_disp.bv_cph)
        BN0 = np.trapz(bv_cph, z_arr)
        lat = iw_disp.latitude
        h_lzt = get_h_lzt(disp_func, dk_radpm, self.dt, 
                                    self.Nkx, self.Nky, self.Nz, self.Nt, 
                                    self.J, lat, BN0)
        x, zeta_xzt = get_zeta_xzt(dk_radpm, h_lzt, xmax, dx_des)
