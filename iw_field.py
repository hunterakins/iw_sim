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

def add_huvw_m_contrib(hlzt, ulzt, vlzt, wlzt, dk_radpm, k, l, m, Nt, tgrid, p_kj, omega_jlm, phi_jlm, phi_jlm_grad, omega_I):
    """
    The contribution for displacment, horizontal velocity, and verticl velocity
    """
    variance = p_kj / 2 / (2*np.pi*k) # first 2 is for real and imag., 2pi k is for cartesian kx, ky
    gplus_jlm = np.random.randn()*np.sqrt(variance) + 1j*np.random.randn()*np.sqrt(variance)
    gminus_jlm = np.random.randn()*np.sqrt(variance) + 1j*np.random.randn()*np.sqrt(variance)
    gplus_jlminus_m = np.random.randn()*np.sqrt(variance) + 1j*np.random.randn()*np.sqrt(variance)
    gminus_jlminus_m = np.random.randn()*np.sqrt(variance) + 1j*np.random.randn()*np.sqrt(variance)

    for t_index in range(Nt):
        pos_m_contrib = (gplus_jlm*np.exp(1j*omega_jlm*tgrid[t_index]) + gminus_jlm*np.exp(-1j*omega_jlm*tgrid[t_index]))*phi_jlm
        neg_m_contrib = (gplus_jlminus_m*np.exp(1j*omega_jlm*tgrid[t_index]) + gminus_jlminus_m*np.exp(-1j*omega_jlm*tgrid[t_index]))*phi_jlm
        hlzt[l-1,:,t_index] += pos_m_contrib + neg_m_contrib
       
        fact = l * omega_jlm + 1j * omega_I * m
        fact *= dk_radpm # to conver l and m to kx and ky
        fact /= k**2
        pos_m_contrib = (-fact.conj()*gplus_jlm*np.exp(1j*omega_jlm*tgrid[t_index]) + fact*gminus_jlm*np.exp(-1j*omega_jlm*tgrid[t_index]))*phi_jlm_grad
        neg_m_contrib = (-fact.conj()*gplus_jlminus_m*np.exp(1j*omega_jlm*tgrid[t_index]) + fact*gminus_jlminus_m*np.exp(-1j*omega_jlm*tgrid[t_index]))*phi_jlm_grad
        ulzt[l-1,:,t_index] += pos_m_contrib + neg_m_contrib


        fact = m * omega_jlm - 1j * omega_I * l
        fact *= dk_radpm
        fact /= k**2
        pos_m_contrib = (-fact*gplus_jlm*np.exp(1j*omega_jlm*tgrid[t_index]) + fact*gminus_jlm*np.exp(-1j*omega_jlm*tgrid[t_index]))*phi_jlm_grad
        neg_m_contrib = (-fact*gplus_jlminus_m*np.exp(1j*omega_jlm*tgrid[t_index]) + fact*gminus_jlminus_m*np.exp(-1j*omega_jlm*tgrid[t_index]))*phi_jlm_grad
        vlzt[l-1,:,t_index] += pos_m_contrib + neg_m_contrib

        wlzt[l-1,:,t_index] = hlzt[l-1, :, t_index] * 1j *omega_jlm 
    return 

def add_m_incoh_contrib(hlzt, k, l, Nt, tgrid, p_kj, omega_jlm, phi_jlm):
    variance = p_kj / (2*np.pi*k) # 2pi k is for cartesian kx, ky
    for t_index in range(Nt):
        pos_m_contrib = variance*phi_jlm*np.cos(omega_jlm * tgrid[t_index])
        neg_m_contrib = pos_m_contrib
        hlzt[l-1,:,:,t_index] += pos_m_contrib + neg_m_contrib
    return

def get_h_lzt(iw_disp_func, dk_radpm, dt, Nx, Ny, Nz, Nt,J, latitude, BN0, jstar, E0):
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
        kx = dk_radpm*l
        for m in range(l, Ny+1):
            ky = dk_radpm*m
            k_radpm = np.sqrt(kx**2 + ky**2)
            k_cpkm = k_radpm * 1e3 /(2*np.pi) # convert to cpm
            omegas, phi_arr = iw_disp_func(k_cpkm)
            for j in range(J):
                mode_j = j+1
                p_kj = get_pkj(k_radpm, mode_j, latitude, J, jstar, Hj_norm, BN0, E0) # variance of coeffs.
                omega_jlm = omegas[j]
                phi_jlm = phi_arr[:,j]
                add_m_contrib(hlzt, k_radpm, l, Nt, tgrid, p_kj, omega_jlm, phi_jlm)
                """
                Add another contribution for m and l switched (since modes and omega depend only on the norm...)
                """
                if l != m:
                    add_m_contrib(hlzt, k_radpm, m, Nt, tgrid, p_kj, omega_jlm, phi_jlm) 
    return hlzt

def get_incoh_h_lzt(iw_disp_func, dk_radpm, dt, Nx, Ny, Nz, Nt,J, latitude, BN0, jstar, E0, w=False, ww=False):
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
        BN0 : float
            integral of N^2 over the ocean depth in CPH
        jstar : int
            mode scale for GM
        E0 : float
            energy density factor (4.0 is typical). See Richard Evans notes
            for some details
        w : boolean
            if True, compute vertical velocity correlation

    NOTE: I DO NOT FACTOR IN THE APPROPRIATE SCALING FOR THE VARIANCES BASED ON DK,
    I HANDLE THAT IN AT THE END WHEN I DO THE FOURIER TRANSFORM OVER KX
    AS A RESULT THE FUNCTION RETURNED IS OFF FROM IT'S TRUE VALUE BY A FACTOR OF \SQRT(DK)
    (real h(l,z,t) = \sqrt(DK) h(l,z,t) returned here)
    w gives you the vertical displacement time derivative
    ww gives you the second displacement time derivative
    """
    tgrid = np.linspace(0, dt*(Nt-1), Nt)

    Hj_norm = np.sum(1 / (np.linspace(1, J, J)**2 + jstar**2))

    hlzt = np.zeros((Nx, Nz, Nz, Nt), dtype=np.complex_)

    """
    A note on the summation:
        Here I sum over the upper triangle of the 2D wavenumber space, along with the diagonal
        Since the modes and frequencies depend only on the norm of the wavenumber
        For each l, I need a contribution from m= 1, \dots, N_y
        However, the contribution to (l, m) uses the same wavenumbers as the contribution to (m,l)
        (and therefore the same frequencies, modes, and variances)
    """
    for l in range(1, Nx+1):
        #print('l', l)
        kx = dk_radpm*l
        for m in range(l, Ny+1):
            ky = dk_radpm*m
            k_radpm = np.sqrt(kx**2 + ky**2)
            k_cpkm = k_radpm * 1e3 /(2*np.pi) # convert to cpm
            omegas, phi_arr = iw_disp_func(k_cpkm)
            for j in range(J):
                mode_j = j+1
                p_kj = get_pkj(k_radpm, mode_j, latitude, J, jstar, Hj_norm, BN0, E0) # variance of coeffs.
                omega_jlm = omegas[j]
                phi_jlm = phi_arr[:,j]
                full_phi_jlm = np.outer(phi_jlm, phi_jlm)
                if w:# vertical velocity
                    full_phi_jlm *= omega_jlm**2
                elif ww:
                    full_phi_jlm *= omega_jlm**4
                add_m_incoh_contrib(hlzt, k_radpm, l, Nt, tgrid, p_kj, omega_jlm, full_phi_jlm)
                """
                Add another contribution for m and l switched (since modes and omega depend only on the norm...)
                """
                if l != m:
                    add_m_incoh_contrib(hlzt, k_radpm, m, Nt, tgrid, p_kj, omega_jlm, full_phi_jlm) 
    return hlzt

def get_huvw_lzt(iw_disp_func, z_arr, dk_radpm, dt, Nx, Ny, Nz, Nt,J, latitude, BN0, jstar, E0):
    """
    Get the Fourier transform of a displacement field realization on the plane y=0
    as well as the Fourier transform of the velocity field realization on the plane y=0
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
    fI = 1/12 * np.sin(latitude * np.pi / 180)
    omegaI = fI * 2 * np.pi / 3600

    Hj_norm = np.sum(1 / (np.linspace(1, J, J)**2 + jstar**2))

    hlzt = np.zeros((Nx, Nz, Nt), dtype=np.complex_)
    ulzt = np.zeros((Nx, Nz, Nt), dtype=np.complex_)
    vlzt = np.zeros((Nx, Nz, Nt), dtype=np.complex_)
    wlzt = np.zeros((Nx, Nz, Nt), dtype=np.complex_)

    """
    A note on the summation:
        Here I sum over the upper triangle of the 2D wavenumber space, along with the diagonal
        Since the modes and frequencies depend only on the norm of the wavenumber
        For each l, I need a contribution from m= 1, \dots, N_y
        However, the contribution to (l, m) uses the same wavenumbers as the contribution to (m,l)
        (and therefore the same frequencies, modes, and variances)
    """
    for l in range(1, Nx+1):
        #print('l', l)
        kx = dk_radpm*l
        for m in range(l, Ny+1):
            ky = dk_radpm*m
            k_radpm = np.sqrt(kx**2 + ky**2)
            k_cpkm = k_radpm * 1e3 /(2*np.pi) # convert to cpkm for iw_disp_func
            omegas, phi_arr = iw_disp_func(k_cpkm)
            for j in range(J):
                mode_j = j+1
                p_kj = get_pkj(k_radpm, mode_j, latitude, J, jstar, Hj_norm, BN0, E0) # variance of coeffs.
                omega_jlm = omegas[j]
                phi_jlm = phi_arr[:,j]
                phi_jlm_grad = np.gradient(phi_arr[:,j], z_arr)
                add_huvw_m_contrib(hlzt, ulzt, vlzt, wlzt, dk_radpm, k_radpm, l, m, Nt, tgrid, p_kj, omega_jlm, phi_jlm, phi_jlm_grad, omegaI)
                """
                Add another contribution for m and l switched (since modes and omega depend only on the norm...)
                """
                if l != m:
                    add_huvw_m_contrib(hlzt, ulzt, vlzt, wlzt, dk_radpm, k_radpm, m, l, Nt, tgrid, p_kj, omega_jlm, phi_jlm, phi_jlm_grad, omegaI)
    return hlzt, ulzt, vlzt, wlzt

def invert_lzt_to_xzt(dk, h_lzt, xmax, dx_des, verbose=False):
    """
    Take inverse Fourier transform of h_lzt
    Zero pad to get the desired resolution dx_des
    Truncate IFFT output to only include values up to xmax
    Take transform from l space to x space
    """
    Nkx = h_lzt.shape[0] # not counting l=0
    # add on the l= 0  term...
    zero_shape = list(h_lzt.shape[1:] )
    zero_shape = [1] + zero_shape
    zero_shape = tuple(zero_shape)
    h_lzt = np.concatenate((np.zeros((zero_shape), dtype=np.complex_), h_lzt), axis=0)

    """ 
    zeros pad for finer r resolution
    """
    Xmax = 2*np.pi / dk
    kmax = h_lzt.shape[0] * dk
    dx = np.pi / kmax
    kmax_des = np.pi / dx_des
    factor = kmax_des / kmax
    factor = max(1, factor)
    des_Nkx = int(Nkx * factor)
    if verbose:
        print('Xmax', Xmax)
        print('dx', dx)
        print('factor', factor)
        print('des Nkx', des_Nkx)
    zeros_to_add = des_Nkx - Nkx
    #add_arr = np.zeros_like(h_lzt)
    #add_arr = add_arr[:zeros_to_add,...]
    zero_shape = list(h_lzt.shape[1:] )
    zero_shape = [zeros_to_add] + zero_shape
    zero_shape = tuple(zero_shape)
    #h_lzt = np.concatenate((h_lzt, add_arr), axis=0)
    h_lzt = np.concatenate((h_lzt, np.zeros((zero_shape), dtype=np.complex_)), axis=0)
    Nx = 2*des_Nkx + 1 
    x = np.linspace(0, Xmax, Nx)
    # take the IRFFT
    zeta_xzt = dk*Nx*np.fft.irfft(h_lzt, n=Nx,axis=0) # get rid of scaling
    trunc_inds = x <= xmax
    x = x[trunc_inds]
    zeta_xzt = zeta_xzt[trunc_inds,...]
    return x, zeta_xzt

class IWField:
    def __init__(self, iw_disp):
        self.iw_disp = iw_disp
        self.dk_cpkm = None
        self.Nkx = None
        self.Nky = None
        self.dt = None
        self.Nt = None
        self.xmax = None
        self.dx_des = None
        self.J = None
        self.jstar = None
        self.E0 = None

    def add_GM_params(self, jstar, E0):
        self.jstar = jstar
        self.E0 = E0
        return

    def add_field_params(self, dk_cpkm,
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

    def gen_zeta_field(self):
        """
        Draw a random displacement field
        using Garrett-Munk spectrum
        """
        iw_disp = self.iw_disp
        disp_func = iw_disp.make_disp_func()
        dk_radpm = self.dk_cpkm * 2*np.pi / 1e3
        zgrid = iw_disp.zgrid
        bv_cph = np.interp(zgrid, iw_disp.bv_zgrid, iw_disp.bv_cph)
        BN0 = np.trapz(bv_cph, zgrid)
        lat = iw_disp.latitude
        h_lzt = get_h_lzt(disp_func, dk_radpm, self.dt, 
                                    self.Nkx, self.Nky, self.iw_disp.Nz_sav, self.Nt, 
                                    self.J, lat, BN0, self.jstar, self.E0)
        x, zeta_xzt = invert_lzt_to_xzt(dk_radpm, h_lzt, self.xmax, self.dx_des)
        t = np.linspace(0, (self.Nt - 1)*self.dt, self.Nt)
        z = self.iw_disp.zgrid_sav
        return x, z, t, zeta_xzt

    def get_zeta_corr_field(self):
        """
        Get correlation function 
        rho(x-x', y-y', z, z', t-t')
        Fix x=0, y=0, t=0
        return as 
        rho(Delta x, Delta y, z, z', Delta t)
        """
        iw_disp = self.iw_disp
        disp_func = iw_disp.make_disp_func()
        dk_radpm = self.dk_cpkm * 2*np.pi / 1e3
        zgrid = iw_disp.zgrid
        print(iw_disp.bv_zgrid.shape, iw_disp.bv_cph.shape)
        bv_cph = np.interp(zgrid, iw_disp.bv_zgrid, iw_disp.bv_cph)
        BN0 = np.trapz(bv_cph, zgrid)
        lat = iw_disp.latitude
        h_lzt = get_incoh_h_lzt(disp_func, dk_radpm, self.dt, 
                                    self.Nkx, self.Nky, self.iw_disp.Nz_sav, self.Nt, 
                                    self.J, lat, BN0, self.jstar, self.E0)
        x, incoh_zeta_xzt = invert_lzt_to_xzt(dk_radpm, h_lzt, self.xmax, self.dx_des)
        incoh_zeta_xzt *= dk_radpm
        z = iw_disp.zgrid_sav
        t = np.linspace(0, (self.Nt - 1)*self.dt, self.Nt)
        return x, z, t, incoh_zeta_xzt

    def get_zeta_depth_variance(self):
        """ 
        Get variance of displacement as function of depth
        """
        iw_disp = self.iw_disp
        disp_func = iw_disp.make_disp_func()
        kgrid_cpkm = np.linspace(iw_disp.kgrid_cpkm.min(), iw_disp.kgrid_cpkm.max(), 1000) # use fine grid since it's just for integrating pkj
        kradpm_grid = kgrid_cpkm * 2*np.pi / 1e3
        dk = kradpm_grid[1] - kradpm_grid[0]
        J = self.J
        jstar = self.jstar

        zgrid = iw_disp.zgrid_sav
        bv_cph = np.interp(zgrid, iw_disp.bv_zgrid, iw_disp.bv_cph)
        BN0 = np.trapz(bv_cph, zgrid)

        depth_var=  np.zeros_like(zgrid)
        jgrid = np.linspace(1, J, J)
        Hj_norm = np.sum(1 / (jgrid**2 + jstar**2))

        for k in range(kradpm_grid.size):
            kradpm = kradpm_grid[k]
            kcpkm = kradpm * 1e3 / 2 / np.pi
            pkj = get_pkj(kradpm, jgrid, iw_disp.latitude, J, jstar, Hj_norm, BN0, self.E0)
            phi = iw_disp.get_phi(kcpkm)
            depth_var += np.sum(pkj * phi**2, axis=1) #sum over modes
        depth_var *= dk
        return zgrid, depth_var

    def get_zeta_depth_cov(self, tlag):
        """ 
        Get covariance of displacement as function of depth and lag time
        E((zeta(z, t) - zeta(z, t'))^2)
        """
        iw_disp = self.iw_disp
        disp_func = iw_disp.make_disp_func()
        kgrid_cpkm = np.linspace(iw_disp.kgrid_cpkm.min(), iw_disp.kgrid_cpkm.max(), 1000) # use fine grid since it's just for integrating pkj
        kradpm_grid = kgrid_cpkm * 2*np.pi / 1e3
        dk = kradpm_grid[1] - kradpm_grid[0]
        J = self.J
        jstar = self.jstar

        zgrid = iw_disp.zgrid_sav
        bv_cph = np.interp(zgrid, iw_disp.bv_zgrid, iw_disp.bv_cph)
        BN0 = np.trapz(bv_cph, zgrid)

        depth_var=  np.zeros((zgrid.size, tlag.size))
        jgrid = np.linspace(1, J, J)
        Hj_norm = np.sum(1 / (jgrid**2 + jstar**2))

        for k in range(kradpm_grid.size):
            kradpm = kradpm_grid[k]
            kcpkm = kradpm * 1e3 / 2 / np.pi
            pkj = get_pkj(kradpm, jgrid, iw_disp.latitude, J, jstar, Hj_norm, BN0, self.E0)
            phi = iw_disp.get_phi(kcpkm)
            omega = iw_disp.get_omega(kcpkm)
            lag_term = np.cos(np.outer(omega , tlag))
            depth_var += np.sum((pkj * phi**2)[:,:,None] * lag_term[None, ...], axis=1) #sum over modes
        depth_var *= dk
        return zgrid, depth_var

    def get_w_depth_variance(self):
        """ 
        Get variance of vertical speed as function of depth
        """
        iw_disp = self.iw_disp
        disp_func = iw_disp.make_disp_func()
        kgrid_cpkm = np.linspace(iw_disp.kgrid_cpkm.min(), iw_disp.kgrid_cpkm.max(), 1000) # use fine grid since it's just for integrating pkj
        kradpm_grid = kgrid_cpkm * 2*np.pi / 1e3
        dk = kradpm_grid[1] - kradpm_grid[0]
        J = self.J
        jstar = self.jstar

        zgrid = iw_disp.zgrid_sav
        bv_cph = np.interp(zgrid, iw_disp.bv_zgrid, iw_disp.bv_cph)
        BN0 = np.trapz(bv_cph, zgrid)

        depth_var=  np.zeros_like(zgrid)
        jgrid = np.linspace(1, J, J)
        Hj_norm = np.sum(1 / (jgrid**2 + jstar**2))

        for k in range(kradpm_grid.size):
            kradpm = kradpm_grid[k]
            kcpkm = kradpm * 1e3 / 2 / np.pi
            pkj = get_pkj(kradpm, jgrid, iw_disp.latitude, J, jstar, Hj_norm, BN0, self.E0)
            phi = iw_disp.get_phi(kcpkm)
            omega = iw_disp.get_omega(kcpkm)
            phi *= omega # I square it below
            depth_var += np.sum(pkj * phi**2, axis=1) #sum over modes
        depth_var *= dk #need extra factor of dk because it's a variance calculation not a field calculation
        return zgrid, depth_var

    def get_w_depth_cov(self, tlag):
        """ 
        Get covariance of vertical speed as function of depth and lag time
        E((zeta(z, t) - zeta(z, t'))^2)
        """
        iw_disp = self.iw_disp
        disp_func = iw_disp.make_disp_func()
        kgrid_cpkm = np.linspace(iw_disp.kgrid_cpkm.min(), iw_disp.kgrid_cpkm.max(), 1000) # use fine grid since it's just for integrating pkj
        kradpm_grid = kgrid_cpkm * 2*np.pi / 1e3
        dk = kradpm_grid[1] - kradpm_grid[0]
        J = self.J
        jstar = self.jstar

        zgrid = iw_disp.zgrid_sav
        bv_cph = np.interp(zgrid, iw_disp.bv_zgrid, iw_disp.bv_cph)
        BN0 = np.trapz(bv_cph, zgrid)

        depth_var=  np.zeros((zgrid.size, tlag.size))
        jgrid = np.linspace(1, J, J)
        Hj_norm = np.sum(1 / (jgrid**2 + jstar**2))

        for k in range(kradpm_grid.size):
            kradpm = kradpm_grid[k]
            kcpkm = kradpm * 1e3 / 2 / np.pi
            pkj = get_pkj(kradpm, jgrid, iw_disp.latitude, J, jstar, Hj_norm, BN0, self.E0)
            phi = iw_disp.get_phi(kcpkm)
            omega = iw_disp.get_omega(kcpkm)
            phi *= omega # I square it below
            lag_term = np.cos(np.outer(omega , tlag))
            term1 = (pkj * phi**2)[:,:,None] # add dummy axis for lag
            term2 = lag_term[None,...] # add dummy axis for depth
            depth_var += np.sum(term1*term2, axis=1) #sum over modes
        depth_var *= dk # extra dk since it's a squarex quantity
        return zgrid, depth_var

    def get_w_corr_field(self):
        """
        Get correlation function 
        sigma(x-x', y-y', z, z', t-t')
        Fix x=0, y=0, t=0
        return as 
        sigma(Delta x, Delta y, z, z', Delta t)
        """
        iw_disp = self.iw_disp
        disp_func = iw_disp.make_disp_func()
        dk_radpm = self.dk_cpkm * 2*np.pi / 1e3
        zgrid = iw_disp.zgrid
        bv_cph = np.interp(zgrid, iw_disp.bv_zgrid, iw_disp.bv_cph)
        BN0 = np.trapz(bv_cph, zgrid)
        lat = iw_disp.latitude
        h_lzt = get_incoh_h_lzt(disp_func, dk_radpm, self.dt, 
                                    self.Nkx, self.Nky, self.iw_disp.Nz_sav, self.Nt, 
                                    self.J, lat, BN0, self.jstar, self.E0, w=True)
        z = iw_disp.zgrid_sav
        t = np.linspace(0, (self.Nt - 1)*self.dt, self.Nt)
        x, incoh_w_xzt = invert_lzt_to_xzt(dk_radpm, h_lzt, self.xmax, self.dx_des)
        incoh_w_xzt *= dk_radpm
        return x, z, t, incoh_w_xzt

    def get_ww_corr_field(self):
        """
        """
        iw_disp = self.iw_disp
        disp_func = iw_disp.make_disp_func()
        dk_radpm = self.dk_cpkm * 2*np.pi / 1e3
        zgrid = iw_disp.zgrid
        bv_cph = np.interp(zgrid, iw_disp.bv_zgrid, iw_disp.bv_cph)
        BN0 = np.trapz(bv_cph, zgrid)
        lat = iw_disp.latitude
        h_lzt = get_incoh_h_lzt(disp_func, dk_radpm, self.dt, 
                                    self.Nkx, self.Nky, self.iw_disp.Nz_sav, self.Nt, 
                                    self.J, lat, BN0, self.jstar, self.E0, ww=True)
        z = iw_disp.zgrid_sav
        t = np.linspace(0, (self.Nt - 1)*self.dt, self.Nt)
        x, incoh_w_xzt = invert_lzt_to_xzt(dk_radpm, h_lzt, self.xmax, self.dx_des)
        incoh_w_xzt *= dk_radpm # the inversion only includes on of the dk factors since its not for the product of kernels
        return x, z, t, incoh_w_xzt

    def gen_zuvw_field(self):
        """
        Generate a random realization
        of displacement, horizontal velocity, 
        and vertical velocity
        """
        iw_disp = self.iw_disp
        disp_func = iw_disp.make_disp_func()
        dk_radpm = self.dk_cpkm * 2*np.pi / 1e3
        zgrid = iw_disp.zgrid
        bv_cph = np.interp(zgrid, iw_disp.bv_zgrid, iw_disp.bv_cph)
        BN0 = np.trapz(bv_cph, zgrid)
        lat = iw_disp.latitude
        h_lzt, u_lzt, v_lzt, w_lzt = get_huvw_lzt(disp_func, self.iw_disp.zgrid_sav, dk_radpm, self.dt, 
                                    self.Nkx, self.Nky, self.iw_disp.Nz_sav, self.Nt, 
                                    self.J, lat, BN0, self.jstar, self.E0)
        x, zeta_xzt = invert_lzt_to_xzt(dk_radpm, h_lzt, self.xmax, self.dx_des)
        x, u_xzt = invert_lzt_to_xzt(dk_radpm, u_lzt, self.xmax, self.dx_des)
        x, v_xzt = invert_lzt_to_xzt(dk_radpm, v_lzt, self.xmax, self.dx_des)
        x, w_xzt = invert_lzt_to_xzt(dk_radpm, w_lzt, self.xmax, self.dx_des)
        z = iw_disp.zgrid_sav
        t = np.linspace(0, (self.Nt - 1)*self.dt, self.Nt)
        return x, z, t, zeta_xzt, u_xzt, v_xzt, w_xzt


