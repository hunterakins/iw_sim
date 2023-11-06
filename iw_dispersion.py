"""
Description:
    Object to manage the dispersion relation for internal waves.
    Idea is to run the exact solver at a grid of horizontal wavenumbers
    Then these results are interpolated to get the dispersion relation

    IWDispersion stores all saved results

Date:
    11/5/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from numba import njit
from iw_sim import interp
from pykrak import wave_sturm_seq as wss
from pykrak import wave_inverse_iteration as wii
from matplotlib import pyplot as plt


def iw_solve(z_arr, b_sq, h, J, kgrid_cpkm, omega_I, sav_z_arr, verbose=True):
    """
    Get phi and gammas
    z_arr - np 1d array 
        grid depths for b sq
    b_sq - np 1d array
        buoyance frequency squared minus intertial frequency squared
        in (RADIANS / S)**2
        NOTE: IT HAS THE TOP AND BOTTOM POINTS TRUNCATED, SO IT IS 
        SUITABLE FOR THE STURM-SEQ ALGORITHM
    h - float
        grid spacing of z_arr
    kgrid - np 1d array
        horizontal wavenumbers in cycles / km
    omega_I - float
        inertial frequency in rad /s 
    sav_z_arr - np 1d array
        depths to save phi at
    """

    if b_sq.size != z_arr.size - 2:
        raise ValueError('b_sq must have size z_arr.size - 2. z_arr should include surface and bottom depths for mode calculation')

    modes = []

    omega_arr = np.zeros((J, kgrid_cpkm.size))
    phi_arr = np.zeros((sav_z_arr.size, J, kgrid_cpkm.size))

    for i in range(kgrid_cpkm.size):
        k = kgrid_cpkm[i]
        k_radpm = k * 2 * np.pi /1000 # convert to radians / m
        if verbose:
            print('Running solver iter. ({0}/{1}) for k (rad/m)'.format(i+1, kgrid_cpkm.size), k_radpm)
            print('k (c/km)',k)
        gammas = wss.get_comp_gammas(k_radpm, h, b_sq, J) # truncate top and bottom points to get pressure release
        # convert gamma to to cph
        omegas = np.sqrt(k_radpm ** 2 / (gammas**2) + omega_I**2) #rad/s
        omega_arr[:,i] = omegas
        phi = wii.get_phi(gammas, k_radpm, h, b_sq)
        for j in range(J):
            phi_arr[:,j,i] = interp.vec_lin_int(sav_z_arr, z_arr, phi[:,j])
    return omega_arr, phi_arr

@njit
def omega_interp(k_cpkm, omega_arr, omega_spl_arr, J, kgrid_cpkm):
        omegas = np.zeros(J)
        for i in range(J):
            omegas[i] = interp.splint(k_cpkm, kgrid_cpkm, omega_arr[i,:], omega_spl_arr[i,:])[0]
        return omegas

@njit
def phi_interp(k_cpkm, phi_arr, phi_spl_arr, J, Nz, kgrid_cpkm):
        phi = np.zeros((Nz, J))
        for i in range(J):
            for j in range(Nz):
                phi[j,i] = interp.splint(k_cpkm, kgrid_cpkm, phi_arr[j,i,:], phi_spl_arr[j,i,:])[0]
        return phi

class IWDispersion:
    def __init__(self, bv_zgrid, bv_cph, latitude):
        """
        bv_zgrid : np 1d array
            depth grid for Brunt Vaisala frequency in meters
        bv_cph : np 1d array
            Brunt Vaisala frequency in cycles per hour
        latitude : float
            latitude in degrees

        Manage internal wave simulation. Store model run
        outputs. Interpolate to get dispersion at new
        horizontal wavenumbers. 

        Uninitialized attributes:
        kgrid_cpkm : np 1d array
            horizontal wavenumber grid in cycles per kilometer
        J : int
            number of modes to include
        omega_arr : np 2d array
            omega_arr[j,k] is the frequency of the jth mode at the kth
            horizontal wavenumber in radians per second
        phi_arr : np 3d array
            phi_arr[z,j,k] is the phase of the jth mode at the kth
            horizontal wavenumber at the zth depth 
        """
        self.latitude = latitude # degrees
        self.fI = 1/12 * np.sin(latitude*np.pi/180) # in cycles per hour
        self.omegaI = self.fI * 2 * np.pi / 3600 # rad / s
        if bv_zgrid[0] != 0:
            raise ValueError('bv_zgrid must start at 0')
        self.bv_zgrid = bv_zgrid
        bv_cph[bv_cph < self.fI] = self.fI + 1e-15
        self.bv_cph = bv_cph
        self.bv_radps = bv_cph * 2 * np.pi / 3600
        self.J = None
        self.kgrid_cpkm = None
        self.omega_arr = None
        self.phi_arr = None
        self.Nk = None
        self.Nz = None # mesh number of points
        self.zgrid = None # mesh grid
        """
        self.J = omega_arr.shape[0]
        self.Nk = omega_arr.shape[1]
        self.Nz = phi_arr.shape[0]
        self.kgrid_cpkm = kgrid_cpkm
        self.omega_arr = omega_arr # radians per second
        self.omega_arr_cph = omega_arr * 3600 / (2 * np.pi)
        self.phi_arr = phi_arr
        self.kgrid_radpm = kgrid_cpkm * 2 * np.pi / 1000
        self._fill_omega_spl_arr()
        self._fill_phi_spl_arr()
        """

    def _get_b_sq(self, mesh_dz):
        """
        Get b^2 (N^2 - f^2) in (rad/s)^2
        on a mesh
        First and last points of z_arr are the surface and bottom
        """
        z = self.bv_zgrid
        bv_radps = self.bv_radps
        bv_sq = bv_radps**2
        lat = self.latitude
        fI = 1/12 * np.sin(lat*np.pi/180) # in cycles per hour
        omegaI = fI * 2 * np.pi/3600 # rad / s
        b_sq = bv_sq - omegaI**2

        if np.any(b_sq < 0):
            raise ValueError('BV is less than inertial frequency')

        # interpolate onto grid
        Z = z[-1] - z[0]
        Nz = int(Z / mesh_dz) + 1
        zgrid = np.linspace(z[0], z[-1], Nz)
        b_sq = interp.vec_lin_int(zgrid, z, b_sq)
        self.zgrid = zgrid
        self.b_sq = b_sq
        self.Nz = Nz
        h = zgrid[1] - zgrid[0]
        return zgrid, b_sq, h

    def _set_J(self, J):
        """
        Set number of modes to include
        """
        self.J = J
        return

    def _set_kgrid(self, kgrid_cpkm):
        """
        Set horizontal wavenumber grid in cycles per kilometer
        """
        self.kgrid_cpkm = kgrid_cpkm
        self.Nk = kgrid_cpkm.shape[0]
        return

    def _set_mesh_dz(self, mesh_dz):
        """
        Set mesh spacing for internal wave simulation
        """
        self.mesh_dz = mesh_dz
        return

    def _set_sav_dz(self, sav_dz):
        """
        Set mesh spacing for saving internal wave simulation
        """
        self.sav_dz = sav_dz
        return

    def set_sim_params(self, kgrid_cpkm, J, mesh_dz, sav_dz):
        """
        Set simulation parameters
        """
        self._set_kgrid(kgrid_cpkm)
        self._set_J(J)
        self._set_mesh_dz(mesh_dz)
        self._set_sav_dz(sav_dz)
        return
    
    def compute_omega_phi_arr(self,verbose=True):
        """
        Get brunt vaissala frequency squared minus inertial frequency squared
        in radians per second on a mesh with spacing at least mesh_dz
        """
        if self.J is None:
            raise ValueError('J must be set. Use set_sim_params(kgrid, J, mesh_dz)')
        mesh_dz = self.mesh_dz
        J = self.J
        zgrid, b_sq, h = self._get_b_sq(mesh_dz)
        kgrid_cpkm = self.kgrid_cpkm
        omega_I = self.omegaI
        Z = zgrid[-1] - zgrid[0]
        N_sav = int(Z / self.sav_dz) + 1
        sav_zgrid = np.linspace(zgrid[0], zgrid[-1], N_sav)
        self.zgrid_sav = sav_zgrid
        self.Nz_sav = N_sav
        omega_arr, phi_arr = iw_solve(zgrid, b_sq[1:-1], h, J, kgrid_cpkm, omega_I, sav_zgrid, verbose=verbose   )
        self.omega_arr = omega_arr
        self.phi_arr = phi_arr
        self._fill_omega_spl_arr()
        self._fill_phi_spl_arr()
        return

    def _fill_omega_spl_arr(self):
        omega_arr = self.omega_arr
        kgrid_cpkm = self.kgrid_cpkm
        omega_spl_arr = np.zeros((self.J, self.Nk))
        for i in range(self.J):
            spl = interp.get_spline(kgrid_cpkm, omega_arr[i,:], 1e30, 1e30)
            omega_spl_arr[i,:] = spl
        self.omega_spl_arr = omega_spl_arr
        return

    def _fill_phi_spl_arr(self):
        phi_arr = self.phi_arr
        kgrid_cpkm = self.kgrid_cpkm
        phi_spl_arr = np.zeros((self.Nz_sav, self.J, self.Nk))
        for i in range(self.Nz_sav):
            for j in range(self.J):
                spl = interp.get_spline(kgrid_cpkm, phi_arr[i,j,:], 1e30, 1e30)
                phi_spl_arr[i,j,:] = spl
        self.phi_spl_arr = phi_spl_arr
        return

    def get_omega(self, k_cpkm):
        omega_arr, omega_spl_arr = self.omega_arr, self.omega_spl_arr
        J = self.J
        kgrid_cpkm = self.kgrid_cpkm
        omegas = omega_interp(k_cpkm, omega_arr, omega_spl_arr, J, kgrid_cpkm)
        return omegas

    def get_phi(self, k_cpkm):
        phi_arr, phi_spl_arr = self.phi_arr, self.phi_spl_arr
        J, Nz = self.J, self.Nz_sav
        phi = phi_interp(k_cpkm, phi_arr, phi_spl_arr, J, Nz, kgrid_cpkm)
        return phi

    def __call__(self, k_cpkm):
        omegas = self.get_omega(k_cpkm)
        phi = self.get_phi(k_cpkm)
        return omegas, phi

    def make_disp_func(self):
        omega_arr, omega_spl_arr = self.omega_arr, self.omega_spl_arr
        phi_arr, phi_spl_arr = self.phi_arr, self.phi_spl_arr
        J, Nz = self.J, self.Nz_sav
        kgrid_cpkm = self.kgrid_cpkm
        @njit
        def disp_func(k_cpkm):
            omegas = omega_interp(k_cpkm, omega_arr, omega_spl_arr, J, kgrid_cpkm)
            phi = phi_interp(k_cpkm, phi_arr, phi_spl_arr, J, Nz, kgrid_cpkm)
            return omegas, phi
        return disp_func
