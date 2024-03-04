#!/usr/bin/env python3

"""
Created on Mon Apr  3 11:43:48 2023

@author: lunarampinelli
"""

import matplotlib.pyplot as plt
from astropy.io import fits
import scipy
import numpy as np
from gofish import imagecube

def N_radial_profile(path, Q, T, Tex, gu, Aul, Eu, orto_para=False, r_min=0, r_max=3, dr=0.1, inc=0.0, PA=0.0, z0=0.0, psi=1.0, r_taper=np.inf, q_taper=1.0, r_cavity=0.0):
    
    dim = int((r_max-r_min)/dr)
    rvals = np.arange(start=r_min+dr/2., stop=r_max+dr/2., step=dr)
    
    prefix, suffix = path.split('.image')
    new_string = prefix + '.JvMcorr' + '.image' + suffix
    M0 = imagecube(path)
    M0_jvm = imagecube(new_string)
    r_jvm, I_jvm, dI_jvm = M0_jvm._radial_profile_2D(rvals=rvals, assume_correlated=True, x0=0.0, y0=0.0, inc=inc, PA=PA, z0=z0, psi=psi, r_taper=r_taper, q_taper=q_taper, r_cavity=r_cavity, mask_frame='disk')
    r, I, dI = M0._radial_profile_2D(rvals=rvals, assume_correlated=True, x0=0.0, y0=0.0, inc=inc, PA=PA, z0=z0, psi=psi, r_taper=r_taper, q_taper=q_taper, r_cavity=r_cavity, mask_frame='disk')

    
    N = np.zeros(dim)
    dN = np.zeros(dim)
    if orto_para is True:
        assert gu.size==2, 'provide an array of gu for orto and para'
        assert Aul.size==2, 'provide an array of Aul for orto and para'
        assert Eu.size==2, 'provide an array of Eu for orto and para'
        N_o, N_p, dN_o, dN_p = np.zeros((4,dim))
        for i in range(dim):
            N_o[i], dN_o[i] = average_column_density(dF=0.75*dI[i], Q=Q, T=T, Tex=Tex, gu=gu[0], Aul=Aul[0], F_mean=0.75*I_jvm[i], Eu=Eu[0], Area=M0_jvm._calculate_beam_area_str()/(4.84813681e-6)**2)
            N_p[i], dN_p[i] = average_column_density(dF=0.25*dI[i], Q=Q, T=T, Tex=Tex, gu=gu[1], Aul=Aul[1], F_mean=0.25*I_jvm[i], Eu=Eu[1], Area=M0_jvm._calculate_beam_area_str()/(4.84813681e-6)**2)
            N[i] = N_o[i] + N_p[i]
            dN[i] = np.sqrt(dN_o[i]**2 + dN_p[i]**2)
    else:
        for i in range(dim):
            N[i], dN[i] =  average_column_density(dF=dI[i], Q=Q, T=T, Tex=Tex, gu=gu, Aul=Aul, F_mean=I_jvm[i], Eu=Eu, Area=M0_jvm._calculate_beam_area_str()/(4.84813681e-6)**2)

    return r, N, dN

def average_column_density(Q, T, Tex, gu, Aul, F_mean, Eu, Area, dF):
    # Q -> array
    # T [K] -> array
    # Tex [K] -> float
    # gu -> upper level degeneracy
    # Aul (log) [1/s] -> Einstein coefficient
    # F_mean [Jy m/s] -> average flux density over a region
    # Eu [K] -> upper state energy level
    # Area [as^2] -> integration area
    
    # interpolate partition function with a cubic spline
    assert Q.size==T.size, 'Q and T have different sizes: provide Q corresponding to each T'
    interpolate = scipy.interpolate.CubicSpline(T, Q)
    Q_Tex = interpolate(Tex)
    
    # convertions
    h = 6.62607015e-34                                  # J s
    c = 3*10**8                                         # m/s
    
    Aul = 10**(Aul)                  
    F_mean = F_mean*10**(-26)                           # J/(m s)
    dF = dF*10**(-26)
    Area = Area*(4.84813681e-6)**2
    # average column density
    N0 = Q_Tex*4*np.pi / gu
    N = N0 * F_mean * np.exp(Eu/Tex) / (Area*h*c*Aul*10**4) # cm^(-2)
    dN = abs(N * np.sqrt((dF/F_mean)**2))
    #print((dF/F_mean)**2)
    #print((Eu*dTex/Tex**2)**2)
    
    return N, dN

def radial_column_density(dim, Q, T, Tex, gu, Aul, Eu, flux, radius, dr, plot=False, xlim=None, ylim=None, title=None, compare=None, legend=None):
    # T = np.array([18.75, 37.50, 75.00])
    # area = array of annulus area (dim)
    # flux = array of flux for each annulus (dim+1)
    # Q = np.array([Q1, Q2, Q3])
    # xlim, ylim = 2D array with limits for axes
    # radius [arcsec] = array from annular flux density
    # dr [arcsec] = annulus width
    
    assert type(dim) == int, 'dim is the number of radial points, must be integer: check with annular flux density sampling'
    N = np.zeros(dim)

    
    for i in range(dim):
        N[i] =  average_column_density(Q=Q, T=T, Tex=Tex, gu=gu, Aul=Aul, F_mean=flux[i], Eu=Eu, Area=2*np.pi*(radius[i])*dr)
        
    if plot==True:
        fig = plt.figure(figsize=(10,5), dpi=100)
        if compare is not None:
            assert compare.size == N.size, 'provide another array of N of the same size'
            assert legend is not None, 'provide labels for the legend: array 2D of str'
            plt.plot(radius, compare, label=legend[1])
            plt.plot(radius, N, label=legend[0])
            plt.legend()
        else:
            plt.plot(radius,N)
            
        if xlim is not None:
            assert xlim.size == 2, 'provide a 2D numpy array'
            plt.xlim(xlim[0],xlim[1])
        if ylim is not None:
            assert ylim.size == 2, 'provide a 2D numpy array'
            plt.ylim(ylim[0],ylim[1])
        if title is not None:
            plt.title(title)
        fig.show()

        plt.xlabel('Radius [arcsec]')
        plt.ylabel(r'Column density [cm$^{-2}$]')
        plt.yscale('log');
    
    return N

def RMS(path, r_c=5.5, r=2.5, dim=5):
    
    # r_c [arcsec] = distance of circle to integrate flux density
    # r [arcsec] = radius of the circle to calculate integrated flux density
    # dim (int) = number of iteration
    M0 = fits.open(path)
    pix_area_as = abs(M0[0].header['CDELT1'])*abs(M0[0].header['CDELT2'])*(3600**2)
    M0.close()
    
    int_flux = np.zeros(dim)
    M0 = imagecube(path)
    assert M0.data.ndim == 2, 'Provided map is not 2D: provide an integrated intensity map!'

    
    r = r_c  #arcsec
    for i in range(dim):
        theta = np.random.rand()*2*np.pi
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        mask = M0.get_mask(r_min=0, r_max=r, mask_frame='sky', x0=x, y0=y)
        data = M0.data[mask]

        int_flux[i] = np.sum(data)/(M0._calculate_beam_area_arcsec()/pix_area_as) 
    
    rms = np.nanstd(int_flux)
    return rms