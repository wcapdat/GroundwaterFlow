# -*- coding: utf-8 -*-
"""
Implicit singularity handling module

Created on Thu May 20 15:01:57 2021

@author: Nathaniel Bowden
"""

# external imports
import numpy as np

# intra-package
from .explicit import ExplicitSingularity
from ..utils import knRoot, cmlog, cmsqrt



def fracture_Bmatrix(zb, frac, DBV):
    # zb = list of boundary points
    if not isinstance(frac, Fracture):
        raise TypeError('`frac` must be an instance of the Fracture class or a subclass')
    
    nb = len(zb)
    
    # build and return the 2*nb x nf matrix
    B = np.zeros((2*nb, frac.nf))
    
    # iterate over boundary conditions
    for k in range(nb):
        if np.isin(k, DBV): # if a Dirichlet condition...
            B[2*k, 0:frac.nL] = [2*np.pi*frac.e_aNegN(n, zb[k]).real for n in range(frac.nL)]
            B[2*k, frac.nL:frac.nL+frac.nP] = [2*np.pi*frac.e_bj(j, zb[k]).real for j in range(frac.nP)]
            B[2*k, frac.nL+frac.nP:frac.nf] = [-2*np.pi*frac.e_cj(j, zb[k]).imag for j in range(1, frac.nP-1)]
            
            B[2*k+1, :] = 0
            
        else: # Neumann condition
            B[2*k, :] = 0
            
            B[2*k+1, 0:frac.nL] = [-2*np.pi*frac.e_aNegN(n, zb[k]).imag for n in range(frac.nL)]
            B[2*k+1, frac.nL:frac.nL+frac.nP] = [-2*np.pi*frac.e_bj(j, zb[k]).imag for j in range(frac.nP)]
            B[2*k+1, frac.nL+frac.nP:frac.nf] = [-2*np.pi*frac.e_cj(j, zb[k]).real for j in range(1, frac.nP-1)]
    
    return B

def fracture_Cmatrix(zb, frac):
    # zb = list of boundary points
    
    if not isinstance(frac, Fracture):
        raise TypeError('`frac` must be an instance of the Fracture class or a subclass')
    
    nb = len(zb)
    
    # construct the nf x 2nb matrix
    C = np.zeros((frac.nf, 2*nb))
    
    # iterate over boundary conditions (is this supposed to be zb?)
    for i in range(frac.nf):
        for k in range(nb):
            h_k = cmlog((zb[(k+1)%nb]-frac.zf[i])/(zb[k]-frac.zf[i]))
            h_k_1 = cmlog((zb[k]-frac.zf[i])/(zb[k-1]-frac.zf[i]))
            C[i, 2*k] = -((frac.z2-frac.z1)/(2*np.pi*1j*frac.Lf) 
                          * (h_k_1/(zb[k]-zb[k-1]) - h_k/(zb[(k+1)%nb]-zb[k]))).real \
                * frac.kf_rel * frac.wf  # C1j 
            C[i, 2*k+1] = ((frac.z2-frac.z1)/(2*np.pi*1j*frac.Lf) * 
                           (h_k_1/(zb[k]-zb[k-1])- h_k/(zb[(k+1)%nb]-zb[k]))).imag \
                * frac.kf_rel * frac.wf  # C2j 
                           
    return C


def fracture_Dmatrix(frac):
    # zb = list of boundary points
    if not isinstance(frac, Fracture):
        raise TypeError('`frac` must be an instance of the Fracture class or a subclass')
    
    # construct the nf x nf matrix
    D = np.zeros((frac.nf, frac.nf))
    
    # iterate over fracture pts
    # 0.999 used instead of 1 to avoid NaN/inf values
    for i in range(frac.nf):
        chi_i = frac.chi(frac.zf[i])
        D[i, 0:frac.nL] = [(frac.e_aNegN(n, frac.zf[i]) - frac.e_aNegN(n, frac.zf[i],conj=True)).imag 
                           + (4*(n+1)/(frac.Lf*chi_i**(n)*(chi_i**2-0.9999))).real 
                           * frac.kf_rel*frac.wf for n in range(frac.nL)]
        D[i, frac.nL:frac.nL+frac.nP] = [(frac.e_bj(j,frac.zf[i])-frac.e_bj(j,frac.zf[i],conj=True)).imag 
                                         + ((4*chi_i**2/(frac.Lf*(chi_i**2-0.9999))) 
                                            * (frac.e_bj_deriv(j,frac.zf[i]))).real 
                                         * frac.kf_rel*frac.wf for j in range(frac.nP)]

        D[i, frac.nL+frac.nP:frac.nf] = [(frac.e_cj(j, frac.zf[i]) - frac.e_cj(j, frac.zf[i], conj=True)).real 
                                         - ((4*chi_i**2/(frac.Lf*(chi_i**2-0.9999)))
                                            * (frac.e_bj_deriv(j,frac.zf[i]))).imag 
                                         * frac.kf_rel*frac.wf for j in range(1, frac.nP-1)]
        
    return D


def fracSingConstant(frac, exp_sing):
    """
    
    """
    
    if not isinstance(frac, Fracture):
        raise TypeError("`frac` must be an instance of the Fracture class")
        
    if not isinstance(exp_sing, ExplicitSingularity):
        raise TypeError('`exp_sing` must be an instance of a class derived from ExplicitSingularity')
        
    # expect frac.zf to be np.ndarray
    F = frac.kf_rel*frac.wf*((frac.z2-frac.z1)/frac.Lf*exp_sing.dOmegaDz(frac.zf)).real
    return F


# In[129]:


class Fracture():
    
    def __init__(self, z1, z2, nL, nP, wf, kf_rel=1e6, rPX=0.5):
        # number of parameters
        self.nL = nL
        self.nP = nP # this just counts the ones in the upper X plane
        self.nf = nL+2*nP-2
        
        # length, width, relative conductivity
        self.Lf = abs(z2-z1) # assuming they are complex, sort of
        self.kf_rel = kf_rel
        self.wf = wf
        self.z1 = z1
        self.z2 = z2
        
        # radius for poles inside fracture
        if rPX <= 0 or rPX >= 1:
            raise ValueError('Radius of poles in Chi plane, rPX, must be positive and less than 1')
        self.rPX = rPX
        
        # poles given by the roots of unity times rPX
        # only need the top half, since the ones at neg. y are included with complex conjugates
        self.Xp = [self.rPX*knRoot(k, 2*nP-2, 1) for k in range(self.nP)]
        
        # set array of fracture points: (look into numpy way of doing, maybe np.arange and transforms)
        _zf = []
        _zf.append(z1)
        for i in range(self.nf-2):
            _zf.append(z1 + (i+1)*(z2-z1)/(self.nf-1))                   
        _zf.append(z2)
        self.zf = np.array(_zf)
        
    @classmethod
    def from_center(cls, zc, Lf, theta, nL, nP, wf, kf_rel=1e6, rPX=0.5):
        z1 = zc - Lf/2*cm.exp(theta*1j)
        z2 = zc + Lf/2*cm.exp(theta*1j)
        return cls(z1, z2, nL, nP, wf, kf_rel=kf_rel, rPX=rPX)
        
    def Z(self, z):
        return (2*z - (self.z1 + self.z2))/(self.z2 - self.z1)
    
    def chi(self, z):
        return self.Z(z) + cmsqrt(self.Z(z)**2-1) # vectorized cm sqrt
    
    def e_aNegN(self, n, z, conj=False): # eigenfunction of a_(-n) eval'd. at z (just return the function?)
        if (conj == True):
            return self.chi(z).conjugate()**(-n-1)
        else:
            return self.chi(z)**(-n-1)
    
    def e_bj(self, j, z, conj=False): # eigenfunction of bj
        Xpj = self.Xp[j]
        if (conj == True):
            X = self.chi(z).conjugate()
        else:
            X = self.chi(z)
        return Xpj/(X-Xpj) + Xpj.conjugate()/(X-Xpj.conjugate())
    
    def e_cj(self, j, z, conj=False): # eigenfunction of cj
        if j < 1 or j > self.nP-2:
            warn('Calling the zero function, `j` must be between 1 and nP-2, inclusive, to yield an eigenfunction')
            return 0
        
        Xpj = self.Xp[j]
        if (conj == True):
            X = self.chi(z).conjugate()
        else:
            X = self.chi(z)
        
        return Xpj/(X-Xpj) - Xpj.conjugate()/(X-Xpj.conjugate())
                                              
    def e_bj_deriv(self, j, z): # eigenfunction of bj
        Xpj = self.Xp[j]
        X = self.chi(z)
        return Xpj/(X-Xpj)**2 + Xpj.conjugate()/(X-Xpj.conjugate())**2

    def e_cj_deriv(self, j, z): # eigenfunction of cj
        if j < 1 or j > self.nP-2:
            warn('Calling the zero function, `j` must be between 1 and nP-2, inclusive, to yield an eigenfunction')
            return 0
        else:
            Xpj = self.Xp[j]
            X = self.chi(z)
            return Xpj/(X-Xpj)**2 - Xpj.conjugate()/(X-Xpj.conjugate())**2
    
    def buildOmegaF(self, abc):
        # do I want it as fracture bound method?
        
        if len(abc) != self.nf:
            raise ValueError(f'`abc` must be an iterable of parameters of length nf = {self.nf}.')
        
        def OmegaF(z):
            z = np.array(z)
            eigen = np.zeros((self.nf, *z.shape), dtype=np.complex128)
            # move earlier and make it an array of lambdas
            # I have to run the list operations below every z...
            
            # makes a nf x z.dim1 x z.dim2... array
            eigen[0:self.nL] = [self.e_aNegN(n, z) for n in range(self.nL)]
            eigen[self.nL:self.nL+self.nP] = [self.e_bj(j, z) for j in range(self.nP)]
            eigen[self.nL+self.nP:self.nf] = [self.e_cj(j, z) for j in range(1, self.nP-1)]
            
            return np.tensordot(abc, eigen, axes=1) # sum products over last in abc, first in eigen
        
        return OmegaF