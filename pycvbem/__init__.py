# -*- coding: utf-8 -*-
"""
pycvbem:
    Package implementing the Complex Variable Boundary Element Method (CVBEM) 
    in Python with support for implicit and explicit singularity programming.
    

Created on Tue May 18 21:07:11 2021

@author: Nathaniel Bowden
"""

# intra-module imports
from . import examples, utils, singularity
from .singularity import explicit, implicit
from .utils import cmlog


# external imports
import numpy as np
import cmath as cm    #use for complex arithmetic



def _H(k,j,z):
    nb = len(z)
    return ((z[k]-z[j])/(z[(j+1)%nb]-z[j]))*cm.log((z[(j+1)%nb]-z[k])/(z[j]-z[k]))


def _I(k,j,z):
    nb = len(z)
    return ((z[k]-z[(j+1)%nb])/(z[(j+1)%nb]-z[j]))*cm.log((z[(j+1)%nb]-z[k])/(z[j]-z[k]))


def _G(k):
    # array imag, real, imag, real
    pass


def _theta(zb):
    # way to compute theta from zb:
    nb = len(zb)
    theta = np.zeros(nb)
    for k in range(nb):
        
        # vectors oriented positively along boundary
        a = zb[(k+1)%nb] - zb[k]
        b = zb[k] - zb[k-1]
        
        # direction of b defines zero via rotation, but the angle is pi
        # less if a deviates to the left of b, more if to the right
        theta[k] = np.pi - cm.phase(a/b)
        
    return theta


def _dualNodeMatrix(nb, z, theta, DBV): # returns A but with wrong column ordering, correct row ordering
    # algorithm for the RHS of the CVBEM nodal equations
    
    # store coefficients for both nodal equations as 1, 2, 1, 2 etc on 
    # successive rows
    Coef = np.zeros((2*nb, 2*nb))
    
    for k in np.arange(nb):
        x = np.zeros(2*nb) #array with the values Phi_0, Phi_1,...,Phi_nb-1, Psi_1,...,Psi_nb-1 for first nodal equation (location nb = location 0)
        y = np.zeros(2*nb) #array with the values Phi_0, Phi_1,...,Phi_nb-1, Psi_1,...,Psi_nb-1 for second nodal equation
        
        # error from phi and psi being adjacent?
        
        # this part is the same for both
        for j in np.arange(nb):  
            if j!=k and j+1!=k and j != k-1+nb: # where does the last condition come from?
                x[j] -= _I(k,j,z).imag
                x[(j+1)%nb] += _H(k,j,z).imag
                x[nb+j] -= _I(k,j,z).real
                x[nb+(j+1)%nb] += _H(k,j,z).real # HERE: mod is distributive, so this index is equivalent to (j+1)%nb, need to add nb after mod
                y[j] -= _I(k,j,z).real
                y[(j+1)%nb] += _H(k,j,z).real
                y[nb+j] += _I(k,j,z).imag
                y[nb+(j+1)%nb] -= _H(k,j,z).imag # error was also present here
        
        if (DBV==k).any(): # test if k in DBV
            # set Phi coefs
            x[k] += 2*np.pi-theta[k]
            # set Psi coefs
            x[nb+k] += np.log(abs((z[(k+1)%nb]-z[k])/(z[k-1]-z[k])))
            # do same for 2nd nodal equation
            y[k] += np.log(abs((z[(k+1)%nb]-z[k])/(z[k-1]-z[k])))
            y[nb+k] += theta[k]

        else:
            x[k] -= theta[k]
            x[nb+k] += np.log(abs((z[(k+1)%nb]-z[k])/(z[k-1]-z[k])))
            y[k] += np.log(abs((z[(k+1)%nb]-z[k])/(z[k-1]-z[k])))
            y[nb+k] -= (2*np.pi-theta[k])
        
        # assign to coefficient matrix
        Coef[2*k, 0:-1:2] = x[0:nb] # even rows, even columns
        Coef[2*k, 1:2*nb:2] = x[nb:2*nb] # even rows, odd columns
        Coef[2*k+1, 0:-1:2] = y[0:nb] # odd rows, even columns
        Coef[2*k+1, 1:2*nb:2] = y[nb:2*nb] # odd rows, odd columns
    # end loop over k
    
    return Coef


def _knownNodalSolns(nb, z, theta, P, S, DVB):
    raise NotImplementedError
    pass


def _unknownNodalSolns(nb, z, theta, P, S, DVB):
    raise NotImplementedError
    pass


def _dualTargetVector(zb, P, S, DBV, expl_sing=None):
    nb = len(zb)
    
    Y = np.zeros((2*nb,))
    for k in np.arange(nb):
        if (DBV==k).any():
            Y[2*k] = 2*np.pi*(P[k]-expl_sing.Phi(zb[k])) if expl_sing else 2*np.pi*P[k]
        else:
            Y[2*k+1] = -2*np.pi*(S[k]-expl_sing.Psi(zb[k])) if expl_sing else -2*np.pi*S[k]
    
    return Y


def _discreteCauchyIntegral(nb, zeta, nodals):
    # below will fail with nb == 2 (which should never happen)
    omegaJp1 = np.array(list(nodals[1:nb])+[nodals[0]]) # 1st to end concat. zeroth
    zetaJp1 = np.array(list(zeta[1:nb])+[zeta[0]]) # same. nb is a shorthand for len(nodals)
    omegaJ = np.array(nodals)
    zetaJ = np.array(zeta)
    # print(omegaJp1.shape, zetaJp1.shape, omegaJ.shape, zetaJ.shape)
    
    ## !! not properly vectorized, only works with one z point
    def omegaTilde(z):
        integrand = (((z-zetaJ)*omegaJp1-(z-zetaJp1)*omegaJ)/(zetaJp1-zetaJ) 
                     * cmlog((zetaJp1-z)/(zetaJ-z)))
        # print('discreteCauchyIntegral: len(integrand)==nb?', len(integrand)==nb) # true expected
        return integrand.sum()/(2*np.pi*1j)
    
    return omegaTilde


def CVBEM(zb, P, S, equiv='dual', fracture=None, expl_sing=None):
    
    # handle exp_sing = iterable of ExplicitSingularity instances
    
    nb = len(zb)
    theta = _theta(zb)
    
    # identify Dirichlet conditions by index
    DBV = np.where(~np.isnan(P)) # bitwise negation. keep in mind nan != nan always true
    
    # get nodal equation RHS matrix:
    # and vector of LHS
    if equiv == 'dual' or equiv == 3:
        A = _dualNodeMatrix(nb, zb, theta, DBV)
        Y = _dualTargetVector(zb, P, S, DBV, expl_sing=expl_sing)
        
        if fracture:
            B = implicit.fracture_Bmatrix(zb, fracture, DBV)
            C = implicit.fracture_Cmatrix(zb, fracture)
            D = implicit.fracture_Dmatrix(fracture)
            F = implicit.fracSingConstant(fracture, expl_sing)
            
            ABCD = np.block([[A, B], [C, D]])
            YF = np.concatenate((Y, F))
            OX = np.linalg.inv(ABCD).dot(YF)
            
            phiPsiVector = OX[0:2*nb]
            abc = OX[2*nb:2*nb+fracture.nf]
        else:
            phiPsiVector = np.linalg.inv(A).dot(Y)
            abc = None
        
    elif equiv == 'known' or equiv == 1:
        solns = _knownNodalSolns(nb, zb, theta, P, S, DBV)
        # these are more complicated, i need to keep 
        # track of which elements are Phi and Psi
        phiPsiVector = None
        
    elif equiv == 'unknown' or equiv == 2:
        solns = _unknownNodalSolns(nb, zb, theta, DBV)
        # these are more complicated
        phiPsiVector = None
        
    else:
        raise ValueError("Unsupported value for `equiv`.")
    
    # alternating Phi, Psi
    nodals = phiPsiVector[0:-1:2] + phiPsiVector[1:2*nb:2]*1j
    
    OmegaNS = _discreteCauchyIntegral(nb, zb, nodals) # change to express fractures and singularities
    # not sure how to do it for multiple fractures
    # OmegaF = np.array([fractures[m].buildOmegaF(abc[m]) for m in range(len(fractures))])
    
    if fracture and expl_sing:
        funcApprox = lambda x: OmegaNS(x) + fracture.buildOmegaF(abc)(x) + expl_sing.Omega(x)
    elif fracture:
        funcApprox = lambda x: OmegaNS(x) + fracture.buildOmegaF(abc)(x)
    elif expl_sing:
        funcApprox = lambda x: OmegaNS(x) + expl_sing.Omega(x)
    else:
        funcApprox = OmegaNS
    
    return funcApprox, nodals, abc # replace with a bunch object of variables



__all__ = ["CVBEM", "examples", "utils", "singularity"]
 
if __name__ == '__main__':
    pass