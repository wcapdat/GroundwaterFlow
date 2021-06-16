# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:13:18 2021

@author: Nathaniel Bowden
"""


import numpy as np
import cmath as cm    #use for complex arithmetic

from pycvbem.singularity.explicit import Well, ExpSingSuperposition
from pycvbem.singularity.implicit import Fracture


def uniformTiltedFlow():
    # see Sato Appendix E
    # amenable to tests of basic CVBEM
    
    z=[0, 1, 1+1j, 1j]  # Boundary points
    # theta=[np.pi/2]*4  # interior angles of polygon boundary
    P=[0, -1/2, -(1+np.sqrt(3))/2, np.nan]  # Dirichlet specified boundary values; use 0 for place holder=[0,0,0,-1/2]  
    S=[np.nan, np.nan, np.nan, -1/2]    # Neumann specified boundary values
    trueNodals = [0+0j, -0.5+0.866j, -1.366+0.366j, -0.866-0.5j]
    test_z = 0.5+0.5j
    Omega_tz = -cm.exp(-np.pi/3*1j)*(0.5+0.5j)
    
    return z, P, S, trueNodals, test_z, Omega_tz

def quart5ptPattern(nb):
    # see Sato p. 255
    # for tests of CVBEM + wells, no fractures
    
    w1 = Well(0, 1, thetaBC=0)
    w2 = Well(1+1j, -1, thetaBC=np.pi)
    
    zb = []
    P = []
    S = []
    for k in range(nb//4): # probably could do this with numpy
        zb.append(w1.rw + k*(1-w1.rw)/(nb//4))
        S.append(1/4)
        P.append(np.nan)
    for k in range(nb//4):
        zb.append(1 + k*1j/(nb//4))
        if k == 0:
            S.append(np.nan) # Dirichlet anchor
            P.append(0)
        else:
            S.append(1/4)
            P.append(np.nan)
    for k in range(nb//4):
        zb.append(w2.zw-w2.rw - k*(1-w2.rw)/(nb//4))
        S.append(0)
        P.append(np.nan)
    for k in range(nb//4):
        zb.append(1j - k*1j/(nb//4))
        if k == 0:
            S.append(np.nan) # Dirichlet anchor
            P.append(0)
        else:
            S.append(0)
            P.append(np.nan)
    
    
    # way to compute theta from zb:
    nb = len(zb)
    theta = []
    for k in range(nb): # which way does angle go?
        a = zb[k] - zb[(k+1)%nb]
        b = zb[k] - zb[k-1]
        
        # rotate frame and measure angle w.r.t a vector?
        theta.append(abs(cm.phase(b) - cm.phase(a)))
    
    return np.array(zb), P, S, ExpSingSuperposition([w1, w2])
    
    
def quart5ptPatternFracture(nb, Lf=0.5):
    # see Sato p. 255
    # for tests of CVBEM + wells, no fractures
    
    w1 = Well(0, 1, thetaBC=0)
    w2 = Well(1+1j, -1, thetaBC=np.pi)
    nL = round((nb//4)*Lf)
    f = Fracture.from_center(0.5+0.5j, Lf, 0, nL, nL, 0.0001, kf_rel=100)
    
    zb = []
    P = []
    S = []
    for k in range(nb//4): # probably could do this with numpy
        zb.append(w1.rw + k*(1-w1.rw)/(nb//4))
        S.append(1/4)
        P.append(np.nan)
    for k in range(nb//4):
        zb.append(1 + k*1j/(nb//4))
        if k == 0:
            S.append(np.nan) # Dirichlet anchor
            P.append(0)
        else:
            S.append(1/4)
            P.append(np.nan)
    for k in range(nb//4):
        zb.append(w2.zw-w2.rw - k*(1-w2.rw)/(nb//4))
        S.append(0)
        P.append(np.nan)
    for k in range(nb//4):
        zb.append(1j - k*1j/(nb//4))
        if k == 0:
            S.append(np.nan) # Dirichlet anchor
            P.append(0)
        else:
            S.append(0)
            P.append(np.nan)
    
    
    # way to compute theta from zb:
    nb = len(zb)
    theta = []
    for k in range(nb): # which way does angle go?
        a = zb[k] - zb[(k+1)%nb]
        b = zb[k] - zb[k-1]
        
        # rotate frame and measure angle w.r.t a vector?
        theta.append(abs(cm.phase(b) - cm.phase(a)))
    
    return np.array(zb), P, S, ExpSingSuperposition([w1, w2]), f

if __name__ == '__main__':
    pass
    