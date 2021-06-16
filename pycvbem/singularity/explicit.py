# -*- coding: utf-8 -*-
"""
Explicit Singularity handling module

Created on Thu May 20 14:50:09 2021

@author: Owner
"""

from abc import ABC, abstractmethod # abstract base class
# from typing import final # version 3.8, used in ExplicitSingularity

from warnings import warn
import numpy as np
from cmath import exp as cexp


class ExplicitSingularity(ABC):
    
    # @final
    def __call__(self, z): # make these typing.final methods, since there is not reason they should be overridden
        return self.Phi(z) + self.Psi(z)*1j
    
    # @final
    def Omega(self, z):
        return self.Phi(z) + self.Psi(z)*1j
    
    @abstractmethod
    def Phi(self, z):
        pass
    
    @abstractmethod
    def Psi(self, z):
        pass
    
    @abstractmethod
    def dOmegaDz(self, z):
        pass
    

class Well(ExplicitSingularity):
    def __init__(self, zw, qw_h, thetaBC=0, lowerBC=0, upExcl=True, ccw=True, rw=0.001):
        """
        
        """
        
        self.zw = complex(zw)
        self.qw_h = qw_h
        if thetaBC < 0: # angle principal (-pi, pi] branch is rotated to get the new branch
            warn('thetaBC must not be lower than 0, replacing with 0')
            self.thetaBC = 0
        elif thetaBC >= 2*np.pi:
            warn('thetaBC must not be greater than or equal to 2pi, replacing with 0')
            self.thetaBC = 0
        else:
            self.thetaBC = thetaBC
        self.lowerBC = lowerBC
        self.upperBC = lowerBC + 2*np.pi
        self.upExcl = upExcl
        self.ccw = ccw
        self.rw = rw # figure out picking of different points.
   
    # inherits __call__ and Omega from parent
    
    def dOmegaDz(self, z):
        return -self.qw_h/(2*np.pi)/(z-self.zw)
    
    def Phi(self, z):
        return -self.qw_h/(2*np.pi)*np.log(np.abs(z - self.zw))
    
    def Psi(self, z):
        return -self.qw_h/(2*np.pi)*self._theta(z)
    
    def _theta(self, z): # not properly vectorized
        
        alf = np.angle((z-self.zw)*cexp(-self.thetaBC*1j)) # rotated back to principal branch cut
        
        # p.b.c is (-pi, pi], so pi indicates on the new branch cut
        if alf == np.pi: # vectorization issue here. use where or mask for entire operation
            t = self.lowerBC if self.upExcl else self.upperBC
        else:
            t = alf + self.lowerBC + np.pi
            
        if self.ccw:
            return t # ccw is default
        else:
            return self.upperBC - t # compute the upperBC complement if measuring cw
        
class ExpSingSuperposition(ExplicitSingularity):
    def __init__(self, singList=[]):
        self._singList = []
        for s in singList:
            if isinstance(s, ExplicitSingularity):
                self._singList.append(s)
            else:
                # will this print multiple times? is there a way to tell how many or which indices were ignored?
                warn("Each singularity provided must be an instance of ExplicitSingularity, some entries excluded")
    
    @property
    def singList(self):
        return self._singList # danger of modifications to this affecting the original
    
    def add(self, s, index=None): # order does not really matter, I don't think. Unless needed for resolving BC issues
        if isinstance(s, ExplicitSingularity):
            if index:
                self._singList.insert(index, s)
            else:
                self._singList.append(s)
        else:
            raise TypeError("Singularity to add must be an instance of a child class of ExplicitSingularity")
            
    def remove(self, index=None):
        if index:
            return self._singList.pop(index)
        else:
            return self._singList.pop() # this behavior might be default if index is none
        
    # inherits Omega and __call__
    
    def Psi(self, z): # need a general rule for how psi combines around wells, though I think general addition is correct
        psi = 0
        for s in self._singList:
            psi += s.Psi(z)
        return psi
    
    def Phi(self, z):
        phi = 0
        for s in self._singList:
            phi += s.Phi(z)
        return phi
    
    def dOmegaDz(self, z):
        dOmdz = 0
        for s in self._singList:
            dOmdz += s.dOmegaDz(z)
        return dOmdz