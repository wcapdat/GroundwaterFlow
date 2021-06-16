# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:42:14 2021

@author: Owner
"""

import cmath as cm
import numpy as np


cmlog = np.vectorize(cm.log)


cmsqrt = np.vectorize(cm.sqrt)


def knRoot(k, n, z):
    [r, phi] = cm.polar(z)
    root = cm.rect(r**(1/n), phi/n) * cm.exp(2*k*cm.pi/n*1j)
    return root
