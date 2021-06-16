# -*- coding: utf-8 -*-
"""
Singularity handling subpackage

Modules:
--------

    :mod:`singularity.explicit`
        For handling explicit singularities, e.g. Wells
        
    :mod:`singularity.implicit`
        For handling implicit singularities, e.g. fractures

Created on Thu May 20 15:13:56 2021

@author: Nathaniel Bowden
"""

from . import implicit, explicit

__all__ = ['implicit', 'explicit']