#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:19:45 2021

@author: Denise Lanzieri
"""

from DifferentiableHOS.numerical_deriv import Flow_jac,compute_powerspectrum
import tensorflow as tf
from numpy.testing import assert_allclose


#%%
z_source=1.
field=5.
box_size=200.
nc=16
Omega_c= 0.2589
sigma8= 0.8159
nsteps=2
#%%

def test_Nbody_jacobian():
    """ This function tests the lightcone implementation in TensorFlow 
    comparing it with Lenstools
      """
   # initial_conditions=compute_initial_cond(Omega_c)
    theoretical, numerical_jac=tf.test.compute_gradient( compute_powerspectrum, [Omega_c], delta=0.01)
    FlowPM_jac= Flow_jac(Omega_c)
    assert_allclose(numerical_jac[0],FlowPM_jac, rtol=1e-1)
    
