#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:38:27 2021

@author: Denise Lanzieri
"""
import numpy as np
import tensorflow as tf
import flowpm
from flowpm.tfbackground import cosmo,afactor,z2a,chifactor
from flowpm.raytracing import  lightcone, Born
from flowpm.angular_power_tf import measure_power_spectrum_tf
from flowpm.tfpower import linear_matter_power



#%%
nc=[64,64,640]   # size of the cube, number of cells
plane_size=64                    # number of pixel for x and  y 
Boxsize=[200,200,2000]          # Physical size of the cube
r = np.linspace(0,2000,10, endpoint=True)
a = afactor(r)  
a_s=z2a(1.00)
ds=chifactor(a_s)
field=5.
a0=0.1
af=1.0
n_steps=10

#%%


@tf.function
def power_spectrum_for_cosmology(
              Omega0_m,
              sigma8):
    cosmology=cosmo
    cosmology['Omega0_m']=tf.convert_to_tensor(Omega0_m,dtype=tf.float32)
    cosmology['sigma8']=tf.convert_to_tensor(sigma8,dtype=tf.float32)
    init_stages = np.linspace(a0, af, n_steps, endpoint=True)
    initial_conditions = flowpm.linear_field(nc,    
                                            Boxsize, 
                                             lambda k: tf.cast(linear_matter_power(cosmo, k), tf.complex64),         
                                             batch_size=1)
    # Sample particles
    state = flowpm.lpt_init(initial_conditions, 0.1)   
    # Evolve particles down to z=0
    final_state = flowpm.nbody(state, init_stages, nc)         
    # Retrieve final density field
    state, lps_a, lps=lightcone(final_state, a[::-1], 
                                  nc, 
                                    field*60/plane_size, plane_size,
              cosmology)
    k_map=Born(lps_a,lps,ds,nc,Boxsize,plane_size,field,cosmology)
    k_map=tf.cast(k_map,dtype=tf.complex64)
    ell, power_spectrum=measure_power_spectrum_tf(k_map,field,plane_size)
    return ell, power_spectrum, k_map

