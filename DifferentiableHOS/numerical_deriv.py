#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:59:00 2021

@author: Denise Lnazieri
"""

import numdifftools as nd
import tensorflow as tf
import numpy as np
import flowpm
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
from DifferentiableHOS.pk import power_spectrum 

nsteps = 2
nc = 16
box_size = 128
sigma8 = 0.8159


#%%
def compute_initial_cond(Omega_c):
    # Instantiates a cosmology with desired parameters
    cosmology = flowpm.cosmology.Planck15(Omega_c=Omega_c)
    # Compute linear matter power spectrum
    k = tf.constant(np.logspace(-4, 1, 256), dtype=tf.float32)
    pk = tfpower.linear_matter_power(cosmology, k)
    pk_fun = lambda x: tf.cast(
        tf.reshape(
            interpolate.interp_tf(tf.reshape(tf.cast(x, tf.float32), [-1]), k,
                                  pk), x.shape), tf.complex64)

    # And initial conditions
    initial_conditions = flowpm.linear_field([nc, nc, nc],
                                             [box_size, box_size, box_size],
                                             pk_fun,
                                             batch_size=1)
    return initial_conditions


#%%
initial_conditions = compute_initial_cond(0.2589)

#%%


@tf.function
def compute_powerspectrum(Omega_c):
    """ Computes a N-body simulation for a given
        set of cosmological parameters
        """
    # Instantiates a cosmology with desired parameters
    cosmology = flowpm.cosmology.Planck15(Omega_c=Omega_c)
    stages = np.linspace(0.1, 1., nsteps, endpoint=True)

    state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)

    # Evolve particles from initial state down to a=af
    final_state = flowpm.nbody(cosmology, state, stages, [nc, nc, nc])

    # Retrieve final density field i.e interpolate the particles to the mesh
    final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions),
                                   final_state[0])
    final_field = tf.reshape(final_field, [nc, nc, nc])
    k, power_spectrum = power_spectrum(final_field,
                            boxsize=np.array([box_size, box_size, box_size]),
                            kmin=0.1,
                            dk=2 * np.pi / box_size)
    return power_spectrum


#%%


#%%
@tf.function
def Flow_jac(Omega_c):
    """ Computes a N-body simulation for a given
    set of cosmological parameters
    """
    params = tf.stack([Omega_c])
    with tf.GradientTape() as tape:
        tape.watch(params)
        cosmology = flowpm.cosmology.Planck15(Omega_c=params[0])
        stages = np.linspace(0.1, 1., nsteps, endpoint=True)
        state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)

        # Evolve particles from initial state down to a=af
        final_state = flowpm.nbody(cosmology, state, stages, [nc, nc, nc])

        # Retrieve final density field i.e interpolate the particles to the mesh
        final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions),
                                       final_state[0])
        final_field = tf.reshape(final_field, [nc, nc, nc])
        k, power_spectrum = power_spectrum(final_field,
                                boxsize=np.array(
                                    [box_size, box_size, box_size]),
                                kmin=0.1,
                                dk=2 * np.pi / box_size)
    return tape.jacobian(power_spectrum, params, experimental_use_pfor=False)
