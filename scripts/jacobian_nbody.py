#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:17:09 2021

@author: Denise Lanzieri
"""

import tensorflow as tf
import numpy as np
#import DifferentiableHOS as DHOS
import flowpm
# TODO: This is very annoying, we need to fix the FlowPM imports
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
from absl import app
from absl import flags
from DifferentiableHOS.pk import pk as pkl
import pickle
#%%
flags.DEFINE_string("filename", "results.pkl", "Output filename")
flags.DEFINE_float("Omega_c", 0.2589, "Fiducial CDM fraction")
flags.DEFINE_float("sigma8", 0.8159, "Fiducial sigma_8 value")
flags.DEFINE_integer("nc", 32, "Transverse size of the cube")
flags.DEFINE_integer("plane_size", 32, "Number of pixels in x,y")
flags.DEFINE_float("box_size", 200, "Transverse box size [Mpc/h]")
flags.DEFINE_float("field", 5., "Transverse size of the lensing field [deg]")
flags.DEFINE_integer("nsteps", 30, "Number of time steps in the lightcone")

FLAGS = flags.FLAGS

#%%
@tf.function
def compute_Nbody(Omega_c, sigma8):
  """ Computes a N-body simulation for a given
  set of cosmological parameters
  """
  # Instantiates a cosmology with desired parameters
  cosmology = flowpm.cosmology.Planck15(Omega_c=Omega_c, sigma8=sigma8)

  # Compute linear matter power spectrum
  k = tf.constant(np.logspace(-4, 1, 256), dtype=tf.float32)
  pk = tfpower.linear_matter_power(cosmology, k)
  pk_fun = lambda x: tf.cast(tf.reshape(interpolate.interp_tf(tf.reshape(tf.cast(x, tf.float32), [-1]), k, pk), x.shape), tf.complex64)

  # And initial conditions
  initial_conditions = flowpm.linear_field([FLAGS.nc, FLAGS.nc, 10 * FLAGS.nc],
                                           [FLAGS.box_size, FLAGS.box_size,
                                           10 * FLAGS.box_size],
                                           pk_fun,
                                           batch_size=1)

  r = tf.linspace(0., 2000, FLAGS.nsteps)
  a = flowpm.tfbackground.a_of_chi(cosmology, r)

  # Sample particles, using LPT up to the border of the lightcone
  # WARNING: this is probably very approximate
  state = flowpm.lpt_init(cosmology, initial_conditions, a[-1])

  
   # Evolve particles from initial state down to a=af
  final_state = flowpm.nbody(cosmology, state, a, [FLAGS.nc, FLAGS.nc, 10 * FLAGS.nc])         

  # Retrieve final density field i.e interpolate the particles to the mesh
  final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), final_state[0])
  final_field=tf.reshape(final_field, [FLAGS.nc, FLAGS.nc, 10 * FLAGS.nc])
  return final_field
#%%

#bvv=tf.reshape(final_field, [nc, nc,10* nc])

#%%
@tf.function
def compute_jacobian(Omega_c, sigma8):
  """ Function that actually computes the Jacobian of a given statistics
  """
  params = tf.stack([Omega_c, sigma8])
  with tf.GradientTape() as tape:
    tape.watch(params)
    final_field = compute_Nbody(params[0], params[1])
    k, power_spectrum = pkl(final_field,kmin=0.01,dk=0.01,shape=final_field.shape,boxsize=np.array([FLAGS.box_size, FLAGS.box_size, 10 * FLAGS.box_size]))

  return final_field, tape.jacobian(power_spectrum, params,experimental_use_pfor=False), k, power_spectrum



def main(_):
  # Query the jacobian
   final_field, jacobian, k, power_spectrum = compute_jacobian(tf.convert_to_tensor(FLAGS.Omega_c,
                                                             dtype=tf.float32),
                                        tf.convert_to_tensor(FLAGS.sigma8,
                                                             dtype=tf.float32))
 
  # Saving results in requested filename
   pickle.dump({'final_field': final_field.numpy(), 'k': k.numpy(),
                'power_spectrum': power_spectrum.numpy(), 'jac': jacobian.numpy()},
               open(FLAGS.filename, "wb"))
  


if __name__ == "__main__":
  app.run(main)