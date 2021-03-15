import tensorflow as tf
import numpy as np
import DifferentiableHOS as DHOS
import flowpm
# TODO: This is very annoying, we need to fix the FlowPM imports
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
import flowpm.raytracing as raytracing
import pickle
from absl import app
from absl import flags

flags.DEFINE_string("filename", "results.pkl", "Output filename")
flags.DEFINE_float("Omega_c", 0.2589, "Fiducial CDM fraction")
flags.DEFINE_float("sigma8", 0.8159, "Fiducial sigma_8 value")
flags.DEFINE_integer("nc", 32, "Transverse size of the cube")
flags.DEFINE_integer("plane_size", 32, "Number of pixels in x,y")
flags.DEFINE_float("box_size", 200, "Transverse box size [Mpc/h]")
flags.DEFINE_float("field", 5., "Transverse size of the lensing field [deg]")
flags.DEFINE_integer("nsteps", 4, "Number of time steps in the lightcone")
flags.DEFINE_float("z_source", 1., "Source redshift")

FLAGS = flags.FLAGS


# Small utility functions
def z2a(z):
    return 1. / (1. + z)


def a2z(a):
    return (1. / a) - 1.


# TODO: This compute kappa function should actually go to FlowPM
@tf.function
def compute_kappa(Omega_c, sigma8):
  """ Computes a convergence map using ray-tracing through an N-body for a given
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

  # Perform lightcone computation and extract lens planes
  state, lps_a, lps = raytracing.lightcone(cosmology, state,
                                           tf.reverse(a, [0]),
                                           [FLAGS.nc, FLAGS.nc, 10 * FLAGS.nc],
                                           FLAGS.field * 60. / FLAGS.plane_size,
                                           FLAGS.plane_size)

  # Compute source scale factor
  a_s = z2a(tf.convert_to_tensor(FLAGS.z_source, dtype=tf.float32))
  ds = flowpm.tfbackground.rad_comoving_distance(cosmology, a_s)

  # Perform ray tracing
  k_map = raytracing.Born(lps_a,
                          lps,
                          ds,
                          [FLAGS.nc, FLAGS.nc, 10 * FLAGS.nc],
                          [FLAGS.box_size, FLAGS.box_size,
                           10 * FLAGS.box_size],
                          FLAGS.plane_size,
                          FLAGS.field,
                          cosmology)
  return k_map


@tf.function
def compute_jacobian(Omega_c, sigma8):
  """ Function that actually computes the Jacobian of a given statistics
  """
  params = tf.stack([Omega_c, sigma8])
  with tf.GradientTape() as tape:
    tape.watch(params)
    kmap = compute_kappa(params[0], params[1])
    # Here we could have a condition to compute various stats on the same
    # kappa map
    ell, power_spectrum = DHOS.statistics.power_spectrum(kmap,
                                                         FLAGS.field,
                                                         FLAGS.plane_size)
  return kmap, tape.jacobian(power_spectrum, params,
                             experimental_use_pfor=False), ell, power_spectrum


def main(_):
  # Query the jacobian
  kmap, jacobian, ell, ps = compute_jacobian(tf.convert_to_tensor(FLAGS.Omega_c,
                                                            dtype=tf.float32),
                                       tf.convert_to_tensor(FLAGS.sigma8,
                                                            dtype=tf.float32))
  # Saving results in requested filename
  pickle.dump({'map': kmap.numpy(), 'ell': ell.numpy(),
               'ps': ps.numpy(), 'jac': jacobian.numpy()},
              open(FLAGS.filename, "wb"))


if __name__ == "__main__":
  app.run(main)
