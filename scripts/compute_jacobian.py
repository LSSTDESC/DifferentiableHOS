import tensorflow as tf
import numpy as np
import DifferentiableHOS as DHOS
import flowpm
import flowpm.raytracing as raytracing
import pickle
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
from absl import app
from absl import flags
from flowpm.tfpower import linear_matter_power
import astropy.units as u
from itertools import cycle
import tensorflow_addons as tfa


flags.DEFINE_string("filename", "results.pkl", "Output filename")
flags.DEFINE_float("Omega_c", 0.2589, "Fiducial CDM fraction")
flags.DEFINE_float("sigma8", 0.8159, "Fiducial sigma_8 value")
flags.DEFINE_integer("nc", 64,
                     "Number of transverse voxels in the simulation volume")
flags.DEFINE_integer("field_npix", 512,
                     "Number of pixels in the lensing field")
flags.DEFINE_float("box_size", 100.,
                   "Transverse comoving size of the simulation volume")
flags.DEFINE_float("field_size", 5., "TSize of the lensing field in degrees")
flags.DEFINE_integer("n_lens", 22, "Number of lensplanes in the lightcone")
flags.DEFINE_float("batch_size", 1, "Number of simulations to run in parallel")

FLAGS = flags.FLAGS


@tf.function
def compute_kappa(Omega_c, sigma8):
  """ Computes a convergence map using ray-tracing through an N-body for a given
    set of cosmological parameters
    """
  # Instantiates a cosmology with desired parameters
  cosmology = flowpm.cosmology.Planck15(Omega_c=Omega_c, sigma8=sigma8)

  # Schedule the center of the lensplanes we want for ray tracing
  r = tf.linspace(0., FLAGS.box_size * FLAGS.n_lens, FLAGS.n_lens + 1)
  r_center = 0.5 * (r[1:] + r[:-1])

  # Retrieve the scale factor corresponding to these distances
  a = flowpm.tfbackground.a_of_chi(cosmology, r)
  a_center = flowpm.tfbackground.a_of_chi(cosmology, r_center)

  # We run 4 steps from initial scale factor to start of raytracing
  init_stages = tf.linspace(0.1, a[-1], 4)
  # Then one step per lens plane
  stages = tf.concat([init_stages, a_center[::-1]], axis=0)

  # Create some initial conditions
  k = tf.constant(np.logspace(-4, 1, 256), dtype=tf.float32)
  pk = linear_matter_power(cosmology, k)
  pk_fun = lambda x: tf.cast(tf.reshape(interpolate.interp_tf(tf.reshape(tf.cast(x, tf.float32), [-1]), k, pk), x.shape), tf.complex64)
  initial_conditions = flowpm.linear_field(
          [FLAGS.nc, FLAGS.nc, FLAGS.nc],
          [FLAGS.box_size, FLAGS.box_size, FLAGS.box_size],
          pk_fun,
          batch_size=1)
  initial_state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)

  # Run the Nbody
  states = flowpm.nbody(cosmology,
                        initial_state,
                        stages, [FLAGS.nc, FLAGS.nc, FLAGS.nc],
                        return_intermediate_states=True)

  # Extract the lensplanes
  lensplanes = []
  matrix = flowpm.raytracing.rotation_matrices()
  for i, j in zip(range(len(a_center)), cycle(range(6))):
    plane = flowpm.raytracing.density_plane(
        states[::-1][i][1],
        [FLAGS.nc, FLAGS.nc, FLAGS.nc],
        FLAGS.nc // 2,
        width=FLAGS.nc,
        plane_resolution=256,
        rotation=matrix[j],
        shift=flowpm.raytracing.random_2d_shift(),
    )

    plane = tf.expand_dims(plane, axis=-1)
    lensplanes.append((r_center[i], states[::-1][i][0], plane[..., 0]))
  xgrid, ygrid = np.meshgrid(
      np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                  endpoint=False),  # range of X coordinates
      np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                  endpoint=False))  # range of Y coordinates

  coords = np.stack([xgrid, ygrid], axis=0) * u.deg
  c = coords.reshape([2, -1]).T.to(u.rad)
  # Create array of source redshifts
  z_source = tf.linspace(0.5, 1, 4)
  m = flowpm.raytracing.convergenceBorn(cosmology,
                                        lensplanes,
                                        dx=FLAGS.box_size / 256,
                                        dz=FLAGS.box_size,
                                        coords=c,
                                        z_source=z_source)

  m = tf.reshape(m, [1, FLAGS.field_npix, FLAGS.field_npix, -1])

  return m, lensplanes, r_center, a_center


def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return tf.math.reduce_mean(tf.math.reduce_mean(tf.reshape(a,sh),axis=-1),axis=1)


@tf.function
def compute_jacobian(Omega_c, sigma8):
  """ Function that actually computes the Jacobian of a given statistics
    """
  params = tf.stack([Omega_c, sigma8])
  with tf.GradientTape() as tape:
    tape.watch(params)
    m, lensplanes, r_center, a_center = compute_kappa(params[0], params[1])
    ell0, power_spectrum0 = DHOS.statistics.power_spectrum(
        m[0, :, :, -1], FLAGS.field_size, FLAGS.field_npix)
    a=tf.reshape(ell0,(4,64))
    ell=tf.reshape(rebin(a,(4,8)),(1,32))
    power_spectrum=interpolate.interp_tf(tf.reshape(ell, [-1]), ell0,power_spectrum0)
    # k1 = tf.where(ell < 650, False, True)
    # ell = tf.boolean_mask(ell, tf.math.logical_not(k1))
    # power_spectrum = tf.boolean_mask(power_spectrum, tf.math.logical_not(k1))

  return m, lensplanes, r_center, a_center, tape.jacobian(
      power_spectrum, params, experimental_use_pfor=False), ell, power_spectrum


def main(_):
  # Query the jacobian
  m, lensplanes, r_center, a_center, jacobian, ell, ps = compute_jacobian(
      tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32), tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32))
  # Saving results in requested filename
  pickle.dump(
      {
          'a': a_center,
          'lensplanes': lensplanes,
          'r': r_center,
          'map': m.numpy(),
          'ell': ell.numpy(),
          'ps': ps.numpy(),
          'jac': jacobian.numpy()
      }, open(FLAGS.filename, "wb"))

if __name__ == "__main__":
  app.run(main)
