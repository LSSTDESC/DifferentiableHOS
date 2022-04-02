import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ[
    'XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/global/common/software/nersc/cos1.3/cuda/11.3.0'
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
import time
from flowpm import tfpm
import tensorflow_probability as tfp
from scipy.stats import norm
from flowpm.fourier_smoothing import fourier_smoothing
from flowpm.tfbackground import rad_comoving_distance

flags.DEFINE_string(
    "filename",
    "/pscratch/sd/d/dlan/maps_noia_ode/maps5_nocorr/results_maps.pkl",
    "Output filename")
flags.DEFINE_float("Omega_c", 0.2589, "Fiducial CDM fraction")
flags.DEFINE_float("sigma8", 0.8159, "Fiducial sigma_8 value")
flags.DEFINE_integer("nc", 128,
                     "Number of transverse voxels in the simulation volume")
flags.DEFINE_integer("field_npix", 1024,
                     "Number of pixels in the lensing field")
flags.DEFINE_float("box_size", 205.,
                   "Transverse comoving size of the simulation volume")
flags.DEFINE_float("field_size", 5., "TSize of the lensing field in degrees")
flags.DEFINE_integer("n_lens", 11, "Number of lensplanes in the lightcone")
flags.DEFINE_integer("batch_size", 1,
                     "Number of simulations to run in parallel")
flags.DEFINE_integer("nmaps", 100, "Number maps to generate.")

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
  stages = a_center[::-1]
  # Create some initial conditions
  k = tf.constant(np.logspace(-4, 1, 128), dtype=tf.float32)
  pk = linear_matter_power(cosmology, k)
  pk_fun = lambda x: tf.cast(
      tf.reshape(
          interpolate.interp_tf(tf.reshape(tf.cast(x, tf.float32), [-1]), k, pk
                                ), x.shape), tf.complex64)
  initial_conditions = flowpm.linear_field(
      [FLAGS.nc, FLAGS.nc, FLAGS.nc],
      [FLAGS.box_size, FLAGS.box_size, FLAGS.box_size],
      pk_fun,
      batch_size=FLAGS.batch_size)
  initial_state = flowpm.lpt_init(cosmology, initial_conditions,
                                  0.14285714254594556)
  initial_state = initial_state[0:2]
  res = tfp.math.ode.DormandPrince(rtol=1e-5, atol=1e-5).solve(
      flowpm.make_ode_fn([FLAGS.nc, FLAGS.nc, FLAGS.nc]),
      0.14285714254594556,
      initial_state,
      solution_times=stages,
      constants={
          'Omega_c': Omega_c,
          'sigma8': sigma8
      })
  # Extract the lensplanes
  lensplanes = []
  matrix = flowpm.raytracing.rotation_matrices()
  for i, j in zip(range(len(a_center)), cycle(range(6))):
    plane = flowpm.raytracing.density_plane(
        res.states[::-1][i],
        [FLAGS.nc, FLAGS.nc, FLAGS.nc],
        FLAGS.nc // 2,
        width=FLAGS.nc,
        plane_resolution=2048,
        rotation=matrix[j],
        shift=flowpm.raytracing.random_2d_shift(),
    )

    fourier_smoothing(plane, sigma=1.024, resolution=2048)
    lensplanes.append((r_center[i], res.times[::-1][i], plane))
  xgrid, ygrid = np.meshgrid(
      np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                  endpoint=False),  # range of X coordinates
      np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                  endpoint=False))  # range of Y coordinates

  coords = np.stack([xgrid, ygrid], axis=0)
  c = coords.reshape([2, -1]).T / 180. * np.pi  # convert to rad from deg
  # Create array of source redshifts
  z_source = tf.convert_to_tensor([1.034], dtype=tf.float32)
  m = flowpm.raytracing.convergenceBorn(cosmology,
                                        lensplanes,
                                        dx=FLAGS.box_size / 2048,
                                        dz=FLAGS.box_size,
                                        coords=c,
                                        z_source=z_source,
                                        field_npix=FLAGS.field_npix)

  return m, res, initial_conditions


def main(_):
  for i in range(FLAGS.nmaps):
    t = time.time()
    m, res, initial_conditions = compute_kappa(
        tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32),
        tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32))
    # Saving results in requested filename
    pickle.dump(
        {
            'm': m.numpy(),
            'res_times': res.times.numpy(),
            'res_solutions': res.states.numpy(),
            'initial_conditions': initial_conditions.numpy()
        }, open(FLAGS.filename + '_%d' % i, "wb"))
    print("iter", i, "took", time.time() - t)


if __name__ == "__main__":
  app.run(main)
