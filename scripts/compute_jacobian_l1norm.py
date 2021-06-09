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


flags.DEFINE_string("filename", "results_l1norm.pkl", "Output filename")
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
flags.DEFINE_integer("nmaps", 10, "Number maps to generate.")


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
  k = tf.constant(np.logspace(-4, 1, 128), dtype=tf.float32)
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

  coords = np.stack([xgrid, ygrid], axis=0) 
  c = coords.reshape([2, -1]).T / 180.*np.pi # convert to rad from deg
  # Create array of source redshifts
  z_source = tf.constant([1.])
  m = flowpm.raytracing.convergenceBorn(cosmology,
                                        lensplanes,
                                        dx=FLAGS.box_size / 256,
                                        dz=FLAGS.box_size,
                                        coords=c,
                                        z_source=z_source)

  m = tf.reshape(m, [1, FLAGS.field_npix, FLAGS.field_npix, -1])

  return m, lensplanes, r_center, a_center


def desc_y1_analysis(kmap):
  """
  Adds noise and apply smoothing we might expect in DESC Y1 SRD setting
  """
  ngal = 10                          # gal/arcmin **2
  pix_scale = 5/512*60              # arcmin
  ngal_per_pix = ngal * pix_scale**2 # galaxies per pixels (I think)
  sigma_e = 0.26 / np.sqrt(ngal_per_pix) # Rescaled noise sigma
  sigma_pix=2./pix_scale             # Smooth at 1 arcmin

  # Add noise
  kmap  = kmap + sigma_e * tf.random.normal(kmap.shape)
  # Add smoothing
  kmap = tfa.image.gaussian_filter2d(kmap,21,sigma=sigma_pix)
  return kmap

def rebin(a, shape):
    sh = shape,a.shape[0]//shape
    return tf.math.reduce_mean(tf.reshape(a,sh),axis=-1)

@tf.function
def compute_jacobian(Omega_c, sigma8):
  """ Function that actually computes the Jacobian of a given statistics
    """
  params = tf.stack([Omega_c, sigma8])
  with tf.GradientTape() as tape:
    tape.watch(params)
    m, lensplanes, r_center, a_center = compute_kappa(params[0], params[1])
    
    # Adds realism to convergence map
    kmap = desc_y1_analysis(m)
    
    # Compute power spectrum
    #ell, power_spectrum = DHOS.statistics.power_spectrum(
    #    kmap[0, :, :, -1], FLAGS.field_size, FLAGS.field_npix)

    # Keep only ell below 3000
    #ell = ell[:21] 
    #power_spectrum = power_spectrum[:21]

    # Further reducing the nnumber of points
    #ell=rebin(ell,7)
    #power_spectrum=rebin(power_spectrum,7)

    # # Compute the peak counts
    # vmin=-0.025
    # vmax=0.1
    # bins = tf.linspace(vmin, vmax, 8)
    # counts, edges = DHOS.statistics.peaks_histogram_tf(tf.clip_by_value(kmap[0, :, :, -1], vmin, vmax),
    #                                                   bins=bins)

    # Compute l1norm
    l1 = DHOS.statistics.l1norm(kmap[...,0], nbins=7, value_range=[-0.05, 0.05])[1][0]

#     stat = tf.stack([power_spectrum, l1])
  jac = tape.jacobian(l1, params, experimental_use_pfor=False)

  return m, kmap, lensplanes, r_center, a_center, jac, l1# edges, counts

def main(_):
  for i in range(FLAGS.nmaps):
    t = time.time()
    # Query the jacobian
    m, kmap, lensplanes, r_center, a_center, jac, l1 = compute_jacobian(
      tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32), tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32))
    # Saving results in requested filename
    pickle.dump(
      {
          'a': a_center,
          'lensplanes': lensplanes,
          'r': r_center,
          'map': m.numpy(),
          'kmap': kmap.numpy(),
          'jac': jac.numpy(),
          'l1': l1.numpy()
      }, open(FLAGS.filename+'_%d'%i, "wb"))
    print("iter",i, "took", time.time()-t)

if __name__ == "__main__":
  app.run(main)
