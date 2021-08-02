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

flags.DEFINE_string("filename", "results_jac_ps.pkl", "Output filename")
flags.DEFINE_float("Omega_c", 0.2589, "Fiducial CDM fraction")
flags.DEFINE_float("sigma8", 0.8159, "Fiducial sigma_8 value")
flags.DEFINE_integer("nc", 128,
                     "Number of transverse voxels in the simulation volume")
flags.DEFINE_integer("field_npix", 1024,
                     "Number of pixels in the lensing field")
flags.DEFINE_float("box_size", 128.,
                   "Transverse comoving size of the simulation volume")
flags.DEFINE_float("field_size", 5., "TSize of the lensing field in degrees")
flags.DEFINE_integer("n_lens", 20, "Number of lensplanes in the lightcone")
flags.DEFINE_float("batch_size", 1, "Number of simulations to run in parallel")
flags.DEFINE_integer("nmaps", 20, "Number maps to generate.")
flags.DEFINE_float("B", 1, "Scale resolution factor")
flags.DEFINE_float("alpha0", 0.01, "alpha0 parameter of PGD correction")
flags.DEFINE_float("mu",-1.659049, "mu parameter of PGD correction")
flags.DEFINE_float("ks", 12.49952, "short range scale parameter of PGD correction")
flags.DEFINE_float("kl", 1.747188, "long range scale parameter of PGD correction")

FLAGS = flags.FLAGS


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
  k = tf.constant(np.logspace(-4, 1, 512), dtype=tf.float32)
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
  dx=[]
  new_states=[]
  ks=FLAGS.ks*0.7*0.5/(FLAGS.nc*2/FLAGS.box_size)
  kl=FLAGS.kl*0.7*0.5/(FLAGS.nc*2/FLAGS.box_size)
  for i in range(len(states)):
        alpha=FLAGS.alpha0*states[i][0]**FLAGS.mu
        dx.append(tfpm.PGD_correction(states[i][1],[FLAGS.nc,FLAGS.nc,FLAGS.nc],alpha,kl,ks,pm_nc_factor=2))
        new_states.append((states[i][0], dx[i]+states[i][1][0]))
  # Extract the lensplanes
  lensplanes = []
  for i in range(len(a_center)):
    plane = flowpm.raytracing.density_plane(new_states[::-1][i][1],
                                            [FLAGS.nc, FLAGS.nc, FLAGS.nc],
                                            FLAGS.nc//2,
                                            width=FLAGS.nc,
                                            plane_resolution=256,
                                            shift=flowpm.raytracing.random_2d_shift())
    plane = tf.expand_dims(plane, axis=-1)
    plane = tf.image.random_flip_left_right(plane)
    plane = tf.image.random_flip_up_down(plane)
    lensplanes.append((r_center[i], states[::-1][i][0], plane[..., 0]))
  xgrid, ygrid = np.meshgrid(
      np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                  endpoint=False),  # range of X coordinates
      np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                  endpoint=False))  # range of Y coordinates
  
  coords = np.stack([xgrid, ygrid], axis=0)*u.deg
  c = coords.reshape([2, -1]).T.to(u.rad)
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
  pix_scale = FLAGS.field_size/FLAGS.field_npix*60              # arcmin
  ngal_per_pix = ngal * pix_scale**2 # galaxies per pixels 
  sigma_e = 0.26 / np.sqrt(2 * ngal_per_pix) # Rescaled noise sigma
  sigma_pix=2./pix_scale             # Smooth at 1 arcmin
  # Add noise
  kmap  = kmap + sigma_e * tf.random.normal(kmap.shape)
  # Add smoothing
  kmap = tfa.image.gaussian_filter2d(kmap,51,sigma=sigma_pix)
  return kmap

def rebin(a, shape):
    sh = shape,a.shape[0]//shape
    return tf.math.reduce_mean(tf.reshape(a,sh),axis=-1)


def compute_jacobian(Omega_c, sigma8):
  """ Function that actually computes the Jacobian of a given statistics
    """
  params = tf.stack([Omega_c, sigma8])
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(params)
    m, lensplanes, r_center, a_center = compute_kappa(params[0], params[1])
    
    # Adds realism to convergence map
    kmap = desc_y1_analysis(m)
    
    # Compute power spectrum
    ell, power_spectrum = DHOS.statistics.power_spectrum(
        kmap[0, :, :, -1], FLAGS.field_size, FLAGS.field_npix)

    # Keep only ell below 3000
    ell = ell[:21] 
    power_spectrum = power_spectrum[:21]

    # Further reducing the nnumber of points
    ell=rebin(ell,7)
    power_spectrum=rebin(power_spectrum,7)
  jac = tape.jacobian(power_spectrum, params, experimental_use_pfor=False)

  return m, kmap, lensplanes, r_center, a_center, jac, ell, power_spectrum

def main(_):
  # Query the jacobian
  m, kmap, lensplanes, r_center, a_center, jac_ps, ell, ps = compute_jacobian(
      tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32), tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32))
  # Saving results in requested filename
  pickle.dump(
      {
          'a': a_center,
          'lensplanes': lensplanes,
          'r': r_center,
          'map': m.numpy(),
          'kmap': kmap.numpy(),
          'ell': ell.numpy(),
          'ps': ps.numpy(),
          'jac_ps': jac_ps.numpy()
      }, open(FLAGS.filename, "wb"))

    

if __name__ == "__main__":
  app.run(main)
