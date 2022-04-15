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
from scipy.stats import norm
from flowpm.NLA_IA import k_IA
import time
import tensorflow_probability as tfp
from flowpm.fourier_smoothing import fourier_smoothing
from flowpm.neural_ode_nbody import  make_neural_ode_fn
from flowpm.tfbackground import rad_comoving_distance

flags.DEFINE_string("filename", "jac_output_",
                    "Output filename")
flags.DEFINE_string("correction_params", "/global/homes/d/dlan/flowpm/notebooks/camels_25_64_pkloss.params",
                    "Correction parameter files")
flags.DEFINE_float("Omega_c", 0.2589, "Fiducial CDM fraction")
flags.DEFINE_float("Omega_b", 0.04860, "Fiducial baryonic matter fraction")
flags.DEFINE_float("sigma8", 0.8159, "Fiducial sigma_8 value")
flags.DEFINE_float("n_s", 0.9667, "Fiducial n_s value")
flags.DEFINE_float("h", 0.6774, "Fiducial Hubble constant value")
flags.DEFINE_float("w0", -1.0, "Fiducial w0 value")
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
flags.DEFINE_integer("nmaps", 10, "Number maps to generate.")
flags.DEFINE_float(
    "Aia", 0.,
    "The amplitude parameter A describes the strength of the tidal coupling")
flags.DEFINE_boolean("Convergence_map", True, "True if we want to compute the simulate the convergence map")
flags.DEFINE_boolean("Power_Spectrum", False, "True if we want to compute the jacobian of the ps")
flags.DEFINE_boolean("Peak_counts", False, "True if we want to compute the jacobian of the pk")
flags.DEFINE_boolean("l1_norm", False, "True if we want to compute the jacobian of the l1norm")

FLAGS = flags.FLAGS

@tf.function
def compute_kappa(Omega_c, sigma8, Omega_b, n_s, h, w0, Aia):
  """ Computes a convergence map using ray-tracing through an N-body for a given
    set of cosmological parameters
    """
  
  cosmology = flowpm.cosmology.Planck15(
      Omega_c=Omega_c, sigma8=sigma8, Omega_b=Omega_b, n_s=n_s, h=h, w0=w0)
  r = tf.linspace(0., FLAGS.box_size * FLAGS.n_lens, FLAGS.n_lens + 1)
  r_center = 0.5 * (r[1:] + r[:-1])
  a = flowpm.tfbackground.a_of_chi(cosmology, r)
  a_center = flowpm.tfbackground.a_of_chi(cosmology, r_center)
  stages = a_center[::-1]
  k = tf.constant(np.logspace(-4, 1, 128), dtype=tf.float32)
  pk = linear_matter_power(cosmology, k)
  pk_fun = lambda x: tf.cast(
      tf.reshape(
          interpolate.interp_tf(
              tf.reshape(tf.cast(x, tf.float32), [-1]), k, pk), x.shape), tf.
      complex64)
  initial_conditions = flowpm.linear_field(
      [FLAGS.nc, FLAGS.nc, FLAGS.nc],
      [FLAGS.box_size, FLAGS.box_size, FLAGS.box_size],
      pk_fun,
      batch_size=FLAGS.batch_size)
  initial_state = flowpm.lpt_init(cosmology, initial_conditions,
                                  0.14285714254594556)
  initial_state = initial_state[0:2]
  res = tfp.math.ode.DormandPrince(
      rtol=1e-5, atol=1e-5).solve(
          ode_function,
          0.14285714254594556,
          initial_state,
          solution_times=stages,
          constants={
              'Omega_c': Omega_c,
              'sigma8': sigma8,
              'Omega_b': Omega_b,
              'n_s': n_s,
              'h': h,
              'w0': w0,
          })
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

    plane = fourier_smoothing(plane, sigma=1.024, resolution=2048)
    lensplanes.append((r_center[i], res.times[::-1][i], plane))
  xgrid, ygrid = np.meshgrid(
      np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                  endpoint=False),  # range of X coordinates
      np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                  endpoint=False))  # range of Y coordinates

  coords = np.stack([xgrid, ygrid], axis=0)
  c = coords.reshape([2, -1]).T / 180. * np.pi  # convert to rad from deg
  lens_source = res.states[::-1][-1]
  lens_source_a = res.times[::-1][-1]
  z_source = 1 / lens_source_a - 1
  m = flowpm.raytracing.convergenceBorn(
      cosmology,
      lensplanes,
      dx=FLAGS.box_size / 2048,
      dz=FLAGS.box_size,
      coords=c,
      z_source=z_source,
      field_npix=FLAGS.field_npix)
  r_source = rad_comoving_distance(cosmology, lens_source_a)
  plane_source = flowpm.raytracing.density_plane(
        lens_source,
        [FLAGS.nc, FLAGS.nc, FLAGS.nc],
        FLAGS.nc // 2,
        width=FLAGS.nc,
        plane_resolution=2048,
    )
  im_IA = flowpm.raytracing.interpolation(plane_source,
                                            dx=FLAGS.box_size / 2048,
                                            r_center=r_source,
                                            field_npix=FLAGS.field_npix,
                                            coords=c)
  k_ia = k_IA(cosmology, lens_source_a, im_IA, Aia)
  kmap_IA =  m - k_ia
  return  kmap_IA


def desc_y1_analysis(kmap,hos=True):
  """
  Adds noise and apply smoothing we might expect in DESC Y1 SRD setting
  """
  ngal = 10 
  pix_scale = FLAGS.field_size / FLAGS.field_npix * 60 
  ngal_per_pix = ngal * pix_scale**2  
  sigma_e = 0.26 / np.sqrt(2 * ngal_per_pix) 
  sigma_pix = 4. / pix_scale  
  kmap = kmap + sigma_e * tf.random.normal(kmap.shape)
  if hos:
      return kmap
  else:
      kmap = fourier_smoothing(kmap,sigma=sigma_pix,resolution=FLAGS.field_npix)
      return kmap
    
    

def rebin(a, shape):
  sh = shape, a.shape[0] // shape
  return tf.math.reduce_mean(tf.reshape(a, sh), axis=-1)



@tf.function
def compute_jacobian_ps(
    Omega_c,
    sigma8,
    Omega_b,
    n_s,
    h,
    w0,Aia):
  """ Function that actually computes the Jacobian of a given statistics
    """
  params = tf.stack([Omega_c, sigma8, Omega_b, n_s, h, w0,Aia])
  with tf.GradientTape() as tape:
    tape.watch(params)
    m = compute_kappa(params[0], params[1], params[2], params[3], params[4],
                      params[5],params[6])
    # Adds realism to convergence map
    kmap = desc_y1_analysis(m, hos=False)
    # Compute power spectrum
    ell, power_spectrum = DHOS.statistics.power_spectrum(
        kmap[0], FLAGS.field_size, FLAGS.field_npix)
    # Keep only ell between 300 and 3000
    ell = ell[2:46]
    power_spectrum = power_spectrum[2:46]

    # Further reducing the nnumber of points
    ell = rebin(ell, 11)
    ps = rebin(power_spectrum, 11)
  jac = tape.jacobian(
      ps, params, experimental_use_pfor=False, parallel_iterations=1)
  return jac, ps, ell, kmap


@tf.function
def compute_jacobian_pk(Omega_c, sigma8, Omega_b, n_s, h, w0, Aia):
    """ Function that actually computes the Jacobian of a given statistics
    """
    params = tf.stack([Omega_c, sigma8, Omega_b, n_s, h, w0, Aia,])
    with tf.GradientTape() as tape:
        tape.watch(params)
        kmap = compute_kappa(params[0], params[1], params[2], params[3],
                                params[4], params[5], params[6])

        # Adds realism to convergence map
        kmap = desc_y1_analysis(kmap)
        # Compute the peak counts
        counts, bins = DHOS.statistics.peaks_histogram_tf_mulscale(
                 kmap, nscales=5, bins=tf.linspace(-.1, 1., 8))
        counts = counts[3:]
        counts = tf.stack(counts)
    jac = tape.jacobian(counts,
                        params,
                        experimental_use_pfor=False,
                        parallel_iterations=1)

    return jac,counts,kmap,bins
    
    
    
@tf.function
def compute_jacobian_l1norm(Omega_c, sigma8, Omega_b, n_s, h, w0, Aia):
    """ Function that actually computes the Jacobian of a given statistics
    """
    params = tf.stack([Omega_c, sigma8, Omega_b, n_s, h, w0, Aia])
    with tf.GradientTape() as tape:
        tape.watch(params)
        kmap_IA = compute_kappa(
            params[0], params[1], params[2], params[3], params[4], params[5],
            params[6])
        kmap = desc_y1_analysis(kmap_IA)

        l1norm = DHOS.statistics.l1norm(kmap,
                                        nscales=5,
                                        nbins=8,
                                        value_range=[-.1, 1.])[3:]
     
    jac = tape.jacobian(l1norm,
                        params,
                        experimental_use_pfor=False,
                        parallel_iterations=1)

    return jac, l1norm ,kmap
    
    
    
def main(_):
  global ode_function
  ode_function=make_neural_ode_fn(FLAGS.nc,FLAGS.batch_size,FLAGS.correction_params)
  if FLAGS.Convergence_map:
      t = time.time()
      for i in range(FLAGS.nmaps):
          kmap_IA=compute_kappa(
              tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.Omega_b, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.n_s, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.h, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.w0, dtype=tf.float32),
          tf.convert_to_tensor(FLAGS.Aia, dtype=tf.float32))
          pickle.dump(
              {    'kmap':kmap_IA,
              },
              open(FLAGS.filename + 'kmap_%d' % i+'.pkl', "wb"))
          print("iter", i, "took", time.time() - t)
  if FLAGS.Power_Spectrum:
      t = time.time()
      for i in range(FLAGS.nmaps):
            
          jac, ps, ell, kmap = compute_jacobian_ps(
              tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.Omega_b, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.n_s, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.h, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.w0, dtype=tf.float32),
          tf.convert_to_tensor(FLAGS.Aia, dtype=tf.float32))
          pickle.dump(
              {    'kmap': kmap.numpy(),
                  'ell': ell.numpy(),
                  'ps': ps.numpy(),
                  'jac': jac.numpy()
              },
              open(FLAGS.filename + 'ps_%d' % i+'.pkl', "wb"))
          print("iter", i, "took", time.time() - t)
  if FLAGS.Peak_counts:
    t = time.time()
    for i in range(FLAGS.nmaps):
        jac,counts,kmap, bins=compute_jacobian_pk(
              tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.Omega_b, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.n_s, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.h, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.w0, dtype=tf.float32),
          tf.convert_to_tensor(FLAGS.Aia, dtype=tf.float32))
        pickle.dump(
              {   'kmap': kmap.numpy(),
                  'jac': jac.numpy(),
                'counts': counts.numpy(),
                'bin': bins.numpy()
              },
              open(FLAGS.filename + 'pcounts_%d' % i+'.pkl', "wb"))
        print("iter", i, "took", time.time() - t)
  if FLAGS.l1_norm:
    t = time.time()
    for i in range(FLAGS.nmaps):
        jac, l1 ,kmap=compute_jacobian_l1norm(
              tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.Omega_b, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.n_s, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.h, dtype=tf.float32),
              tf.convert_to_tensor(FLAGS.w0, dtype=tf.float32),
          tf.convert_to_tensor(FLAGS.Aia, dtype=tf.float32))
        pickle.dump(
              { 
                'kmap': kmap.numpy(),
                'jac': jac.numpy(),
                'l1': l1.numpy()
            }, open(FLAGS.filename + 'l1norm_%d' % i+'.pkl', "wb"))
        print("iter", i, "took", time.time() - t)
    



if __name__ == "__main__":
  app.run(main)

