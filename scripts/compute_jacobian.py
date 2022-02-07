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
from flowpm.NLA_IA import k_IA
from flowpm.fourier_smoothing import fourier_smoothing
from flowpm.tfbackground import rad_comoving_distance

flags.DEFINE_string("filename", "jac_ps__halo.pkl", "Output filename")
flags.DEFINE_string("pgd_params", "results_fit_PGD_205_128.pkl",
                    "PGD parameter files")
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
flags.DEFINE_float(
    "Aia", 1.,
    "The amplitude parameter A describes the strength of the tidal coupling")
flags.DEFINE_float(
    "sigma_k", 0.1,
    "Value of the sigma in two-dimensional smoothing kernel used to compute the projected tidal shear"
)

FLAGS = flags.FLAGS


@tf.function
def compute_kappa(Omega_c, sigma8, Omega_b, n_s, h, w0, Aia, pgdparams):
    """ Computes a convergence map using ray-tracing through an N-body for a given
    set of cosmological parameters
    """
    # Instantiates a cosmology with desired parameters
    cosmology = flowpm.cosmology.Planck15(Omega_c=Omega_c,
                                          sigma8=sigma8,
                                          Omega_b=Omega_b,
                                          n_s=n_s,
                                          h=h,
                                          w0=w0)

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
    pk_fun = lambda x: tf.cast(
        tf.reshape(
            interpolate.interp_tf(tf.reshape(tf.cast(x, tf.float32), [-1]), k,
                                  pk), x.shape), tf.complex64)
    initial_conditions = flowpm.linear_field(
        [FLAGS.nc, FLAGS.nc, FLAGS.nc],
        [FLAGS.box_size, FLAGS.box_size, FLAGS.box_size],
        pk_fun,
        batch_size=FLAGS.batch_size)
    initial_state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)

    # Run the Nbody
    states = flowpm.nbody(cosmology,
                          initial_state,
                          stages, [FLAGS.nc, FLAGS.nc, FLAGS.nc],
                          return_intermediate_states=True,
                          pgdparams=pgdparams)

    # Extract the lensplanes
    lensplanes = []
    matrix = flowpm.raytracing.rotation_matrices()
    for i, j in zip(range(len(a_center)), cycle(range(6))):
        plane = flowpm.raytracing.density_plane(
            states[::-1][i][1],
            [FLAGS.nc, FLAGS.nc, FLAGS.nc],
            FLAGS.nc // 2,
            width=FLAGS.nc,
            plane_resolution=2048,
            rotation=matrix[j],
            shift=flowpm.raytracing.random_2d_shift(),
        )

        fourier_smoothing(plane, sigma=1.024, resolution=2048)
        lensplanes.append((r_center[i], states[::-1][i][0], plane))
    xgrid, ygrid = np.meshgrid(
        np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                    endpoint=False),  # range of X coordinates
        np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                    endpoint=False))  # range of Y coordinates

    coords = np.stack([xgrid, ygrid], axis=0)
    c = coords.reshape([2, -1]).T / 180. * np.pi  # convert to rad from deg
    # Create array of source redshifts
    lens_source = states[::-1][11][1]
    lens_source_a = states[::-1][11][0]
    z_source = 1 / lens_source_a - 1
    m = flowpm.raytracing.convergenceBorn(cosmology,
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
    kmap_IA = m - k_ia
    return kmap_IA, states, lensplanes, r_center, a_center


def desc_y1_analysis(kmap):
    """
  Adds noise and apply smoothing we might expect in DESC Y1 SRD setting
  """
    ngal = 10  # gal/arcmin **2
    pix_scale = FLAGS.field_size / FLAGS.field_npix * 60  # arcmin
    ngal_per_pix = ngal * pix_scale**2  # galaxies per pixels
    sigma_e = 0.26 / np.sqrt(2 * ngal_per_pix)  # Rescaled noise sigma
    sigma_pix = 1. / pix_scale  # Smooth at 1 arcmin
    # Add noise
    kmap = kmap + sigma_e * tf.random.normal(kmap.shape)
    # Add smoothing
    kmap = fourier_smoothing(kmap,
                             sigma=sigma_pix,
                             resolution=FLAGS.field_npix)
    return kmap


def rebin(a, shape):
    sh = shape, a.shape[0] // shape
    return tf.math.reduce_mean(tf.reshape(a, sh), axis=-1)


@tf.function
def compute_jacobian(Omega_c, sigma8, Omega_b, n_s, h, w0, Aia, pgdparams):
    """ Function that actually computes the Jacobian of a given statistics
    """
    params = tf.stack([Omega_c, sigma8, Omega_b, n_s, h, w0, Aia])
    with tf.GradientTape() as tape:
        tape.watch(params)
        kmap_IA, states, lensplanes, r_center, a_center = compute_kappa(
            params[0], params[1], params[2], params[3], params[4], params[5],
            params[6], pgdparams)

        # Adds realism to convergence map
        kmap = desc_y1_analysis(kmap_IA)
        # Compute power spectrum
        ell, power_spectrum = DHOS.statistics.power_spectrum(
            kmap[0], FLAGS.field_size, FLAGS.field_npix)

        # Keep only ell between 300 and 3000
        ell = ell[2:46]
        power_spectrum = power_spectrum[2:46]

        # Further reducing the nnumber of points
        ell = rebin(ell, 11)
        power_spectrum = rebin(power_spectrum, 11)
    jac = tape.jacobian(power_spectrum,
                        params,
                        experimental_use_pfor=False,
                        parallel_iterations=1)

    return kmap, lensplanes, r_center, a_center, jac, ell, power_spectrum


def main(_):
    # Query the jacobian
    # Loading PGD parameters
    t = time.time()
    with open(FLAGS.pgd_params, "rb") as f:
        pgd_data = pickle.load(f)
        pgdparams = pgd_data['params']

    kmap, lensplanes, r_center, a_center, jac_ps, ell, ps = compute_jacobian(
        tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32),
        tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32),
        tf.convert_to_tensor(FLAGS.Omega_b, dtype=tf.float32),
        tf.convert_to_tensor(FLAGS.n_s, dtype=tf.float32),
        tf.convert_to_tensor(FLAGS.h, dtype=tf.float32),
        tf.convert_to_tensor(FLAGS.w0, dtype=tf.float32),
        tf.convert_to_tensor(FLAGS.Aia, dtype=tf.float32), pgdparams)
    # Saving results in requested filename
    pickle.dump(
        {
            'a': a_center,
            'lensplanes': lensplanes,
            'r': r_center,
            'kmap': kmap.numpy(),
            'ell': ell.numpy(),
            'ps': ps.numpy(),
            'jac_ps': jac_ps.numpy()
        }, open(FLAGS.filename, "wb"))
    print("computation time for Jacobian", time.time() - t)


if __name__ == "__main__":
    app.run(main)
