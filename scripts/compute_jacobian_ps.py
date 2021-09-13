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
from flowpm import tfpm
import time
from scipy.stats import norm

flags.DEFINE_string("filename", "jac_1_20.pkl", "Output filename")
flags.DEFINE_string("pgd_params", "results_fit_PGD_205_128.pkl",
                    "PGD parameter files")
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
flags.DEFINE_integer("nmaps", 20, "Number maps to generate.")

FLAGS = flags.FLAGS


def make_power_map(power_spectrum, size, kps=None):
    #Ok we need to make a map of the power spectrum in Fourier space
    k1 = np.fft.fftfreq(size)
    k2 = np.fft.fftfreq(size)
    kcoords = np.meshgrid(k1, k2)
    # Now we can compute the k vector
    k = np.sqrt(kcoords[0]**2 + kcoords[1]**2)
    if kps is None:
        kps = np.linspace(0, 0.5, len(power_spectrum))
    # And we can interpolate the PS at these positions
    ps_map = np.interp(k.flatten(), kps, power_spectrum).reshape([size, size])
    ps_map = ps_map
    return ps_map


def fourier_smoothing(kappa, sigma, resolution):
    im = tf.signal.fft2d(tf.cast(kappa, tf.complex64))
    kps = np.linspace(0, 0.5, resolution)
    filter = norm(0, 1. / (2. * np.pi * sigma)).pdf(kps)
    m = make_power_map(filter, resolution, kps=kps)
    m /= m[0, 0]
    im = tf.cast(tf.reshape(m, [1, resolution, resolution]), tf.complex64) * im
    return tf.cast(tf.signal.ifft2d(im), tf.float32)


@tf.function
def compute_kappa(Omega_c, sigma8, pgdparams):
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
    z_source = tf.constant([1.])
    m = flowpm.raytracing.convergenceBorn(cosmology,
                                          lensplanes,
                                          dx=FLAGS.box_size / 2048,
                                          dz=FLAGS.box_size,
                                          coords=c,
                                          z_source=z_source)

    m = tf.reshape(m, [FLAGS.batch_size, FLAGS.field_npix, FLAGS.field_npix])

    return m, lensplanes, r_center, a_center


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
def compute_jacobian(Omega_c, sigma8, pgdparams):
    """ Function that actually computes the Jacobian of a given statistics
    """
    params = tf.stack([Omega_c, sigma8])
    with tf.GradientTape() as tape:
        tape.watch(params)
        m, lensplanes, r_center, a_center = compute_kappa(
            params[0], params[1], pgdparams)

        # Adds realism to convergence map
        kmap = desc_y1_analysis(m)

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

    return m, kmap, lensplanes, r_center, a_center, jac, ell, power_spectrum


def main(_):
    with open(FLAGS.pgd_params, "rb") as f:
        pgd_data = pickle.load(f)
        pgdparams = pgd_data['params']
    for i in range(FLAGS.nmaps):
        t = time.time()
        # Query the jacobian
        m, kmap, lensplanes, r_center, a_center, jac, ell, ps = compute_jacobian(
            tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32),
            tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32), pgdparams)
        # Saving results in requested filename
        t = time.time()
        pickle.dump(
            {
                'a': a_center,
                'lensplanes': lensplanes,
                'r': r_center,
                'map': m.numpy(),
                'kmap': kmap.numpy(),
                'jac_ps': jac.numpy(),
                'ps': ps.numpy(),
                'ell': ell.numpy(),
            },
            open(FLAGS.filename + '_%d' % i, "wb"))
        print("iter", i, "took", time.time() - t)


if __name__ == "__main__":
    app.run(main)
