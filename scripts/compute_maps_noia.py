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
from scipy.stats import norm
from flowpm.fourier_smoothing import fourier_smoothing
from flowpm.tfbackground import rad_comoving_distance

flags.DEFINE_string("filename",
                    "/pscratch/sd/d/dlan/maps_noia/maps4/results_maps.pkl",
                    "Output filename")
flags.DEFINE_string(
    "pgd_params",
    "/global/homes/d/dlan/DifferentiableHOS/scripts/results_fit_PGD_205_128.pkl",
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
flags.DEFINE_integer("nmaps", 50, "Number maps to generate.")

FLAGS = flags.FLAGS


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
    states_nopgd = flowpm.nbody(cosmology,
                                initial_state,
                                stages, [FLAGS.nc, FLAGS.nc, FLAGS.nc],
                                return_intermediate_states=True)
    # Extract the lensplanes
    lensplanes = []
    lensplanes_nopgd = []
    matrix = flowpm.raytracing.rotation_matrices()
    for i, j in zip(range(len(a_center)), cycle(range(6))):
        shift_both = flowpm.raytracing.random_2d_shift()
        plane = flowpm.raytracing.density_plane(
            states[::-1][i][1],
            [FLAGS.nc, FLAGS.nc, FLAGS.nc],
            FLAGS.nc // 2,
            width=FLAGS.nc,
            plane_resolution=2048,
            rotation=matrix[j],
            shift=shift_both,
        )

        plane = fourier_smoothing(plane, sigma=1.024, resolution=2048)
        lensplanes.append((r_center[i], states[::-1][i][0], plane))
        plane_nopgd = flowpm.raytracing.density_plane(
            states_nopgd[::-1][i][1],
            [FLAGS.nc, FLAGS.nc, FLAGS.nc],
            FLAGS.nc // 2,
            width=FLAGS.nc,
            plane_resolution=2048,
            rotation=matrix[j],
            shift=shift_both,
        )

        plane_nopgd = fourier_smoothing(plane_nopgd,
                                        sigma=1.024,
                                        resolution=2048)
        lensplanes_nopgd.append(
            (r_center[i], states_nopgd[::-1][i][0], plane_nopgd))
    xgrid, ygrid = np.meshgrid(
        np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                    endpoint=False),  # range of X coordinates
        np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                    endpoint=False))  # range of Y coordinates

    coords = np.stack([xgrid, ygrid], axis=0)
    c = coords.reshape([2, -1]).T / 180. * np.pi  # convert to rad from deg
    # Create array of source redshifts
    z_source = tf.convert_to_tensor([1.034, 0.858, 1.163], dtype=tf.float32)
    m = flowpm.raytracing.convergenceBorn(cosmology,
                                          lensplanes,
                                          dx=FLAGS.box_size / 2048,
                                          dz=FLAGS.box_size,
                                          coords=c,
                                          z_source=z_source,
                                          field_npix=FLAGS.field_npix)
    m_nopgd = flowpm.raytracing.convergenceBorn(cosmology,
                                                lensplanes_nopgd,
                                                dx=FLAGS.box_size / 2048,
                                                dz=FLAGS.box_size,
                                                coords=c,
                                                z_source=z_source,
                                                field_npix=FLAGS.field_npix)
    return m, m_nopgd, states, states_nopgd, r_center, a_center, initial_conditions


def desc_y1_analysis(kmap):
    """
  Adds noise and apply smoothing we might expect in DESC Y1 SRD setting
  """
    ngal = 10  # gal/arcmin **2
    pix_scale = FLAGS.field_size / FLAGS.field_npix * 60  # arcmin
    ngal_per_pix = ngal * pix_scale**2  # galaxies per pixels
    sigma_e = 0.26 / np.sqrt(2 * ngal_per_pix)  # Rescaled noise sigma
    sigma_pix = 1. / pix_scale  # Smooth at 3.6 arcmin
    # Add noise
    kmap = kmap + sigma_e * tf.random.normal(kmap.shape)
    # Add smoothing
    kmap = fourier_smoothing(kmap,
                             sigma=sigma_pix,
                             resolution=FLAGS.field_npix)
    return kmap


def main(_):

    # Loading PGD parameters
    with open(FLAGS.pgd_params, "rb") as f:
        pgd_data = pickle.load(f)
        pgdparams = pgd_data['params']

    for i in range(FLAGS.nmaps):
        t = time.time()

        m, m_nopgd, states, states_nopgd, r_center, a_center, initial_conditions = compute_kappa(
            tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32),
            tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32), pgdparams)
        kmap = desc_y1_analysis(m)
        kmap_nopgd = desc_y1_analysis(m_nopgd)
        # Saving results in requested filename
        pickle.dump(
            {
                'kmap': kmap.numpy(),
                'kmap_nopgd': kmap_nopgd.numpy(),
                'states': [s[1].numpy() for s in states],
                'states_nopgd': [s[1].numpy() for s in states_nopgd],
                'a': a_center.numpy(),
                'm': m.numpy(),
                'm_nopgd': m_nopgd.numpy(),
                'r': r_center.numpy(),
                'initial_conditions': initial_conditions.numpy()
            }, open(FLAGS.filename + '_%d' % i, "wb"))
        print("iter", i, "took", time.time() - t)


if __name__ == "__main__":
    app.run(main)
