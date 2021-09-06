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

flags.DEFINE_string("filename", "results_maps.pkl", "Output filename")
flags.DEFINE_float("Omega_c", 0.2589, "Fiducial CDM fraction")
flags.DEFINE_float("sigma8", 0.8159, "Fiducial sigma_8 value")
flags.DEFINE_integer("nc", 64,
                     "Number of transverse voxels in the simulation volume")
flags.DEFINE_integer("field_npix", 1024,
                     "Number of pixels in the lensing field")
flags.DEFINE_float("box_size", 64.,
                   "Transverse comoving size of the simulation volume")
flags.DEFINE_float("field_size", 5., "TSize of the lensing field in degrees")
flags.DEFINE_float("a_init", 0.1, "Initial scale factor")
flags.DEFINE_integer("init_nsteps", 4, "Number of steps before the lightcone")
flags.DEFINE_integer("n_lens", 36, "Number of lensplanes in the lightcone")
flags.DEFINE_integer("batch_size", 1,
                     "Number of simulations to run in parallel")
flags.DEFINE_integer("nmaps", 20, "Number maps to generate.")
flags.DEFINE_integer("B", 1, "Scale resolution factor")

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

    # We run several steps from initial scale factor to start of raytracing
    init_stages = tf.linspace(FLAGS.a_init, a[-1], FLAGS.init_nsteps)
    # Then one step per lens plane
    stages = tf.concat([init_stages, a_center[::-1]], axis=0)

    # Create some initial conditions
    k = tf.constant(np.logspace(-4, 1, 512), dtype=tf.float32)
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
                          pm_nc_factor=FLAGS.B,
                          return_intermediate_states=True)
    PGD_param = pickle.load(
        open(
            '/global/u1/d/dlan/DifferentiableHOS/scripts/PGD_fit/results_fit_PGD.pkl',
            'rb'))
    new_states = states[FLAGS.init_nsteps - 1:][::-1]
    dx = []
    corrected_states = []
    for i in range(len(new_states)):
        dx.append(
            tfpm.PGD_correction(new_states[i][1],
                                PGD_param['params'][i],
                                [FLAGS.nc, FLAGS.nc, FLAGS.nc],
                                pm_nc_factor=FLAGS.B))
        corrected_states.append(dx[i] + new_states[i][1][0])
    # Extract the lensplanes
    lensplanes = []
    for i in range(len(corrected_states)):
        plane = flowpm.raytracing.density_plane(
            corrected_states[i], [FLAGS.nc, FLAGS.nc, FLAGS.nc],
            FLAGS.nc // 2,
            width=FLAGS.nc,
            plane_resolution=256,
            shift=flowpm.raytracing.random_2d_shift())
        plane = tf.expand_dims(plane, axis=-1)
        plane = tf.image.random_flip_left_right(plane)
        plane = tf.image.random_flip_up_down(plane)
        lensplanes.append((r_center[i], new_states[i][0], plane[..., 0]))
    xgrid, ygrid = np.meshgrid(
        np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                    endpoint=False),  # range of X coordinates
        np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                    endpoint=False))  # range of Y coordinates

    coords = np.stack([xgrid, ygrid], axis=0) * u.deg
    c = coords.reshape([2, -1]).T.to(u.rad)
    # Create array of source redshifts
    z_source = tf.constant([1.])
    m = flowpm.raytracing.convergenceBorn(cosmology,
                                          lensplanes,
                                          dx=FLAGS.box_size / 256,
                                          dz=FLAGS.box_size,
                                          coords=c,
                                          z_source=z_source)

    m = tf.reshape(m, [FLAGS.batch_size, FLAGS.field_npix, FLAGS.field_npix])

    return  m, lensplanes, r_center, a_center


def main(_):
    for i in range(FLAGS.nmaps):
        t = time.time()
        # Query the jacobian
        m,lensplanes, r_center, a_center = compute_kappa(
            tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32),
            tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32))
        # Saving results in requested filename
        pickle.dump(
            {
                'a': a_center,
                'lensplanes': lensplanes,
                'r': r_center,
                'map': m.numpy(),
            }, open(FLAGS.filename + '_%d' % i, "wb"))
        print("iter", i, "took", time.time() - t)


if __name__ == "__main__":
    app.run(main)
