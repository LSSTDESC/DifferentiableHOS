import os
import tensorflow as tf
import numpy as np
import flowpm
import pickle
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
from absl import app
from absl import flags
from flowpm.tfpower import linear_matter_power
import time
import tensorflow_probability as tfp
from flowpm.neural_ode_nbody import make_neural_ode_fn


flags.DEFINE_string("filename", "output_", "Output filename")
flags.DEFINE_string(
    "correction_params",
    "/local/home/dl264294/flowpm/notebooks/camels_25_64_pkloss.params",
    "Correction parameter files")
flags.DEFINE_float("Omega_c", 0.2589, "Fiducial CDM fraction")
flags.DEFINE_float("Omega_b", 0.04860, "Fiducial baryonic matter fraction")
flags.DEFINE_float("sigma8", 0.8159, "Fiducial sigma_8 value")
flags.DEFINE_float("n_s", 0.9667, "Fiducial n_s value")
flags.DEFINE_float("h", 0.6774, "Fiducial Hubble constant value")
flags.DEFINE_float("w0", -1.0, "Fiducial w0 value")
flags.DEFINE_integer("nc", 128,
                     "Number of transverse voxels in the simulation volume")
flags.DEFINE_float("box_size", 205.,
                   "Transverse comoving size of the simulation volume")
flags.DEFINE_integer("batch_size", 1,
                     "Number of simulations to run in parallel")
flags.DEFINE_integer("nmaps", 15, "Number maps to generate.")
FLAGS = flags.FLAGS


@tf.function
def compute_kappa(Omega_c, sigma8, Omega_b, n_s, h, w0):
    """ Computes a convergence map using ray-tracing through an N-body for a given
    set of cosmological parameters
    """

    cosmology = flowpm.cosmology.Planck15(Omega_c=Omega_c,
                                          sigma8=sigma8,
                                          Omega_b=Omega_b,
                                          n_s=n_s,
                                          h=h,
                                          w0=w0)
    r = tf.linspace(0., FLAGS.box_size * 11, 12)
    r_center = 0.5 * (r[1:] + r[:-1])
    a = flowpm.tfbackground.a_of_chi(cosmology, r)
    a_center = flowpm.tfbackground.a_of_chi(cosmology, r_center)
    stages = a_center[::-1]
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
    initial_state = flowpm.lpt_init(cosmology, initial_conditions,
                                    0.14285714254594556)
    initial_state = initial_state[0:2]
    res = tfp.math.ode.DormandPrince(rtol=1e-5,
                                     atol=1e-5).solve(make_neural_ode_fn(FLAGS.nc, FLAGS.batch_size),
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


    return res

def main(_):
    t = time.time()
    for i in range(FLAGS.nmaps):
        res = compute_kappa(
            tf.convert_to_tensor(FLAGS.Omega_c, dtype=tf.float32),
            tf.convert_to_tensor(FLAGS.sigma8, dtype=tf.float32),
            tf.convert_to_tensor(FLAGS.Omega_b, dtype=tf.float32),
            tf.convert_to_tensor(FLAGS.n_s, dtype=tf.float32),
            tf.convert_to_tensor(FLAGS.h, dtype=tf.float32),
            tf.convert_to_tensor(FLAGS.w0, dtype=tf.float32))
        pickle.dump({
            'res': res,
        }, open(FLAGS.filename + 'res_%d' % i + '.pkl', "wb"))
        print("iter", i, "took", time.time() - t)


if __name__ == "__main__":
    app.run(main)
