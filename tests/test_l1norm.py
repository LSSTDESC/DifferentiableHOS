import pickle
import numpy as np
import numdifftools as nd
import tensorflow as tf
import DifferentiableHOS as DHOS
import numdifftools as nd
from numpy.testing import assert_allclose


@tf.function
def compute_gradient(kmap):
  """ Function that actually computes the Jacobian of a given statistics
    """

  with tf.GradientTape() as tape:
    tape.watch(kmap)
    l1 = DHOS.statistics.l1norm(kmap[...,0],nscales=3, nbins=7,value_range=[-0.03, 0.03])[2][0]
    l1=tf.reduce_sum(l1)
  jac = tape.gradient(l1, kmap)
  return jac


@tf.function
def func(kmap):
    kmap=tf.cast(kmap,dtype=tf.float32)
    kmap = tf.expand_dims(kmap,0)
    kmap=tf.reshape(kmap,[32,32])
    kmap = tf.expand_dims(kmap,0)
    l1 = DHOS.statistics.l1norm(kmap, nbins=7,value_range=[-0.03, 0.03],nscales=3)[2][0]
    l1=tf.reduce_sum(l1)
    return l1


def test_l1norm():
  """ Tests the l1norm gradient against numdifftools
  """
  # Import test image
  results = pickle.load( open( "kmap_low32", "rb" ) )
  kmap = results['kmap']
  # Compute gradient using TF
  dl1_tf=compute_gradient(kmap)
  dl1_tf=tf.reshape(dl1_tf,[1024])
  # Compute gradient using numdifftools
  dl1=nd.Gradient(func)
  dl1_np=dl1(kmap[0,...,0])
  # And now comparing results
  assert_allclose(dl1_tf, dl1_np, atol=1e-3)

