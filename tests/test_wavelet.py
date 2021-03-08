import tensorflow as tf
import numpy as np
import lenspack
import DifferentiableHOS
from numpy.testing import assert_allclose


def test_starlet2d():
  """ Tests the starlet decomposition against lenspack
  """
  # Create test image
  im = np.random.rand(64, 64).astype('float32')

  # Compute wavelet decomposition using TF
  tf_coeffs = DifferentiableHOS.transforms.starlet2d(im.reshape(1, 64, 64),
                                                     nscales=3)

  # Compute wavelet decomposition using Lenspack
  lp_coeffs = lenspack.image.transforms.starlet2d(im, nscales=3)

  # And now comparing results
  for tf_coeff, lp_coeff in zip(tf_coeffs, lp_coeffs):
    _, h, w, c = tf.shape(tf_coeff)
    # Discarding borders from the lenspack image
    lp_coeff = tf.image.resize_with_crop_or_pad(lp_coeff.reshape((1, 64, 64, 1)),
                                                h, w)
    assert_allclose(tf_coeff, lp_coeff, atol=1e-6)
