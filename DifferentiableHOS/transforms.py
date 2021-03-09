# This module contains code for computing a wavelet transform of a field
import tensorflow as tf


def starlet2d(image, nscales=4, name="starlet2d"):
  """ Computes the multiscale 2D starlet transform of an image.

  This function only keeps "VALID" regions, i.e. discarding coefficients
  affected by border effects. As a result, each wavelet scale has a
  different shape.

  Args:
    image: `tf.Tensor` of shape [batch_size, h, w] representing the input
      image.
    nscales: `int`. Number of scales to include in the decomposition.
  Returns:
    wavelet_decomposition: `list` of `tf.Tensor` representing each wavelet
      scale plus the smooth approximation.
  """
  with tf.name_scope(name):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=-1)

    # Create the b-spline filter
    h = tf.constant([1, 4, 6, 4, 1], dtype=tf.float32) / 16.
    # Expand to 2d, and reshape appropriately
    W = tf.reshape(tf.tensordot(h, h, axes=0), [5, 5, 1, 1])

    coeffs = []
    approx = image

    for i in range(nscales):
      c_i = tf.nn.atrous_conv2d(approx, W, 2**i, padding="VALID")
      _, h, w, c = c_i.get_shape()
      coeffs.append(tf.image.resize_with_crop_or_pad(approx, h, w) - c_i)
      approx = c_i
    coeffs.append(approx)
    return coeffs
