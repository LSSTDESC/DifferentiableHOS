# This module implements the l1 norm statistic described in
# Ajani et al. 2021 https://arxiv.org/abs/2101.01542
import tensorflow as tf
import numpy as np
from DifferentiableHOS.transforms import starlet2d

__all__ = ['l1norm']


def _get_wavelet_normalization(image, nscales):
  """ Computes normalizing constant for starlet, for given image.
  """
  _, nx, ny = image.get_shape()
  knorm = tf.ones((1, 1, 1, 1), dtype=tf.float32)
  knorm = tf.image.resize_with_crop_or_pad(knorm, nx, ny)
  wt = starlet2d(knorm[..., 0], nscales=nscales)
  return [tf.math.sqrt(tf.reduce_sum(c**2)) for c in wt]


def l1norm(image, nscales=3, value_range=[-1., 1.], nbins=16, name='l1norm'):
  """ Computes the starlet l1-norm statistic on the input image.

  Args:
    image: `tf.Tensor` reprensenting the input image to process.
    nscales: `int` representing the number of wavelet scales to include
      in the decomposition.
    value_range: Shape [2] Tensor of same dtype as image, determines the
      range of values in the Histogram.
    nbins: `int`. Number of bins.
  Returns:
    l1norm: `tf.Tensor` containing the l1 norm statistic.
  """
  with tf.name_scope(name):
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Compute the wavelet normalization factor
    norm_factors = _get_wavelet_normalization(image, nscales)

    # Compute wavelet transform
    wt = starlet2d(image, nscales)
    results = []
    # Loop over all wavelet scales
    for coeffs, factor in zip(wt, norm_factors):
      # Normalizing coefficients to preserve standard deviations
      # across scales
      coeffs = coeffs / factor

      # Histogram the coefficient values
      bins = tf.histogram_fixed_width_bins(coeffs,
                                           value_range,
                                           nbins=nbins,
                                           dtype=tf.int32)
      # Compute l1 norm in each bin
      l1norm = [tf.reduce_sum(tf.math.abs(tf.where(bins == i, coeffs, 0)))
                for i in range(nbins)]
      results.append(tf.stack(l1norm))
    return results
