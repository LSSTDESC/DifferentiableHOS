#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:54:33 2021

@author: Denise Lanzieri, Benjamin Remy
"""

import tensorflow as tf
import numpy as np
from DifferentiableHOS.statistics.l1norm import _get_wavelet_normalization
from DifferentiableHOS.transforms import starlet2d


def measure_power_spectrum_tf(map_data, field, npix):
    """
    Measures power spectrum or 2d data

    Parameters:
    -----------
    map_data: map [npix, npix]

    field: int or float
        transveres degres of the field

    npix : int
           Number of pixel for x and  y

    Returns
    -------
    ell: tf.TensorArray
    power spectrum: tf.TensorArray
    """
    def radial_profile_tf(data):
        """
        Compute the radial profile of 2d image

        Parameters:
        -----------
        data: 2d image

        Returns
        -------
        radial profile
        """
        center = data.shape[0] / 2
        y, x = np.indices((data.shape))
        r = tf.math.sqrt((x - center)**2 + (y - center)**2)
        r = tf.cast(r, dtype=tf.int32)
        tbin = tf.math.bincount(tf.reshape(r, [-1]), tf.reshape(data, [-1]))
        nr = tf.math.bincount(tf.reshape(r, [-1]))
        radialprofile = tf.cast(tbin, dtype=tf.float32) / tf.cast(
            nr, dtype=tf.float32)
        return radialprofile

    def resolution(field, npix):
        """
        pixel resolution

        Returns
        -------
          float
         pixel resolution

        """
        return field * 60 / npix

    def pixel_size_tf(field, npix):
        """
        pixel size

        Returns
        -------

        pizel size: float
        pixel size

        Notes
        -----

        The pixels size is given by:

        .. math::

            pixel_size =  =pi * resolution / 180. / 60. #rad/pixel

        """
        return field / npix / 180 * np.pi

    map_data = tf.cast(map_data, dtype=tf.complex64)
    data_ft = tf.signal.fftshift(tf.signal.fft2d(map_data)) / map_data.shape[0]
    nyquist = tf.cast(map_data.shape[0] / 2, dtype=tf.int32)
    power_spectrum = radial_profile_tf(
        tf.math.real(data_ft * tf.math.conj(data_ft)))[:nyquist]
    power_spectrum = power_spectrum * pixel_size_tf(field, npix)**2
    k = tf.range(power_spectrum.shape[0], dtype=tf.float32)
    ell = 2. * tf.constant(np.pi, dtype=tf.float32) * k / tf.constant(
        pixel_size_tf(field, npix), dtype=tf.float32) / tf.cast(
            map_data.shape[0], dtype=tf.float32)
    return ell, power_spectrum


def power_spectrum_mulscale(map_data, field, npix, nscales=7, nmin=4, nmax=6):
    """
    Measures power spectrum after applied a pass-band filter to the maps by using the wavelet transform

    Parameters:
    -----------
    map_data: tensor [1, npix, npix]
        2d image
    field: int or float
        transveres degres of the field

    npix : int
           Number of pixel for x and  y
           
    nscales: int
        Number of wavelet scales to include
        in the decomposition.
        
    nmin: minimum included scale
    
    nmax: maximu included cale
    Returns
    -------
    ell: tf.TensorArray
    power spectrum: tf.TensorArray
    """
    image = tf.cast(map_data, dtype=tf.float32)
    wt = starlet2d(image, nscales, padding='SAME')
    results = np.zeros(image[0].shape)
    for i in range(nmin, nmax):
        coeffs = wt[i]
        image = tf.reshape(coeffs, [coeffs.shape[1], coeffs.shape[2]])
        results = results + image
    ell, ps = measure_power_spectrum_tf(results, field, npix)
    return ell, ps
