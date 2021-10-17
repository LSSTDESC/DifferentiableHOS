from flowpm.tfbackground import Omega_m_a, D1
import flowpm.constants as constants
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import flowpm
from DifferentiableHOS.fourier_smoothing import fourier_smoothing


def Epsilon1(cosmology, z_source, tidal_field, Aia, C1):
    r""" Compute the real component of the intrinsic ellipticities

  Parameters
  ----------
    cosmology: `Cosmology`,
        cosmology object.
        
    z_source: 1-D `Tensor`
        Source redshifts with shape [Nz].
     
    tidal_field:  Tensor of shape ([3, 2048, 2048]),
        Interpolated projected tidal shear of the source plane 
        
    Aia: Float.
        Amplitude parameter AI, describes the  strength of the tidal coupling. 
    
    C1:  Float. 
        Constant parameter (Calibrated in Brown et al. (2002))
    
  Returns
  -------
    e1: 2-D Tensor.
        Real component of the intrinsic ellipticities.
  """
    e1 = -Aia * C1 * Omega_m_a(cosmology, z_source) * constants.rhocrit * (
        tidal_field[0] - tidal_field[1]) / D1(cosmology, z_source)
    return e1


def Epsilon2(cosmology, z_source, tidal_field, Aia, C1):
    r""" Compute the imaginary component of the intrinsic ellipticities

  Parameters
  ----------
    cosmology: `Cosmology`,
        cosmology object.
        
    z_source: 1-D `Tensor`
        Source redshifts with shape [Nz].
     
    tidal_field:  Tensor of shape ([3, 2048, 2048]),
        Interpolated projected tidal shear of the source plane 
        
    Aia: Float.
        Amplitude parameter AI, describes the  strength of the tidal coupling. 
    
    C1:  Float. 
        Constant parameter (Calibrated in Brown et al. (2002))
    
  Returns
  -------
    e2: 2-D Tensor.
        Imaginary component of the intrinsic ellipticities.
  """
    e2 = -2 * Aia * C1 * Omega_m_a(cosmology, z_source) * constants.rhocrit * (
        tidal_field[2]) / D1(cosmology, z_source)
    return e2


def tidal_field(plane_source, resolution, sigma):
    r""" Compute the projected tidal shear

  Parameters
  ----------
  plane_source: Tensor of Shape([batch_size, plane_resolution, plane_resolution])
    Projected density of the source plane

  resolution : Int
      Pixel resolution of the source plane 
      
  sigma: Float
      Sigma of the two-dimensional smoothing kernel

  Returns
  -------
  tidal_planes : Tensor of shape [3, plane_resolution,plane_resolution] (sx, sy, sz)
      Projected tidal shear of the source plane

  Notes
  -----
  The parametrization :cite:`J. Harnois-De ́raps et al.` for the projected tidal shear
   is given by:
   
  .. math::

   \tilde{s}_{ij,2D}(\textbf{k}_{\bot})=
   2 \pi \left [ \frac{k_ik_j}{k^2}- \frac{1}{3} \right ]
   \tilde{\delta}_{2D}(\textbf{k}_{\bot})
   \mathcal{G}_{2D}(\sigma_g)

  """

    k = np.fft.fftfreq(resolution)
    kx, ky = np.meshgrid(k, k)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.
    sxx = 2 * np.pi * (kx * kx / k2 - 1. / 3)
    sxx[0, 0] = 0.
    syy = 2 * np.pi * (ky * ky / k2 - 1. / 3)
    syy[0, 0] = 0.
    sxy = 2 * np.pi * (kx * ky / k2 - 1. / 3)
    sxy[0, 0] = 0.
    ss = tf.stack([sxx, syy, sxy], axis=0)
    ss = tf.cast(ss, dtype=tf.complex64)
    ftt_plane_source = flowpm.utils.r2c2d(plane_source)
    ss_fac = ss * ftt_plane_source
    ss_smooth = fourier_smoothing(ss_fac, sigma, resolution, source_plane=True)
    tidal_planes = flowpm.utils.c2r2d((ss_smooth))
    return tidal_planes


def interpolation(tidal_planes, dx, r_center, tidial_field_npix, coords):
    r""" Compute the interpolation of the projected tidal shear of the source plane on the light-cones
   Parameters
   ----------
    tidal_planes: Tensor of shape [3, 2048, 2048] (sx, sy, sz).
        Projected tidal shear of the source plane
    
    dx: float 
        transverse pixel resolution of the tidal planes [Mpc/h]

    r_center: tf.Tensor 
        Center of the lensplanes used to build the lightcone[Mpc/h]
    
    tidial_field_npix: Int
        Resolution of the final interpolated projected tidal shear map
    
    coords: 3-D array.
        Angular coordinates in radians of N points with shape [batch, N, 2].

    Returns
    -------
    im= Tensor of shape [3, 2048, 2048].
        Interpolated projected tidal shear of the source plane on the light-cones
    """
    coords = tf.convert_to_tensor(coords, dtype=tf.float32)
    for r in (r_center):
        c = coords * r / dx

        # Applying periodic conditions on lensplane
        shape = tf.shape(tidal_planes)
        c = tf.math.mod(c, tf.cast(shape[1], tf.float32))

        # Shifting pixel center convention
        c = tf.expand_dims(c, axis=0) - 0.5

        im = tfa.image.interpolate_bilinear(tf.expand_dims(tidal_planes, -1),
                                            c,
                                            indexing='xy')
        imx, imy, imxy = im
        imx = tf.reshape(imx, [tidial_field_npix, tidial_field_npix])
        imy = tf.reshape(imy, [tidial_field_npix, tidial_field_npix])
        imxy = tf.reshape(imxy, [tidial_field_npix, tidial_field_npix])
        im = tf.stack([imx, imy, imxy], axis=0)
    return im
