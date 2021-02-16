#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:38:27 2021

@author: Denise Lanzieri
"""
import numpy as np
import tensorflow as tf
import flowpm
from flowpm.tfbackground import cosmo, rad_comoving_distance
from flowpm.tfpm import kick, drift, force
import flowpm.constants as constants

# @tf.function
# def power_spectrum_for_cosmology(omega_m, sigma_8):
#     .... defines a cosmology with omega_m and sigma_8 in it, use it to run the nbody lightcone, compute the born approximation, and power spectrum 
#     return power_spectrum



@tf.function
def power_spectrum_for_cosmology(state, stages, nc,
              plane_resolution, # in arcmin
              plane_size, # in pixels
              field,
              Boxsize,
              ds,
              Omega0_m,
              sigma8,                 
              cosmology=cosmo, pm_nc_factor=1, 
              name="NBody"):
    
    cosmo['Omega0_m']=tf.cast(Omega0_m,dtype=tf.float32)
    cosmo['sigma8']=tf.cast(sigma8,dtype=tf.float32)
    #prefactor from Poisson equation
    cons=tf.cast(3/2*Omega0_m*(cosmo['H0']/constants.c)**2,dtype=tf.float32)
    #mean 3D particle density
    nbar=np.prod(nc)/np.prod(Boxsize)
    #2D mesh area in rad^2 per pixel
    A=((field*np.pi/180/plane_size)**2) 
    #resolution=field*60/plane_size
    #pixel_size =  =pi * resolution / 180. / 60. #rad/pixel
    pixel_size_tf=field/plane_size / 180 *np.pi 
    with tf.name_scope(name):
        state = tf.convert_to_tensor(state, name="state")

        shape = state.get_shape()
        batch_size = shape[1]

        # Unrolling leapfrog integration to make tf Autograph happy
        if len(stages) == 0:
          return state

        ai = stages[0]

        # first force calculation for jump starting
        state = force(state, nc, pm_nc_factor=pm_nc_factor, cosmology=cosmology)

        # Compute the width of the lens planes based on number of time steps
        w = nc[2]//(len(stages)-1)
        nx = nc[0]
        nz = nc[2]
        lps = []
        lps_a = []

        x, p, f = ai, ai, ai
        # Loop through the stages
        for i in range(len(stages) - 1):
            a0 = stages[i]
            a1 = stages[i + 1]
            ah = (a0 * a1) ** 0.5

            # Kick step
            state = kick(state, p, f, ah, cosmology=cosmology)
            p = ah

            # Drift step
            state = drift(state, x, p, a1, cosmology=cosmology)
            x = a1

            # Access the positions of the particles
            pos = state[0]
            d = pos[:,:,2]      

            # This is the transverse comoving distance inside the box
            xy = pos[:,:,:2] - nx/2

            # Compute density plane in sky coordinates around the center of the lightcone
            # TODO: Confirm conversion from comoving distances to angular size! I thought
            # we should be using the angular diameter distance, but as far as I can see
            # everyone uses the transverse comoving distance, and I don't understand exactly why
            lens_plane = tf.zeros([batch_size, plane_size, plane_size])

            # Convert coordinates to angular coords, and then into plane coords
            xy = (xy / tf.expand_dims(d,-1))/np.pi*180*60/plane_resolution
            xy = xy + plane_size/2

            # Selecting only the particles contributing to the lens plane
            mask = tf.where((d>(nz - (i+1)*w)) & (d <= (nz - i*w)),1.,0.)
            # And falling inside the plane, NOTE: This is only necessary on CPU, on GPU
            # cic paint 2d can be made to work with non periodic conditions.
            mask = mask * tf.where((xy[...,0]>0) & (xy[...,0]<plane_size),1.,0.)
            mask = mask * tf.where((xy[...,1]>0) & (xy[...,1]<plane_size),1.,0.)
            # Compute lens planes by projecting particles
            lens_plane = flowpm.utils.cic_paint_2d(lens_plane, xy + plane_size/2 ,mask)
            lps.append(lens_plane)
            lps_a.append(ah)

            # Here we could trim the state vector for particles originally beyond the current lens plane
            # This way the simulation becomes smaller as it runs and we save resources
            state = tf.reshape(state, [3,batch_size, nc[0], nc[1],-1, 3])
            state = state[:,:,:,:,:(nz - i*w - w // 2),:] # We keep w/2 to be safe, so we allow particle to travel
                                                         # A max distance of width/2
            # redefine shape of state
            nc = state.get_shape()[2:5]
            state = tf.reshape(state, [3,batch_size,-1,3])
            # So this seems to work, but we should be a tiny bit careful because we break periodicity in the z
            # direction at z=0.... probably not a big deal but still gotta check what that does.


            # Force
            state = force(state, nc, pm_nc_factor=pm_nc_factor, cosmology=cosmology)
            f = a1

            # Kick again
            state = kick(state, p, f, a1, cosmology=cosmology)
            p = a1
    d=rad_comoving_distance(cosmology,lps_a)
    columndens =(A*nbar)*(d**2)#particles/Volume*angular pixel area* distance^2 -> 1/L units
    w  = ((ds-d)*(d/ds))/(columndens)
    w=w/lps_a  
    #w=tf.cast(w,dtype=tf.float64)
    k_map=0
    for i in range(len(lps_a)):
        k_map += cons*lps[i][0]*  w[i]
    proto_tensor = tf.make_tensor_proto(k_map) 
    k_map=tf.make_ndarray(proto_tensor)
    data_ft = tf.signal.fftshift(tf.signal.fft2d(k_map)) / k_map.shape[0]
    nyquist = tf.cast(k_map.shape[0]/2,dtype=tf.int32)
    data=tf.math.real(data_ft*tf.math.conj(data_ft))
    center = data.shape[0]/2
    y, x = np.indices((data.shape))
    r = tf.math.sqrt((x - center)**2 + (y - center)**2)
    r=tf.cast(r,dtype=tf.int32)
    tbin=tf.math.bincount(tf.reshape(r,[-1]), tf.reshape(data,[-1]))
    nr = tf.math.bincount(tf.reshape(r,[-1]))
    radialprofile=tf.cast(tbin,dtype=tf.float64)/tf.cast(nr,dtype=tf.float64)
    power_spectrum = radialprofile[:nyquist]
    power_spectrum = power_spectrum*pixel_size_tf**2
    k = tf.range(power_spectrum.shape[0],dtype=tf.float64)
    ell = 2. * tf.constant(np.pi,dtype=tf.float64) * k / tf.constant(pixel_size_tf,dtype=tf.float64) / tf.cast(k_map.shape[0],dtype=tf.float64)
    return ell, power_spectrum