"""
Created on Fri Feb 12 12:54:33 2021

@author: Ben Horowitz

From lenspack implementation
"""
import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def find_peaks2d_tf(image,mask = None, ordered=True,threshold=None):
    if mask is not None:
      #  mask = np.atleast_2d(mask)
        if mask.shape != image.shape:
            print("Warning: mask not compatible with image -> ignoring.")
            mask = tf.ones(image.shape)
        else:
            # Make sure mask is binary, i.e. turn nonzero values into ones
            mask = tf.cast(tf.cast(mask,bool),float)
    else:
        mask = tf.ones(image.shape)
        
    if threshold is None:
        threshold = tf.math.reduce_min(image)
    else:
        threshold = tf.math.reduce_max((threshold,tf.math.reduce_min(image)))


    offset = tf.math.reduce_min(image)
    threshold = threshold - offset
    image = image - offset

    map0 = image[1:-1, 1:-1]

    # Extract shifted maps
    map1 = image[0:-2, 0:-2]
    map2 = image[1:-1, 0:-2]
    map3 = image[2:, 0:-2]
    map4 = image[0:-2, 1:-1]
    map5 = image[2:, 1:-1]
    map6 = image[0:-2, 2:]
    map7 = image[1:-1, 2:]
    map8 = image[2:, 2:]

    merge = ((map0 > map1) & (map0 > map2) & (map0 > map3) & (map0 > map4) &
             (map0 > map5) & (map0 > map6) & (map0 > map7) & (map0 > map8))
    
    bordered = tf.pad(merge,tf.constant(((1,1),(1,1))),constant_values=0.0)
    peaksmap = tf.cast(bordered,float)*image*mask
    XY = tf.where(peaksmap > threshold)
    heights = tf.gather_nd(image,XY) + offset

    if ordered:
        ind = tf.argsort(heights)[::-1]
        return tf.gather(XY[:,0],ind),tf.gather(XY[:,1],ind),tf.gather(heights,ind)
    return XY[:,0], XY[:,1],heights

@tf.function
def peaks_histogram_tf(image, bins=None, mask=None):
    """Compute a histogram of peaks in a 2d image.
    Parameters
    ----------
    image : tensor (2D)
        Two-dimensional input tensor
    bins : int or tensor (1D), optional
        Specification of bin edges or the number of bins to use for the
        histogram. If not provided, a default of 10 bins linearly spaced
        between the image minimum and maximum (inclusive) is used.
    mask : tensor (same shape as `image`), optional
        Tensor identifying which pixels of `image` to consider/exclude
        in finding peaks. Can either either be numeric (0 or 1) or boolean 
        (false or true)
    Returns
    -------
    counts, bins : tuple of 1D tensors
        Histogram and bin boundary values. If the returned `counts` has 
        N values, `bin_edges` will have N + 1 values.
     """
    if bins is None:
        bins = tf.linspace(tf.math.reduce_min(image), tf.math.reduce_max(image), 10)
    elif isinstance(bins, int):
        bins = tf.linspace(tf.math.reduce_min(image), tf.math.reduce_max(image), bins)
    else:
        bins = bins
    x, y, heights = find_peaks2d_tf(image, threshold=None, mask=mask)
    counts = tfp.stats.histogram(heights,bins)
    return counts,bins
