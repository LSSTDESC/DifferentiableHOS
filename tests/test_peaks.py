import tensorflow as tf
import numpy as np
import lenspack
import DifferentiableHOS
from numpy.testing import assert_allclose


def test_peak():
    """ Testing tensorflow peak counting implementation vs. lenspack implementation """
    #start with random map
    test_map = np.random.rand(100, 100)

    #calculating peak locations

    DHOS_output = DifferentiableHOS.statistics.find_peaks2d_tf(
        tf.constant(test_map, dtype=tf.float32))

    lenspack_output = lenspack.peaks.find_peaks2d(test_map)

    #checking peak locations
    assert_allclose(DHOS_output[0].numpy(), lenspack_output[0], atol=1e-6)
    assert_allclose(DHOS_output[1].numpy(), lenspack_output[1], atol=1e-6)
    assert_allclose(DHOS_output[2].numpy(), lenspack_output[2], atol=1e-6)

    #generating histogram

    DHOS_output = DifferentiableHOS.statistics.peaks_histogram_tf(
        tf.constant(test_map, dtype=tf.float32))

    lenspack_output = lenspack.peaks.peaks_histogram(test_map)

    #checking histogram locations
    assert_allclose(DHOS_output[0].numpy(), lenspack_output[0], atol=1e-6)
    assert_allclose(DHOS_output[1].numpy(), lenspack_output[1], atol=1e-6)
    print("peak counts test complete")
