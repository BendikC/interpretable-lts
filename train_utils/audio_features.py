import tensorflow as tf

def compute_spectral_centroid(cqt_magnitude, fmin=32.7, bins_per_octave=48):
    """
    Computes the spectral centroid from CQT magnitude.
    
    Args:
        cqt_magnitude: A 2D tensor (batch_size, n_bins) - batch of single CQT frames
        fmin: Minimum frequency (Hz) of first CQT bin (default: 32.7 Hz, C1)
        bins_per_octave: Number of CQT bins per octave (default: 48)
    
    Returns:
        1D tensor (batch_size,) - normalized centroid per example in [0, 1]
    """

    # first we ensure non-negative and add small epsilon
    cqt_magnitude = tf.abs(cqt_magnitude) + 1e-8

    # we get the number of bins, and we define the indices as floats
    n_bins = tf.shape(cqt_magnitude)[1]
    bin_indices = tf.cast(tf.range(n_bins), tf.float32)
    
    # cqt bins are logarithmically spaced
    # this means that frequency for each bin is:
    # f = fmin * 2^(bin_index / bins_per_octave)
    frequencies = fmin * tf.pow(2.0, bin_indices / bins_per_octave)

    # we expand the dims to match the batch size
    frequencies = tf.expand_dims(frequencies, 0)
    
    # we compute the sum of frequency * magnitude
    weighted_freq = tf.reduce_sum(cqt_magnitude * frequencies, axis=1)

    # then we compute the total magnitude
    total_magnitude = tf.reduce_sum(cqt_magnitude, axis=1)

    # spectral centroid = weighted frequency sum / total magnitude
    centroid_hz = weighted_freq / (total_magnitude + 1e-8)
    
    # Clamp centroid to valid frequency range before log
    fmax = fmin * tf.pow(2.0, tf.cast(n_bins, tf.float32) / bins_per_octave)
    centroid_hz = tf.clip_by_value(centroid_hz, fmin, fmax)
    
    # Normalize to [0, 1] using logarithmic scale
    # log(centroid/fmin) / log(fmax/fmin)
    log_centroid = tf.math.log(centroid_hz / fmin + 1e-8)
    log_range = tf.math.log(fmax / fmin + 1e-8)
    centroid_normalized = log_centroid / log_range
    
    # Final clamp to [0, 1] (safety net)
    centroid_normalized = tf.clip_by_value(centroid_normalized, 0.0, 1.0)
    
    return centroid_normalized