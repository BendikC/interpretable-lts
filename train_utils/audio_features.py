import tensorflow as tf

def compute_spectral_centroid(cqt_magnitude, fmin=32.7, bins_per_octave=48):
    """
    Computes the spectral centroid from CQT magnitude.
    Now handles both 2D and 3D inputs.
    
    Args:
        cqt_magnitude: A tensor of CQT magnitudes:
            - 2D: (batch_size, n_bins) - batch of single frames
            - 3D: (batch_size, time_frames, n_bins) - batch of temporal sequences
        fmin: Minimum frequency (Hz) of first CQT bin (default: 32.7 Hz, C1)
        bins_per_octave: Number of CQT bins per octave (default: 48)
    
    Returns:
        1D tensor (batch_size,) - normalized centroid per example in [0, 1]
        For 3D input, returns the mean centroid across all time frames.
    """
    input_shape = tf.shape(cqt_magnitude)
    n_bins = input_shape[2]
    bin_indices = tf.cast(tf.range(n_bins), tf.float32)
    frequencies = fmin * tf.pow(2.0, bin_indices / bins_per_octave)
    
    # Reshape for broadcasting: (1, 1, n_bins)
    frequencies = tf.reshape(frequencies, [1, 1, -1])
    
    # Compute centroid for each time frame
    # weighted_freq shape: (batch_size, time_frames)
    weighted_freq = tf.reduce_sum(cqt_magnitude * frequencies, axis=2)
    # total_magnitude shape: (batch_size, time_frames)
    total_magnitude = tf.reduce_sum(cqt_magnitude, axis=2)
    
    # Centroid per time frame: (batch_size, time_frames)
    centroid_hz = weighted_freq / (total_magnitude + 1e-8)
    
    # Average centroid across time: (batch_size,)
    centroid_hz = tf.reduce_mean(centroid_hz, axis=1)
    
    # Normalize to [0, 1] (same for both 2D and 3D)
    fmax = fmin * tf.pow(2.0, tf.cast(n_bins, tf.float32) / bins_per_octave)
    centroid_hz = tf.clip_by_value(centroid_hz, fmin, fmax)
    
    log_centroid = tf.math.log(centroid_hz / fmin + 1e-8)
    log_range = tf.math.log(fmax / fmin + 1e-8)
    centroid_normalized = log_centroid / log_range
    
    return tf.clip_by_value(centroid_normalized, 0.0, 1.0)