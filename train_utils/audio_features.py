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


def compute_attack_time(cqt_magnitude, hop_length=128, sample_rate=16000, lower_threshold=0.1, upper_threshold=0.9):
    """
    Computes the attack time from the CQT magnitude spectrogram.
    Attack time is defined as the time it takes for the envelope to go from 
    10% to 90% of its maximum value.
    
    Args:
        cqt_magnitude: A 2D tensor of shape (time_frames, n_bins) representing the CQT magnitude spectrogram.
        hop_length: CQT hop length in samples (default is 128).
        sample_rate: Audio sample rate (default is 16000).
    
    Returns:
        A scalar tensor representing the attack time normalized to [0, 1].
        Shorter attacks (percussive) → values closer to 0
        Longer attacks (slow/soft) → values closer to 1
    """
    
    # Compute the temporal envelope by summing magnitude across all frequency bins
    envelope = tf.reduce_sum(tf.abs(cqt_magnitude) + 1e-8, axis=1)  # Shape: (time_frames,)
    
    # Find the maximum value in the envelope
    max_envelope = tf.reduce_max(envelope)
    
    # Define threshold levels (10% and 90% of maximum)
    threshold_10 = lower_threshold * max_envelope
    threshold_90 = upper_threshold * max_envelope
    
    # Find first frame where envelope exceeds 10% threshold
    mask_10 = envelope >= threshold_10
    indices_10 = tf.where(mask_10)
    
    # Find first frame where envelope exceeds 90% threshold
    mask_90 = envelope >= threshold_90
    indices_90 = tf.where(mask_90)
    
    # Handle edge cases where thresholds are not reached
    if tf.size(indices_10) == 0 or tf.size(indices_90) == 0:
        # Return a default middle value if attack cannot be computed
        return tf.constant(0.5, dtype=tf.float32)
    
    # Get the frame indices
    frame_10 = tf.cast(indices_10[0, 0], tf.float32)
    frame_90 = tf.cast(indices_90[0, 0], tf.float32)
    
    # Compute attack time in frames (difference between 90% and 10% points)
    attack_frames = tf.maximum(frame_90 - frame_10, 0.0)
    
    # Convert frames to seconds
    frame_duration = hop_length / sample_rate
    attack_time_seconds = attack_frames * frame_duration
    
    # Normalize to [0, 1] using logarithmic scale
    # Typical attack times:
    # - Very fast/percussive: ~0.001 - 0.01s (drums, plucked strings)
    # - Medium: ~0.01 - 0.1s (most instruments)
    # - Slow: ~0.1 - 1.0s (bowed strings, soft synth pads)
    min_attack_time = 0.001  # 1ms
    max_attack_time = 1.0    # 1 second
    
    # Clip to valid range
    attack_time_seconds = tf.clip_by_value(attack_time_seconds, min_attack_time, max_attack_time)
    
    # Apply logarithmic normalization for better distribution
    log_attack = tf.math.log(attack_time_seconds / min_attack_time + 1e-8)
    log_range = tf.math.log(max_attack_time / min_attack_time + 1e-8)
    attack_normalized = log_attack / log_range
    
    # Final clamp to [0, 1]
    attack_normalized = tf.clip_by_value(attack_normalized, 0.0, 1.0)
    
    return attack_normalized