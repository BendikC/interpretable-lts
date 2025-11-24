"""
Analyze trained VAE models for interpretability.
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import configparser
import sys
import json

from tensorflow.keras import layers

from analysis.dataset_utils import load_nsynth_with_labels, stratified_sample
from analysis.interpretability import (
    compute_descriptor_correlation,
    compute_disentanglement_score,
    independence_test
)
from analysis.visualization import (
    plot_tsne,
    plot_descriptor_correlation,
    plot_latent_traversal,
    plot_latent_distribution
)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze trained VAE')
    parser.add_argument('--config', type=str, default='./configs/default.ini',
                       help='Path to training config file (same as used for training)')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples for analysis')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for analysis (default: workspace/analysis)')
    return parser.parse_args()


def compute_spectral_centroid_tf(cqt_magnitude, fmin=32.7, bins_per_octave=48):
    """Compute spectral centroid using proper frequency mapping."""
    cqt_magnitude = tf.abs(cqt_magnitude) + 1e-8
    n_bins = tf.shape(cqt_magnitude)[1]
    bin_indices = tf.cast(tf.range(n_bins), tf.float32)
    frequencies = fmin * tf.pow(2.0, bin_indices / bins_per_octave)
    frequencies = tf.expand_dims(frequencies, 0)
    weighted_freq = tf.reduce_sum(cqt_magnitude * frequencies, axis=1)
    total_magnitude = tf.reduce_sum(cqt_magnitude, axis=1)
    centroid_hz = weighted_freq / (total_magnitude + 1e-8)
    fmax = fmin * tf.pow(2.0, tf.cast(n_bins, tf.float32) / bins_per_octave)
    centroid_hz = tf.clip_by_value(centroid_hz, fmin, fmax)
    log_centroid = tf.math.log(centroid_hz / fmin + 1e-8)
    log_range = tf.math.log(fmax / fmin + 1e-8)
    centroid_normalized = log_centroid / log_range
    centroid_normalized = tf.clip_by_value(centroid_normalized, 0.0, 1.0)
    return centroid_normalized


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def load_trained_model(config, workdir):
    """Load trained VAE from saved weights."""
    
    # Get model architecture parameters
    latent_dim = config['VAE'].getint('latent_dim')
    n_units = config['VAE'].getint('n_units')
    num_octaves = config['audio'].getint('num_octaves')
    bins_per_octave = config['audio'].getint('bins_per_octave')
    n_bins = int(num_octaves * bins_per_octave)
    VAE_output_activation = config['VAE'].get('output_activation')
    
    # Spectral centroid params (if used)
    spectral_centroid_weight = config['VAE'].getfloat('spectral_centroid_weight', fallback=0.0)
    centroid_dim = config['VAE'].getint('centroid_dim', fallback=0)
    kl_beta = config['VAE'].getfloat('kl_beta')
    
    print(f"\n=== Model Architecture ===")
    print(f"Input shape: (None, {n_bins})")
    print(f"Latent dim: {latent_dim}")
    print(f"Hidden units: {n_units}")
    print(f"Output activation: {VAE_output_activation}")
    
    # Define encoder model
    original_inputs = tf.keras.Input(shape=(n_bins,), name='encoder_input')
    x = layers.Dense(n_units, activation='relu')(original_inputs)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()((z_mean, z_log_var))
    encoder = tf.keras.Model(inputs=original_inputs, outputs=[z_mean, z_log_var, z], name='encoder')
    
    # Define decoder model
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(n_units, activation='relu')(latent_inputs)
    outputs = layers.Dense(n_bins, activation=VAE_output_activation)(x)
    decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')
    
    # Define VAE model (simplified version for analysis)
    class VAE(tf.keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
        
        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstruction = self.decoder(z)
            return reconstruction
    
    # Create VAE instance
    vae = VAE(encoder, decoder)
    
    # Build model with dummy input
    dummy = tf.zeros((1, n_bins))
    _ = vae(dummy)
    
    print("\n=== Loading Weights ===")
    
    # Try to load weights
    model_dir = Path(workdir) / 'model'
    weight_files = [
        model_dir / 'mymodel_last.h5',
        model_dir / 'mymodel_best.h5',
    ]
    
    loaded = False
    for weight_file in weight_files:
        if weight_file.exists():
            print(f"Trying to load: {weight_file}")
            try:
                vae.load_weights(str(weight_file))
                print(f"✓ Loaded weights from: {weight_file}")
                loaded = True
                break
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
    if not loaded:
        raise ValueError(
            f"Failed to load model weights from {model_dir}\n"
            f"Expected files: mymodel_last.h5 or encoder_weights.h5/decoder_weights.h5"
        )
    
    # Verify model works
    test_output = vae(dummy, training=False)
    print(f"✓ Model verified, output shape: {test_output.shape}")
    
    return vae, {
        'spectral_centroid_weight': spectral_centroid_weight,
        'centroid_dim': centroid_dim,
        'bins_per_octave': bins_per_octave,
        'n_bins': n_bins
    }


def check_latent_space_health(vae, data_sample):
    """Check if latent space is healthy (not collapsed)."""
    print("\n=== Checking Latent Space Health ===")
    
    # Encode sample
    encoder_output = vae.encoder.predict(data_sample[:100], batch_size=64, verbose=0)
    z_mean = encoder_output[0]  # [n_samples, latent_dim]
    z_log_var = encoder_output[1]
    
    # Check statistics
    print(f"z_mean stats:")
    print(f"  Shape: {z_mean.shape}")
    print(f"  Min: {z_mean.min():.6f}")
    print(f"  Max: {z_mean.max():.6f}")
    print(f"  Mean: {z_mean.mean():.6f}")
    print(f"  Std: {z_mean.std():.6f}")
    
    print(f"\nz_log_var stats:")
    print(f"  Min: {z_log_var.min():.6f}")
    print(f"  Max: {z_log_var.max():.6f}")
    print(f"  Mean: {z_log_var.mean():.6f}")
    
    # Check for collapse
    if z_mean.std() < 0.01:
        print("⚠️  WARNING: Latent space may have collapsed (very low variance)")
        print("   This happens when KL weight is too high or training failed")
        return False
    
    # Check per-dimension variance
    dim_stds = z_mean.std(axis=0)
    inactive_dims = np.sum(dim_stds < 0.01)
    
    print(f"\nPer-dimension analysis:")
    print(f"  Active dimensions: {z_mean.shape[1] - inactive_dims}/{z_mean.shape[1]}")
    print(f"  Inactive dimensions: {inactive_dims}")
    print(f"  Mean std per dim: {dim_stds.mean():.6f}")
    print(f"  Min/Max std: [{dim_stds.min():.6f}, {dim_stds.max():.6f}]")
    
    if inactive_dims > z_mean.shape[1] * 0.5:
        print(f"⚠️  WARNING: {inactive_dims} dimensions are inactive (>50%)")
        print("   Consider reducing latent_dim or adjusting KL beta")
        return False
    
    print("✓ Latent space appears healthy")
    return True


def main():
    args = parse_arguments()
    
    # Load config
    config = configparser.ConfigParser(allow_no_value=True)
    try:
        config.read(args.config)
    except FileNotFoundError:
        print(f'Config File Not Found at {args.config}')
        sys.exit(1)
    
    # Get workspace directory
    if config['extra'].get('run_path'):
        workdir = Path(config['extra'].get('run_path'))
    else:
        print("Error: No workspace found in config. Train a model first!")
        sys.exit(1)
    
    # Get CQT dataset path
    dataset = Path(config['dataset'].get('datapath'))
    cqt_dataset = config['dataset'].get('cqt_dataset')
    my_cqt = dataset / cqt_dataset
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = workdir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("VAE INTERPRETABILITY ANALYSIS")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Workspace: {workdir}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Load model
    print("\n=== Loading Trained Model ===")
    vae, model_params = load_trained_model(config, workdir)
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    cqt_data, labels, metadata = load_nsynth_with_labels(my_cqt)
    print(f"Loaded {len(cqt_data)} samples")
    print(f"CQT shape: {cqt_data.shape}")
    print(f"Instrument families: {np.unique(labels)}")
    
    # Sample subset for analysis
    if len(cqt_data) > args.n_samples:
        print(f"\nSubsampling {args.n_samples} samples for analysis...")
        indices = np.random.choice(len(cqt_data), args.n_samples, replace=False)
        cqt_data = cqt_data[indices]
        labels = labels[indices]
    
    # Check latent space health
    is_healthy = check_latent_space_health(vae, cqt_data)
    
    if not is_healthy:
        print("\n⚠️  Latent space issues detected. Analysis may produce poor results.")
        print("Consider retraining with:")
        print("  - Lower kl_beta (e.g., 0.001 instead of 0.01)")
        print("  - More training epochs")
        print("  - Check reconstruction loss is decreasing")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # === 1. Latent Distribution ===
    print("\n=== Analyzing Latent Distributions ===")
    try:
        encoder_output = vae.encoder.predict(cqt_data, batch_size=64, verbose=0)
        z_mean = encoder_output[0]
        
        plot_latent_distribution(
            z_mean,
            output_dir / "latent_distributions.png"
        )
    except Exception as e:
        print(f"⚠️  Latent distribution plot failed: {e}")
    
    # === 2. Correlation Analysis ===
    if model_params['spectral_centroid_weight'] > 0:
        print("\n=== Computing Descriptor Correlations ===")
        
        try:
            # Create descriptor function with correct parameters
            def centroid_fn(data):
                return compute_spectral_centroid_tf(
                    data,
                    fmin=32.7,
                    bins_per_octave=model_params['bins_per_octave']
                )
            
            results = compute_descriptor_correlation(
                vae, 
                cqt_data, 
                centroid_fn,
                model_params['centroid_dim']
            )
            
            if np.isnan(results['pearson_r']):
                print("⚠️  Correlation is NaN - latent dimension may be constant")
            else:
                print(f"Spectral Centroid ↔ Dim {model_params['centroid_dim']}:")
                print(f"  Pearson r = {results['pearson_r']:.4f} (p = {results['pearson_p']:.4e})")
                print(f"  Spearman r = {results['spearman_r']:.4f} (p = {results['spearman_p']:.4e})")
            
            plot_descriptor_correlation(
                *results['scatter_data'],
                "Spectral Centroid",
                model_params['centroid_dim'],
                output_dir / "centroid_correlation.png"
            )
        except Exception as e:
            print(f"⚠️  Correlation analysis failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n=== Skipping Correlation Analysis ===")
        print("No spectral centroid regularization was used during training")
    
    # === 3. Disentanglement Metrics ===
    print("\n=== Computing Disentanglement Scores ===")
    
    try:
        descriptors = {
            'spectral_centroid': lambda data: compute_spectral_centroid_tf(
                data,
                fmin=32.7,
                bins_per_octave=model_params['bins_per_octave']
            )
        }
        
        scores = compute_disentanglement_score(vae, cqt_data, descriptors)
        
        # Save scores to json file
        with open(output_dir / 'disentanglement_scores.json', 'w') as f:
            json.dump(scores, f, indent=4)
        
        print(f"\n✓ Saved disentanglement scores to: {output_dir / 'disentanglement_scores.json'}")
        
    except Exception as e:
        print(f"⚠️  Disentanglement analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # === 4. Independence Test ===
    print("\n=== Testing Latent Independence ===")
    
    try:
        corr_matrix, mean_offdiag = independence_test(vae, cqt_data)
        
        # Save correlation matrix
        np.save(output_dir / 'latent_correlation_matrix.npy', corr_matrix)
        print(f"✓ Saved correlation matrix to: {output_dir / 'latent_correlation_matrix.npy'}")
        
    except Exception as e:
        print(f"⚠️  Independence test failed: {e}")
    
    # === 5. t-SNE Visualization ===
    print("\n=== Generating t-SNE Visualization ===")
    
    try:
        # Sample balanced data for t-SNE
        unique_labels = np.unique(labels)
        n_per_class = max(10, min(100, len(cqt_data) // (len(unique_labels) * 2)))
        
        tsne_data, tsne_labels = stratified_sample(
            cqt_data, labels,
            n_per_class=n_per_class
        )
        
        print(f"t-SNE sample size: {len(tsne_data)} ({n_per_class} per class)")
        
        if len(tsne_data) < 30:
            print("⚠️  Warning: t-SNE sample size is small, results may be unreliable")
        
        # Encode to latent space
        encoder_output = vae.encoder.predict(tsne_data, batch_size=64, verbose=0)
        z_mean = encoder_output[0]
        
        # Adjust perplexity based on sample size
        perplexity = min(30, max(5, len(tsne_data) // 4))
        
        plot_tsne(
            z_mean,
            tsne_labels,
            unique_labels,
            output_dir / "tsne_by_instrument.png",
            perplexity=perplexity
        )
        
    except Exception as e:
        print(f"⚠️  t-SNE visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # === 6. Latent Traversal ===
    print("\n=== Generating Latent Traversals ===")
    
    try:
        base_sample = cqt_data[3:4]  # Use first sample
        
        # Traverse centroid dimension if disentanglement is enabled
        if model_params['spectral_centroid_weight'] > 0:
            dims_to_traverse = [
                model_params['centroid_dim'],
                (model_params['centroid_dim'] + 1) % config['VAE'].getint('latent_dim'),
                (model_params['centroid_dim'] + 2) % config['VAE'].getint('latent_dim')
            ]
        else:
            dims_to_traverse = [0, 1, 2]
        
        for dim in dims_to_traverse:
            if dim < config['VAE'].getint('latent_dim'):
                plot_latent_traversal(
                    vae, base_sample, dim,
                    save_path=output_dir / f"traversal_dim{dim}.png",
                    bins_per_octave=model_params['bins_per_octave'],
                    num_octaves=config['audio'].getint('num_octaves')
                )
        
    except Exception as e:
        print(f"⚠️  Latent traversal failed: {e}")
        import traceback
        traceback.print_exc()
    
    # === Summary ===
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")
    
    if is_healthy:
        print("\n✓ Model appears to be working correctly")
        if model_params['spectral_centroid_weight'] > 0:
            print(f"✓ Spectral centroid disentanglement enabled (dim {model_params['centroid_dim']})")
    else:
        print("\n⚠️  Model has issues - consider retraining")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()