"""
Visualization functions for latent space analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE


def plot_tsne(latent_vectors, labels, instrument_families, save_path, 
              perplexity=5, learning_rate=200, n_iter=1000):
    """
    Create t-SNE visualization with different shapes for instrument families.
    
    Args:
        latent_vectors: [n_samples, latent_dim] array of latent representations
        labels: [n_samples] array of instrument family labels
        instrument_families: List of unique instrument types
        save_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        learning_rate: t-SNE learning rate
        n_iter: Number of t-SNE iterations
    """
    print(f"\n=== Running t-SNE ===")
    print(f"Samples: {len(latent_vectors)}")
    print(f"Latent dimensions: {latent_vectors.shape[1]}")
    print(f"Perplexity: {perplexity}")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        random_state=42,
        verbose=1
    )
    
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Define marker shapes (cycle through if more than available)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    
    # Define colors (using a colorblind-friendly palette)
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(unique_labels))))
    if len(unique_labels) > 10:
        # Add more colors if needed
        extra_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels) - 10))
        colors = np.vstack([colors, extra_colors])
    
    # Plot each instrument family with unique marker
    for i, family in enumerate(unique_labels):
        mask = labels == family
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        
        ax.scatter(
            latent_2d[mask, 0],
            latent_2d[mask, 1],
            c=[color],
            marker=marker,
            label=family,
            alpha=0.7,
            s=100,  # Larger size to see shapes better
            edgecolors='black',
            linewidths=1.0
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Latent Space Visualization (t-SNE) by Instrument Family', 
                 fontsize=14, fontweight='bold')
    
    # Place legend outside plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, 
              markerscale=1.2, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved t-SNE plot to {save_path}")
    print(f"  Used {len(unique_labels)} different markers for {len(unique_labels)} instrument families")


def plot_descriptor_correlation(descriptor_values, latent_values, 
                                descriptor_name, dim_idx, save_path):
    """
    Scatter plot showing correlation between descriptor and latent dimension.
    
    Args:
        descriptor_values: [n_samples] array of descriptor values
        latent_values: [n_samples] array of values from one latent dimension
        descriptor_name: Name of the descriptor (e.g., 'Spectral Centroid')
        dim_idx: Index of the latent dimension
        save_path: Path to save the plot
    """
    from scipy.stats import pearsonr
    
    # Compute correlation
    r, p_value = pearsonr(descriptor_values, latent_values)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(descriptor_values, latent_values, alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
    
    # Add regression line
    z = np.polyfit(descriptor_values, latent_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(descriptor_values.min(), descriptor_values.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Linear fit')
    
    # Labels and title
    ax.set_xlabel(descriptor_name, fontsize=12)
    ax.set_ylabel(f'Latent Dimension {dim_idx}', fontsize=12)
    ax.set_title(f'{descriptor_name} vs Latent Dimension {dim_idx}\nPearson r = {r:.4f} (p = {p_value:.4e})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved correlation plot to {save_path}")
    print(f"  Correlation: r = {r:.4f}, p = {p_value:.4e}")


def plot_latent_traversal(vae, base_sample, dim_idx, save_path, n_steps=10, bins_per_octave=48, num_octaves=7):
    """
    Visualize what happens when you interpolate along one dimension.
    Shows spectrograms of reconstructions.
    
    Args:
        vae: Trained VAE model
        base_sample: [1, n_bins] or [n_bins] base CQT sample
        dim_idx: Which latent dimension to traverse
        save_path: Path to save the plot
        n_steps: Number of interpolation steps
    """
    if base_sample.ndim == 1:
        base_sample = base_sample[np.newaxis, :]
    
    # Encode base sample
    encoder_output = vae.encoder.predict(base_sample, verbose=0)
    if isinstance(encoder_output, (list, tuple)):
        z_mean = encoder_output[0]
    else:
        z_mean = encoder_output
    
    # Create traversal range
    traversal_range = np.linspace(-3, 3, n_steps)
    
    # Generate reconstructions
    reconstructions = []
    for value in traversal_range:
        z_modified = z_mean.copy()
        z_modified[0, dim_idx] = value
        reconstruction = vae.decoder.predict(z_modified, verbose=0)
        reconstructions.append(reconstruction[0])

    # 1D vector: reshape to 2D for visualization
    # Assume square-ish shape for visualization
    n_bins = reconstructions[0].shape[0]
    n_rows = num_octaves
    n_cols = bins_per_octave
    
    print(f"Reshaping {n_bins} bins to ({n_rows}, {n_cols}) for visualization")
    
    # Create plot with bar charts instead of spectrograms
    fig, axes = plt.subplots(2, n_steps // 2, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, (recon, value) in enumerate(zip(reconstructions, traversal_range)):
        ax = axes[i]
        # Plot as 1D signal
        ax.plot(recon, linewidth=0.5)
        ax.set_title(f'z[{dim_idx}] = {value:.2f}', fontsize=10)
        ax.set_xlabel('Frequency Bin')
        ax.set_ylabel('Magnitude')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(np.max(r) for r in reconstructions) * 1.1])
    
    
    plt.suptitle(f'Latent Dimension {dim_idx} Traversal', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved latent traversal plot to {save_path}")


# Add to analysis/visualization.py:

import librosa
import soundfile as sf

def generate_audio_traversal(vae, base_sample, dim_to_vary,
                            save_dir, n_steps=7, range_scale='auto',
                            bins_per_octave=48, num_octaves=6,
                            sample_rate=16000, hop_length=512, n_iter=32,
                            dataset_sample=None, normalize=True):
    """
    Generate audio files for latent traversal - so you can HEAR the changes.
    
    Args:
        vae: Trained VAE model
        base_sample: [1, n_bins] input CQT sample
        dim_to_vary: Which latent dimension to traverse
        save_dir: Directory to save audio files
        n_steps: Number of steps in the traversal
        range_scale: 'auto' or float for range
        bins_per_octave: CQT bins per octave
        num_octaves: Number of octaves
        sample_rate: Output sample rate
        hop_length: CQT hop length
        n_iter: Griffin-Lim iterations
        dataset_sample: Sample for auto-ranging
        normalize: Whether to normalize output audio
    """
    print(f"\n=== Generating Audio Traversal for Dimension {dim_to_vary} ===")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Encode base sample
    encoder_output = vae.encoder.predict(base_sample, verbose=0)
    z_mean_base = encoder_output[0]  # [1, latent_dim]
    
    print(f"Base value at dim {dim_to_vary}: {z_mean_base[0, dim_to_vary]:.4f}")
    
    # Determine traversal range (same logic as plot_latent_traversal)
    if range_scale == 'auto':
        if dataset_sample is None:
            range_min, range_max = -3.0, 3.0
        else:
            dataset_encoded = vae.encoder.predict(dataset_sample, batch_size=64, verbose=0)
            dataset_z_mean = dataset_encoded[0]
            dim_values = dataset_z_mean[:, dim_to_vary]
            dim_mean = dim_values.mean()
            dim_std = dim_values.std()
            range_min = dim_mean - 2.5 * dim_std
            range_max = dim_mean + 2.5 * dim_std
            range_min = min(range_min, dim_values.min())
            range_max = max(range_max, dim_values.max())
            print(f"Auto-range: [{range_min:.4f}, {range_max:.4f}]")
    elif isinstance(range_scale, tuple):
        range_min, range_max = range_scale
    else:
        range_min = -abs(range_scale)
        range_max = abs(range_scale)
    
    traversal_values = np.linspace(range_min, range_max, n_steps)
    
    # Generate CQT outputs
    outputs = []
    print("\nGenerating latent traversal outputs...")
    for i, value in enumerate(traversal_values):
        z_modified = z_mean_base.copy()
        z_modified[0, dim_to_vary] = value
        
        output_cqt = vae.decoder.predict(z_modified, verbose=0)
        outputs.append(output_cqt[0])
        print(f"  Step {i+1}/{n_steps}: dim[{dim_to_vary}] = {value:.3f}")
    
    outputs = np.array(outputs)  # [n_steps, n_bins]
    
    # Convert to audio using Griffin-Lim
    print("\nConverting to audio with Griffin-Lim...")
    audio_files = []
    
    for i, cqt_magnitude in enumerate(outputs):
        # Transpose for librosa (expects [n_bins, n_frames])
        # We only have 1 frame, so add dummy frames for Griffin-Lim
        cqt_for_gl = np.repeat(cqt_magnitude[:, np.newaxis], 10, axis=1)
        
        # Apply Griffin-Lim
        audio = librosa.griffinlim_cqt(
            cqt_for_gl,
            sr=sample_rate,
            n_iter=n_iter,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            dtype=np.float32
        )
        
        # Normalize if requested
        if normalize:
            audio = librosa.util.normalize(audio)
        
        # Save audio file
        filename = f"dim{dim_to_vary}_step{i:02d}_value{traversal_values[i]:+.2f}.wav"
        filepath = save_dir / filename
        sf.write(filepath, audio, sample_rate)
        audio_files.append(filepath)
        
        print(f"  Saved: {filename}")
    
    # Also create original for comparison
    print("\nGenerating original reconstruction for comparison...")
    z_original = z_mean_base
    original_cqt = vae.decoder.predict(z_original, verbose=0)[0]
    cqt_for_gl = np.repeat(original_cqt[:, np.newaxis], 10, axis=1)
    original_audio = librosa.griffinlim_cqt(
        cqt_for_gl,
        sr=sample_rate,
        n_iter=n_iter,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        dtype=np.float32
    )
    if normalize:
        original_audio = librosa.util.normalize(original_audio)
    
    original_path = save_dir / f"dim{dim_to_vary}_original.wav"
    sf.write(original_path, original_audio, sample_rate)
    print(f"  Saved: {original_path.name}")
    
    # Create a metadata file
    metadata = {
        'dimension': int(dim_to_vary),
        'n_steps': n_steps,
        'range': [float(range_min), float(range_max)],
        'traversal_values': traversal_values.tolist(),
        'files': [f.name for f in audio_files],
        'original_file': original_path.name,
        'parameters': {
            'sample_rate': sample_rate,
            'hop_length': hop_length,
            'bins_per_octave': bins_per_octave,
            'num_octaves': num_octaves,
            'griffin_lim_iterations': n_iter,
            'normalized': normalize
        }
    }
    
    with open(save_dir / f"dim{dim_to_vary}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Audio traversal complete!")
    print(f"  Generated {len(audio_files)} files in {save_dir}")
    print(f"  Listen to them in order to hear how dimension {dim_to_vary} affects the sound")
    
    return audio_files, metadata


def plot_latent_distribution(latent_vectors, save_path):
    """
    Plot distribution of latent dimensions.
    
    Args:
        latent_vectors: [n_samples, latent_dim] array
        save_path: Path to save the plot
    """
    latent_dim = latent_vectors.shape[1]
    
    # Select subset of dimensions to plot
    n_dims_to_plot = min(16, latent_dim)
    dims_to_plot = np.linspace(0, latent_dim - 1, n_dims_to_plot, dtype=int)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]
        values = latent_vectors[:, dim_idx]
        
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.2f}')
        ax.set_xlabel(f'Dimension {dim_idx}')
        ax.set_ylabel('Count')
        ax.set_title(f'Dim {dim_idx} (σ={values.std():.2f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Latent Space Dimension Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved latent distribution plot to {save_path}")