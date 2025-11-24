"""
Quantitative interpretability metrics.
"""

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def compute_descriptor_correlation(vae, data, descriptor_fn, target_dim):
    """
    Compute correlation between a descriptor and a latent dimension.
    
    Args:
        vae: Trained VAE model
        data: CQT samples [n_samples, n_bins]
        descriptor_fn: Function that computes descriptor from CQT
        target_dim: Which latent dimension should correlate
        
    Returns:
        pearson_r: Pearson correlation coefficient
        spearman_r: Spearman correlation coefficient (for non-linear)
        p_value: Statistical significance
        scatter_data: (descriptor_values, latent_values) for plotting
    """
    # 1. Encode data to get latent representations
    print(f"Encoding {len(data)} samples...")
    encoder_output = vae.encoder.predict(data, batch_size=64, verbose=0)
    z_mean = encoder_output[0]  # [n_samples, latent_dim]
    latent_values = z_mean[:, target_dim]
    
    # 2. Compute descriptor values for each sample
    print(f"Computing descriptor values...")
    descriptor_values = descriptor_fn(data)
    
    # Convert to numpy if tensorflow
    if isinstance(descriptor_values, tf.Tensor):
        descriptor_values = descriptor_values.numpy()
    
    # 3. Calculate correlations
    pearson_r, pearson_p = pearsonr(descriptor_values, latent_values)
    spearman_r, spearman_p = spearmanr(descriptor_values, latent_values)
    
    # 4. Return results
    results = {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'scatter_data': (descriptor_values, latent_values),
        'n_samples': len(data)
    }
    
    return results


def compute_sap_score(latent_vectors, descriptor_values_dict):
    """
    Compute SAP (Separated Attribute Predictability) score.
    
    Measures how well individual latent dimensions predict individual descriptors.
    Higher score = better disentanglement.
    
    Args:
        latent_vectors: [n_samples, latent_dim] encoded representations
        descriptor_values_dict: {'descriptor_name': [n_samples] values}
        
    Returns:
        sap_score: Float between 0 and 1 (higher is better)
        per_descriptor_scores: Dictionary with scores for each descriptor
    """
    print("\n=== Computing SAP Score ===")
    
    n_samples, latent_dim = latent_vectors.shape
    descriptor_names = list(descriptor_values_dict.keys())
    n_descriptors = len(descriptor_names)
    
    # Normalize latent vectors
    scaler = StandardScaler()
    latent_normalized = scaler.fit_transform(latent_vectors)
    
    per_descriptor_scores = {}
    sap_scores = []
    
    for desc_name in descriptor_names:
        desc_values = descriptor_values_dict[desc_name]
        
        # Compute correlation of each latent dim with descriptor
        correlations = np.zeros(latent_dim)
        for i in range(latent_dim):
            correlations[i] = np.abs(pearsonr(latent_normalized[:, i], desc_values)[0])
        
        # SAP score: difference between top 2 correlations
        # (High score = one dimension dominates)
        sorted_corr = np.sort(correlations)[::-1]
        if len(sorted_corr) >= 2:
            sap = sorted_corr[0] - sorted_corr[1]
        else:
            sap = sorted_corr[0]
        
        per_descriptor_scores[desc_name] = {
            'sap': sap,
            'top_dim': int(np.argmax(correlations)),
            'top_correlation': float(sorted_corr[0]),
            'second_correlation': float(sorted_corr[1]) if len(sorted_corr) >= 2 else 0.0
        }
        
        sap_scores.append(sap)
        
        print(f"  {desc_name:20s}: SAP = {sap:.4f}, Top dim = {per_descriptor_scores[desc_name]['top_dim']}")
    
    # Overall SAP score: mean across descriptors
    overall_sap = np.mean(sap_scores)
    
    print(f"\n  Overall SAP Score: {overall_sap:.4f}")
    print(f"  (Higher is better, range [0, 1])")
    
    return overall_sap, per_descriptor_scores


def compute_mig_score(latent_vectors, descriptor_values_dict, n_bins=20):
    """
    Compute MIG (Mutual Information Gap) score.
    
    Measures mutual information between latent dimensions and descriptors.
    Higher score = better disentanglement.
    
    Args:
        latent_vectors: [n_samples, latent_dim] encoded representations
        descriptor_values_dict: {'descriptor_name': [n_samples] values}
        n_bins: Number of bins for discretization
        
    Returns:
        mig_score: Float (higher is better)
        per_descriptor_mig: Dictionary with MIG for each descriptor
    """
    from sklearn.metrics import mutual_info_score
    
    print("\n=== Computing MIG Score ===")
    
    n_samples, latent_dim = latent_vectors.shape
    descriptor_names = list(descriptor_values_dict.keys())
    
    per_descriptor_mig = {}
    mig_scores = []
    
    for desc_name in descriptor_names:
        desc_values = descriptor_values_dict[desc_name]
        
        # Discretize descriptor values
        desc_discrete = np.digitize(desc_values, bins=np.linspace(desc_values.min(), desc_values.max(), n_bins))
        
        # Compute mutual information for each latent dimension
        mi_values = np.zeros(latent_dim)
        for i in range(latent_dim):
            # Discretize latent dimension
            lat_discrete = np.digitize(latent_vectors[:, i], 
                                      bins=np.linspace(latent_vectors[:, i].min(), 
                                                      latent_vectors[:, i].max(), n_bins))
            
            # Compute MI
            mi_values[i] = mutual_info_score(desc_discrete, lat_discrete)
        
        # MIG: difference between top 2 MI values, normalized
        sorted_mi = np.sort(mi_values)[::-1]
        if len(sorted_mi) >= 2 and sorted_mi[0] > 0:
            mig = (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        else:
            mig = 0.0
        
        per_descriptor_mig[desc_name] = {
            'mig': mig,
            'top_dim': int(np.argmax(mi_values)),
            'top_mi': float(sorted_mi[0]),
            'second_mi': float(sorted_mi[1]) if len(sorted_mi) >= 2 else 0.0
        }
        
        mig_scores.append(mig)
        
        print(f"  {desc_name:20s}: MIG = {mig:.4f}, Top dim = {per_descriptor_mig[desc_name]['top_dim']}")
    
    overall_mig = np.mean(mig_scores)
    
    print(f"\n  Overall MIG Score: {overall_mig:.4f}")
    print(f"  (Higher is better, range [0, 1])")
    
    return overall_mig, per_descriptor_mig


def compute_dci_score(latent_vectors, descriptor_values_dict):
    """
    Compute DCI (Disentanglement, Completeness, Informativeness) metrics.
    
    Uses Random Forest to measure how well latent dims predict descriptors.
    
    Args:
        latent_vectors: [n_samples, latent_dim] encoded representations
        descriptor_values_dict: {'descriptor_name': [n_samples] values}
        
    Returns:
        dci_scores: Dictionary with disentanglement, completeness, informativeness
    """
    print("\n=== Computing DCI Scores ===")
    
    n_samples, latent_dim = latent_vectors.shape
    descriptor_names = list(descriptor_values_dict.keys())
    n_descriptors = len(descriptor_names)
    
    # Train Random Forest to predict each descriptor from latent codes
    importance_matrix = np.zeros((n_descriptors, latent_dim))
    prediction_scores = []
    
    for i, desc_name in enumerate(descriptor_names):
        desc_values = descriptor_values_dict[desc_name]
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(latent_vectors, desc_values)
        
        # Get feature importances
        importance_matrix[i, :] = rf.feature_importances_
        
        # Cross-validated R² score (informativeness)
        scores = cross_val_score(rf, latent_vectors, desc_values, cv=5, scoring='r2')
        prediction_scores.append(np.mean(scores))
        
        print(f"  {desc_name:20s}: R² = {np.mean(scores):.4f}")
    
    # Normalize importance matrix (rows sum to 1)
    importance_matrix = importance_matrix / (importance_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    # Disentanglement: how much each latent focuses on one descriptor
    # Entropy of importance across descriptors (lower entropy = more disentangled)
    disentanglement_scores = []
    for j in range(latent_dim):
        importance_col = importance_matrix[:, j]
        if importance_col.sum() > 0:
            # Normalized entropy
            entropy = -np.sum(importance_col * np.log(importance_col + 1e-10))
            max_entropy = np.log(n_descriptors)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            disentanglement_scores.append(1 - normalized_entropy)  # Higher = more disentangled
        else:
            disentanglement_scores.append(0)
    
    disentanglement = np.mean(disentanglement_scores)
    
    # Completeness: how much each descriptor is captured by one latent
    completeness_scores = []
    for i in range(n_descriptors):
        importance_row = importance_matrix[i, :]
        entropy = -np.sum(importance_row * np.log(importance_row + 1e-10))
        max_entropy = np.log(latent_dim)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        completeness_scores.append(1 - normalized_entropy)
    
    completeness = np.mean(completeness_scores)
    
    # Informativeness: how well we can predict descriptors
    informativeness = np.mean(prediction_scores)
    
    dci_scores = {
        'disentanglement': disentanglement,
        'completeness': completeness,
        'informativeness': informativeness,
        'importance_matrix': importance_matrix,
        'per_descriptor_r2': dict(zip(descriptor_names, prediction_scores))
    }
    
    print(f"\n  Disentanglement: {disentanglement:.4f} (higher = each latent focuses on one descriptor)")
    print(f"  Completeness:    {completeness:.4f} (higher = each descriptor captured by one latent)")
    print(f"  Informativeness: {informativeness:.4f} (higher = latents predict descriptors well)")
    
    return dci_scores


def compute_disentanglement_score(vae, data, descriptors_dict):
    """
    Compute all disentanglement metrics (SAP, MIG, DCI).
    
    Args:
        vae: Trained VAE model
        data: CQT samples [n_samples, n_bins]
        descriptors_dict: {'centroid': fn, 'attack': fn, ...}
        
    Returns:
        Dictionary with all disentanglement scores
    """
    print("\n" + "="*60)
    print("COMPUTING DISENTANGLEMENT METRICS")
    print("="*60)
    
    # Encode data
    print(f"\nEncoding {len(data)} samples...")
    encoder_output = vae.encoder.predict(data, batch_size=64, verbose=0)
    z_mean = encoder_output[0]  # [n_samples, latent_dim]

    # Compute all descriptors
    print("\nComputing descriptors...")
    descriptor_values_dict = {}
    for desc_name, desc_fn in descriptors_dict.items():
        print(f"  Computing {desc_name}...")
        values = desc_fn(data)
        if isinstance(values, tf.Tensor):
            values = values.numpy()
        descriptor_values_dict[desc_name] = values
    
    # Compute metrics
    sap_score, sap_details = compute_sap_score(z_mean, descriptor_values_dict)
    mig_score, mig_details = compute_mig_score(z_mean, descriptor_values_dict)
    dci_scores = compute_dci_score(z_mean, descriptor_values_dict)
    
    # Compile results
    results = {
        'sap_score': sap_score,
        'sap_details': sap_details,
        'mig_score': mig_score,
        'mig_details': mig_details,
        'dci_disentanglement': dci_scores['disentanglement'],
        'dci_completeness': dci_scores['completeness'],
        'dci_informativeness': dci_scores['informativeness'],
        'dci_per_descriptor_r2': dci_scores['per_descriptor_r2']
    }
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF DISENTANGLEMENT METRICS")
    print("="*60)
    print(f"SAP Score:           {sap_score:.4f}  (higher = better disentanglement)")
    print(f"MIG Score:           {mig_score:.4f}  (higher = better disentanglement)")
    print(f"DCI Disentanglement: {dci_scores['disentanglement']:.4f}")
    print(f"DCI Completeness:    {dci_scores['completeness']:.4f}")
    print(f"DCI Informativeness: {dci_scores['informativeness']:.4f}")
    print("="*60 + "\n")
    
    return results


def independence_test(vae, data):
    """
    Test independence between latent dimensions.
    Returns correlation matrix of latent dimensions.
    
    Args:
        vae: Trained VAE model
        data: CQT samples [n_samples, n_bins]
        
    Returns:
        correlation_matrix: [latent_dim, latent_dim] correlation matrix
        mean_abs_offdiag: Average absolute off-diagonal correlation (should be low)
    """
    print("\n=== Testing Latent Independence ===")
    
    # Encode data
    encoder_output = vae.encoder.predict(data, batch_size=64, verbose=0)
    z_mean = encoder_output[0]
    latent_dim = z_mean.shape[1]
    
    # Compute correlation matrix
    correlation_matrix = np.corrcoef(z_mean.T)
    
    # Compute mean absolute off-diagonal correlation
    # (excludes diagonal which is always 1)
    mask = ~np.eye(latent_dim, dtype=bool)
    off_diagonal_corr = correlation_matrix[mask]
    mean_abs_offdiag = np.mean(np.abs(off_diagonal_corr))
    max_abs_offdiag = np.max(np.abs(off_diagonal_corr))
    
    print(f"Latent dimensions: {latent_dim}")
    print(f"Mean absolute off-diagonal correlation: {mean_abs_offdiag:.4f}")
    print(f"Max absolute off-diagonal correlation:  {max_abs_offdiag:.4f}")
    print(f"(Lower values indicate more independent dimensions)")
    
    # Find highly correlated pairs
    print("\nHighly correlated dimension pairs (|r| > 0.5):")
    for i in range(latent_dim):
        for j in range(i+1, latent_dim):
            if np.abs(correlation_matrix[i, j]) > 0.5:
                print(f"  Dim {i:3d} ↔ Dim {j:3d}: r = {correlation_matrix[i, j]:+.4f}")
    
    return correlation_matrix, mean_abs_offdiag


def compute_modularity_score(vae, data, descriptors_dict):
    """
    Compute modularity: how well individual latent dims control individual descriptors
    independently of other descriptors.
    
    Args:
        vae: Trained VAE model
        data: CQT samples
        descriptors_dict: Dictionary of descriptor functions
        
    Returns:
        modularity_score: Float between 0 and 1
    """
    print("\n=== Computing Modularity Score ===")
    
    # Encode data
    z_mean, _ = vae.encoder.predict(data, batch_size=64, verbose=0)
    
    # Compute all descriptors
    descriptor_values = {}
    for name, fn in descriptors_dict.items():
        values = fn(data)
        if isinstance(values, tf.Tensor):
            values = values.numpy()
        descriptor_values[name] = values
    
    descriptor_names = list(descriptor_values.keys())
    n_descriptors = len(descriptor_names)
    latent_dim = z_mean.shape[1]
    
    # For each latent dimension, measure how many descriptors it affects
    descriptor_sensitivities = np.zeros((latent_dim, n_descriptors))
    
    for i in range(latent_dim):
        for j, desc_name in enumerate(descriptor_names):
            # Correlation = sensitivity
            corr = np.abs(pearsonr(z_mean[:, i], descriptor_values[desc_name])[0])
            descriptor_sensitivities[i, j] = corr
    
    # Modularity: each latent should affect only one descriptor
    # Use Gini coefficient: 0 = all equal (bad), 1 = one dominant (good)
    modularity_scores = []
    for i in range(latent_dim):
        sensitivities = descriptor_sensitivities[i]
        sorted_sens = np.sort(sensitivities)
        n = len(sorted_sens)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_sens)) / (n * np.sum(sorted_sens)) - (n + 1) / n
        modularity_scores.append(gini)
    
    overall_modularity = np.mean(modularity_scores)
    
    print(f"Modularity score: {overall_modularity:.4f}")
    print(f"(Higher = each latent affects fewer descriptors)")
    
    return overall_modularity