import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def calculate_kmeans_and_silhouette(points, n_clusters=2, init_method='maxmin'):
    """
    Perform k-means clustering and calculate the silhouette score.

    Parameters:
    points (numpy.ndarray): 1D array of points.
    n_clusters (int): Number of clusters for k-means. Default is 2.
    init_method (str): Method for initializing the k-means centers. 
                       Options are 'k-means++' and 'maxmin'. Default is 'k-means++'.

    Returns:
    dict: A dictionary with keys:
        - 'labels': The cluster labels assigned to each point.
        - 'centroids': The centroids of the clusters.
        - 'silhouette_score': The silhouette score of the clustering.
    """
    if len(points.shape) == 1:
        points = points.reshape(-1, 1)

    # Initialize k-means centers
    if init_method == 'maxmin':
        init_centers = np.array([points.min(), points.max()]).reshape(-1, 1)
        if n_clusters > 2:
            raise ValueError("init_method 'maxmin' only supports n_clusters=2")
    else:
        init_centers = init_method

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, init=init_centers)
    labels = kmeans.fit_predict(points)
    centroids = kmeans.cluster_centers_

    # Calculate silhouette score
    if len(np.unique(labels)) > 1:  # Silhouette score requires at least 2 clusters
        score = silhouette_score(points, labels)
    else:
        score = -1  # Invalid silhouette score if all points are in one cluster

    return {
        'labels': labels,
        'centroids': centroids,
        'silhouette_score': score
    }

import numpy as np

def normalize_sources(S_est):
    S_est_abs = np.abs(S_est)
    S_est_normalized = S_est_abs / S_est_abs.max(axis=1, keepdims=True)
    return S_est_normalized

# TODO 
# add a robustness 
def compute_recall(source_est, source_true, threshold_est, threshold_true):
    """Compute recall between an estimated source and a true source."""
    spikes_est = np.where(source_est > threshold_est)[0]
    spikes_true = np.where(source_true > threshold_true)[0]

    if len(spikes_true) == 0:
        return 0  # No true spikes means recall is undefined (set to 0)
    
    # Align estimated spikes to true source based on first spike
    if len(spikes_est) == 0:
        return 0  # No estimated spikes

    align_shift = spikes_true[0] - spikes_est[0]
    spikes_est_aligned = spikes_est + align_shift

    true_positives = np.intersect1d(spikes_est_aligned, spikes_true).size
    false_negatives = np.setdiff1d(spikes_true, spikes_est_aligned).size

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall


import numpy as np




def find_kmeans_threshold(signal):
    """
    Given a 1D signal, run k-means (with 2 clusters) to separate 'non-spikes' and 'spikes'.
    Return the minimum value of the cluster with the highest centroid as the threshold.
    """
    result = calculate_kmeans_and_silhouette(signal, n_clusters=2, init_method='maxmin')
    labels = result['labels']
    centroids = result['centroids'].flatten()  # shape (2,)

    # Identify which cluster has the highest centroid
    high_cluster_idx = np.argmax(centroids)

    # Get all points belonging to that cluster
    cluster_points = signal[labels == high_cluster_idx]

    # The threshold is the minimum value in the high centroid cluster
    if cluster_points.size == 0:
        # If for some reason no points ended up in that cluster, fallback
        return np.mean(signal)  # fallback threshold, or 0, etc.

    threshold_value = cluster_points.min()
    return threshold_value

def compute_precision_recall(source_est, source_true, threshold1, threshold2):
    # Get spike times (indices where values exceed threshold)
    spikes_est = np.where(source_est > threshold1)[0]
    spikes_true = np.where(source_true > threshold2)[0]

    if len(spikes_est) == 0 or len(spikes_true) == 0:
        return "No spikes detected in one or both sources."

    # Align on the first spike
    align_shift = spikes_true[0] - spikes_est[0]
    spikes_est_aligned = spikes_est + align_shift

    # Compute True Positives, False Positives, and False Negatives
    true_positives = np.intersect1d(spikes_est_aligned, spikes_true).size
    false_positives = np.setdiff1d(spikes_est_aligned, spikes_true).size
    false_negatives = np.setdiff1d(spikes_true, spikes_est_aligned).size

    # Compute precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

def match_estimates_to_true(S_est, S_true, threshold_est, threshold_true, ):
    """
    Match each estimated source to the best true source using recall as a distance metric.
    
    Parameters:
    - S_est: np.array of shape (m, T), estimated sources
    - S_true: np.array of shape (n, T), true sources
    - threshold_est: Either a numeric value or 'kmeans' 
                     (if 'kmeans', compute threshold individually for each row in S_est).
    - threshold_true: Either a numeric value or 'kmeans' 
                      (if 'kmeans', compute threshold individually for each row in S_true).
    
    Returns:
    - matches: A list (length m) of the best matched true source index for each estimated source.
    - recall_matrix: np.array of shape (m, n) containing recall values.
    - thresholds_est: list of computed thresholds for each row in S_est.
    - thresholds_true: list of computed thresholds for each row in S_true.
    """
    m, T = S_est.shape
    n, _ = S_true.shape

    # 1) Figure out thresholds for each row in S_est
    if threshold_est == 'kmeans':
        thresholds_est = [find_kmeans_threshold(S_est[i]) for i in range(m)]
    else:
        # same threshold for all estimated sources
        thresholds_est = [threshold_est] * m

    # 2) Figure out thresholds for each row in S_true
    if threshold_true == 'kmeans':
        thresholds_true = [find_kmeans_threshold(S_true[j]) for j in range(n)]
    else:
        # same threshold for all true sources
        thresholds_true = [threshold_true] * n

    # 3) Build the recall matrix
    recall_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            recall_matrix[i, j] = compute_recall(
                S_est[i],
                S_true[j],
                thresholds_est[i],
                thresholds_true[j]
            )

    # 4) Assign each estimated source to the true source with the highest recall
    matches = np.argmax(recall_matrix, axis=1)

    return matches, recall_matrix, thresholds_est, thresholds_true



def evaluate_matches(
    S_est, 
    S_true, 
    matches, 
    thresholds_est, 
    thresholds_true
):
    """
    Given matched pairs of sources (each estimated source matched to a true source),
    compute precision and recall for each matched pair, then return:
      - precision_list
      - recall_list
      - mean_precision
      - mean_recall
    """

    # Number of estimated sources
    m = len(matches)  

    precision_list = np.zeros(m)
    recall_list = np.zeros(m)

    for i in range(m):
        true_idx = matches[i]
        precision, recall = compute_precision_recall(
            S_est[i], 
            S_true[true_idx], 
            thresholds_est[i], 
            thresholds_true[true_idx]
        )
        precision_list[i] = precision
        recall_list[i] = recall

    # Compute mean
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)

    return precision_list, recall_list, mean_precision, mean_recall

# This is the more robust implementation but does not seem to work well either.
def compute_recall_robust(source_est, source_true,
                          threshold_est, threshold_true,
                          R=5):
    """
    Compute recall between an estimated source and a true source.
    We test up to R^2 alignments based on the first R spikes
    in each signal (or fewer, if fewer than R spikes exist).

    We stop as soon as we find a recall >= 0.25, or once all
    pairs (among the first R) are exhausted.
    """
    spikes_est = np.where(source_est > threshold_est)[0]
    spikes_true = np.where(source_true > threshold_true)[0]

    # If there are no true spikes or no estimated spikes, recall=0
    if len(spikes_true) == 0 or len(spikes_est) == 0:
        return 0.0

    # Restrict to the first R spikes in each list, if available
    first_r_est = spikes_est[:R]
    first_r_true = spikes_true[:R]

    best_recall = 0.0

    # Try all pairwise alignments among these up to R spikes
    for e_spike in first_r_est:
        for t_spike in first_r_true:
            # Compute alignment shift to match e_spike -> t_spike
            align_shift = t_spike - e_spike
            # Align all estimated spikes
            spikes_est_aligned = spikes_est + align_shift

            # Compute recall
            true_positives = np.intersect1d(spikes_est_aligned, spikes_true).size
            false_negatives = len(spikes_true) - true_positives  # everything in true not matched
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0.0

            # Update best recall
            if recall > best_recall:
                best_recall = recall

            # Early stop if recall is above 0.25
            if best_recall >= 0.25:
                return best_recall

    return best_recall