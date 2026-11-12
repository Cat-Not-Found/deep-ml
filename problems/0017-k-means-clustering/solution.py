import torch

def k_means_clustering(points, k, initial_centroids, max_iterations) -> list[tuple[float, ...]]:
    """
    Perform k-means clustering on `points` into `k` clusters.
    points: tensor of shape (n_points, n_features)
    initial_centroids: tensor of shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    # Convert to tensors
    points_t = torch.as_tensor(points, dtype=torch.float)
    centroids = torch.as_tensor(initial_centroids, dtype=torch.float)
    # Your implementation here
    for _ in range(max_iterations):
        distances = torch.sum((points_t.unsqueeze(1) - centroids.unsqueeze(0)) ** 2, dim=2)

        # Assign each point to its nearest centroid
        labels = torch.argmin(distances, dim=1)
        new_centroids = centroids.clone()

        for i in range(k):
            cluster_points = points_t[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(dim=0)

        centroids = new_centroids

    return [tuple(x.item() for x in centroid) for centroid in centroids]