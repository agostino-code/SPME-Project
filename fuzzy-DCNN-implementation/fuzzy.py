# pipeline:
# a sample from the dataset is fed into the fuzzy classifier, which outputs membership degrees for each class (m x k matrix, where m is the number of attributes, k is the number of clusters)
# the membership degrees are then fed into the DCNN regressor, which outputs a single value representing the predicted effort


import numpy as np
import skfuzzy as fuzz


"""
Fit FCM on the training data.

Use centroids and memberships to define Gaussian fuzzy sets.

For each sample, compute a feature-by-cluster membership map.

Feed that map to the DCNN.

Predict effort.
"""

class FuzzyCMeans:
    def __init__(self, dataset, num_clusters, m=2, distance_metric='euclidean', error=0.005, maxiter=1000):
        self.dataset = dataset
        self.num_clusters = num_clusters
        self.m = m
        self.distance_metric = distance_metric
        self.error = error
        self.maxiter = maxiter
        self.eps = 1e-10  # small value to avoid division by zero
        self.centroids = None
        self.membership_degrees = None
        self.inv_cov = None

    def fit(self):
        self.__fit_c_means()
        self.__compute_membership_functions()

    def __fit_c_means(self):
        n_features = self.dataset.training_data.shape[0]
        n_samples = self.dataset.training_data.shape[1]

        # Compute Covariance and Inverse Covariance for Mahalanobis metric
        if self.distance_metric == 'mahalanobis':
            cov = np.cov(self.dataset.training_data)
            if n_features == 1:
                cov = np.array([[cov]])
            cov_reg = cov + np.eye(n_features) * 1e-4
            self.inv_cov = np.linalg.pinv(cov_reg)
        else:
            self.inv_cov = np.eye(n_features)

        # expects a matrix of #features x #samples
        self.centroids, self.membership_degrees, _, _, _, _, fpc = fuzz.cluster.cmeans(
            self.dataset.training_data,
            c=self.num_clusters,
            m=self.m,
            error=self.error,
            maxiter=self.maxiter
        )
        self.membership_degrees = self.membership_degrees.T

        assert self.centroids.shape == (self.num_clusters, n_features), "Cluster centers shape mismatch."
        assert self.membership_degrees.shape == (n_samples, self.num_clusters), "Membership degrees shape mismatch."

    def __compute_membership_functions(self):
        if self.centroids is None or self.membership_degrees is None:
            raise ValueError("Centroids and membership degrees must be computed before computing membership functions.")

        n_features = self.dataset.training_data.shape[0]
        self.deltas = np.zeros((self.num_clusters, n_features), dtype=float)

        for r in range(self.num_clusters):
            U_ir = np.clip(self.membership_degrees[:, r], self.eps, 1.0 - self.eps)
            log_u = np.log(U_ir)

            for j in range(n_features):
                x_ij = self.dataset.training_data[j, :]
                c_rj = self.centroids[r, j]

                vals = (-(x_ij - c_rj) ** 2) / (2.0 * log_u)
                vals = np.maximum(vals, 0.0)
                delta_rj = np.mean(np.sqrt(vals))
                self.deltas[r, j] = max(delta_rj, self.eps)

        return self.deltas

    def transform_one(self, sample):
        if self.centroids is None or self.membership_degrees is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_features = self.dataset.training_data.shape[0]
        mu = np.zeros((n_features, self.num_clusters))

        assert sample.shape[0] == n_features, "Sample feature size does not match fitted data."

        for j in range(n_features):
            weight_j = self.inv_cov[j, j] if self.inv_cov is not None else 1.0
            for r in range(self.num_clusters):
                c_rj = self.centroids[r, j]
                d_rj = self.deltas[r, j]
                dist_sq = ((sample[j] - c_rj) ** 2) * max(weight_j, 1e-4)
                mu[j, r] = np.exp(- dist_sq / (2.0 * d_rj**2))

        return mu

    def transform(self, X):
        if self.centroids is None or self.membership_degrees is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            return self.transform_one(X)

        if X.shape[0] != self.dataset.training_data.shape[0]:
            raise ValueError("Input shape must be (n_features, n_samples) or a single sample.")

        n_samples = X.shape[1]
        n_features = X.shape[0]
        batch = np.zeros((n_samples, n_features, self.num_clusters), dtype=float)

        for i in range(n_samples):
            batch[i] = self.transform_one(X[:, i])

        return batch

    def fit_transform(self):
        self.fit()
        return self.transform(self.dataset.training_data)

    def predict(self, sample):
        return self.transform_one(sample)

    def XB(self):
        if self.centroids is None or self.membership_degrees is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.dataset.training_data.T  # shape: (n_samples, n_features)
        U = self.membership_degrees        # shape: (n_samples, n_clusters)
        C = self.centroids                 # shape: (n_clusters, n_features)
        n = X.shape[0]

        diff = X[:, None, :] - C[None, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        numerator = np.sum((U ** 2) * dists_sq)

        centroid_diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        centroid_dists_sq = np.sum(centroid_diff ** 2, axis=2)
        np.fill_diagonal(centroid_dists_sq, np.inf)
        denominator = n * centroid_dists_sq.min()

        if denominator == 0:
            return float('inf')

        return numerator / denominator