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
    def __init__(self, dataset, num_clusters, m=2, error=0.005, maxiter=1000):
        self.dataset = dataset
        self.num_clusters = num_clusters
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.eps = 1e-10  # small value to avoid division by zero
        self.centroids = None
        self.membership_degrees = None

    def fit(self):
        self.__fit_c_means()
        self.__compute_membership_functions()

    def __fit_c_means(self):
        n_features = self.dataset.training_data.shape[0]
        n_samples = self.dataset.training_data.shape[1]

        # expects a matrix of #features x #samples
        self.centroids, self.membership_degrees, _, _, _, _, fpc = fuzz.cluster.cmeans(
            self.dataset.training_data,
            c=self.num_clusters,
            m=self.m,
            error=self.error,
            maxiter=self.maxiter
        )
        # transpose self.membership_degrees to match the paper
        # membership_degrees[i,r] represents the membership degree of ith data point to the rth cluster.
        self.membership_degrees = self.membership_degrees.T
        
        # now we have the cluster centers (in a matrix of #clusters x #features) and the partition matrix (in a matrix of #samples x #clusters)
        # The cluster centroid (Cr) is explained as a vector with m values, as seen in Eq. 2, wherein each value in this vector (i.e., crj) signifies the
        # projection of the centroid of the rth cluster along the jth dimension

        # cluster centers and membership degrees is all we need to compute the "membership functions" which are Gaussian fuzzy membership functions

        print ("Fuzzy C-Means Clustering ended.")
        print (f"fpc: {fpc:.4f}, Fuzzy Partition Coefficient (1 is perfect, 0 is worst, above 0.5 is good).")

        print ("Cluster centers shape: ", self.centroids.shape)
        print ("Membership degrees shape: ", self.membership_degrees.shape)

        assert self.centroids.shape == (self.num_clusters, n_features), "Cluster centers shape mismatch."
        assert self.membership_degrees.shape == (n_samples, self.num_clusters), "Membership degrees shape mismatch."
        print ("OK: Fuzzy C-Means Clustering completed successfully.")

    def __compute_membership_functions(self):
        if self.centroids is None or self.membership_degrees is None:
            raise ValueError("Centroids and membership degrees must be computed before computing membership functions.")
        
        # Compute Gaussian fuzzy membership functions based on cluster centers and membership degrees
        
        # Each fuzzy function’s center is generated using the centroid vector of the relevant cluster, and the breadth
        # is produced by examining the total distance between data points and their membership Along each dimension

        # Specifically, for each feature j, the centroids c1j, c2j, ..., ckj (the j-th column of the "centroids" input parameter) are
        # selected to calculate the width of fuzzy membership functions of that feature.

        n_features = self.dataset.training_data.shape[0]
        self.deltas = np.zeros((self.num_clusters, n_features), dtype=float)  # shape: (num_clusters, num_features)
        
        # TODO review: vectorized by LLM
        for r in range(self.num_clusters):
            # delta_rj = 1/n sum_{i=1}^{n} ( sqrt( \dfrac{-(x_ij - c_rj)^2}{2 * log(U_ir)}) ) )

            # where U_ir is the membership degree of the i-th data point to the r-th cluster
            # and x_ij is the j-th feature of the i-th data point, and c_rj is the j-th feature of the r-th cluster center.

            U_ir = np.clip(self.membership_degrees[:, r], self.eps, 1.0 - self.eps)
            log_u = np.log(U_ir)

            for j in range(n_features):
                x_ij = self.dataset.training_data[j, :]
                c_rj = self.centroids[r, j]

                vals = (-(x_ij - c_rj) ** 2) / (2.0 * log_u)
                vals = np.maximum(vals, 0.0) # edge case
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
            for r in range(self.num_clusters):
                c_rj = self.centroids[r, j]
                d_rj = self.deltas[r, j]
                mu[j, r] = np.exp(-(((sample[j] - c_rj) / (2.0 * d_rj)) ** 2)) + self.eps

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

        # usually the squared distances are used, but the original paper uses the Euclidean distance, so we will use that as well

        # Numerator: sum_j sum_i U_ij^2 * ||C_j - x_i||
        # dists[i, j] = ||x_i - C_j||
        dists = np.linalg.norm(X[:,None,:] - C[None,:,:], axis=2)
        numerator = np.sum((U ** 2) * dists)

        # Denominator: n * min_{j != t} ||C_j - C_t||
        # Pairwise centroid distances
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]       # (k, k, n_features)
        centroid_dists = np.linalg.norm(diff, axis=2)              # (k, k)
        np.fill_diagonal(centroid_dists, np.inf)                # exclude j == t
        denominator = n * centroid_dists.min()

        return numerator / denominator