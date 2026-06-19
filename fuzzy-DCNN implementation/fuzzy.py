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
        self.centers = None
        self.membership_degrees = None

    def fit(self):
        self.__fit_c_means()
        self.__compute_membership_functions(self.centers, self.membership_degrees)

    def __fit_c_means(self):
        n_features = self.dataset.training_data.shape[0]
        n_samples = self.dataset.training_data.shape[1]

        # expects a matrix of #features x #samples
        self.centers, self.membership_degrees, _, _, _, _, fpc = fuzz.cluster.cmeans(
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
        print ("fpc: ", fpc, "Fuzzy Partition Coefficient (1 is perfect, 0 is worst, above 0.5 is good).")

        print ("Cluster centers shape: ", self.centers.shape)
        print ("Membership degrees shape: ", self.membership_degrees.shape)

        assert self.centers.shape == (self.num_clusters, n_features), "Cluster centers shape mismatch."
        assert self.membership_degrees.shape == (n_samples, self.num_clusters), "Membership degrees shape mismatch."
        print ("OK: Fuzzy C-Means Clustering completed successfully.")

    def __compute_membership_functions(self, centroids, membership_degrees):
        if centroids is None or membership_degrees is None:
            raise ValueError("Centroids and membership degrees must be computed before computing membership functions.")
        
        # Compute Gaussian fuzzy membership functions based on cluster centers and membership degrees
        
        # Each fuzzy function’s center is generated using the centroid vector of the relevant cluster, and the breadth
        # is produced by examining the total distance between data points and their membership Along each dimension

        # Specifically, for each feature j, the centroids c1j, c2j, ..., ckj (the j-th column of the "centroids" input parameter) are
        # selected to calculate the width of fuzzy membership functions of that feature.

        deltas = np.zeros((self.num_clusters, self.dataset.training_data.shape[0]))  # shape: (num_clusters, num_features)

        for r in range(self.num_clusters):
            for j in range(self.dataset.training_data.shape[0]):
                # delta_rj = 1/n sum_{i=1}^{n} ( sqrt( \dfrac{-(x_ij - c_rj)^2}{2 * log(U_ir)}) ) )

                # where U_ir is the membership degree of the i-th data point to the r-th cluster
                # and x_ij is the j-th feature of the i-th data point, and c_rj is the j-th feature of the r-th cluster center.

                # TODO review, vectorized by LLM
                x_ij = self.dataset.training_data[j, :]  # shape: (n_samples,)
                c_rj = centroids[r, j]  # scalar
                U_ir = membership_degrees[:, r]  # shape: (n_samples,)

                # compute the delta_rj
                delta_rj = np.mean(np.sqrt(-(x_ij - c_rj) ** 2 / (2 * np.log(U_ir + 1e-10))))  # add small value to avoid log(0)
                deltas[r, j] = delta_rj

        self.mu = np.zeros((self.dataset.training_data.shape[1], self.num_clusters, self.dataset.training_data.shape[0]))  # shape: (n_samples, num_clusters, num_features)

        # compute mu for each data point
        for i in range(self.dataset.training_data.shape[1]):
            for r in range(self.num_clusters):
                # mu_irj = exp( -(x_ij - c_rj)^2 / (2 * delta_rj) )
                x_ij = self.dataset.training_data[:, i]  # shape: (n_features,)
                c_rj = centroids[r, :]  # shape: (n_features,)
                delta_rj = deltas[r, :]  # shape: (n_features,)

                self.mu[:, r, i] = np.exp(-( ((x_ij - c_rj) / (2 * delta_rj + 1e-10)) ** 2 ) )  # add small value to avoid division by zero

        assert self.mu.shape == (self.dataset.training_data.shape[1], self.num_clusters, self.dataset.training_data.shape[0]), "Mu shape mismatch."
        print ("OK: Membership functions computed successfully.")

    def predict(self, sample):
        # TODO
        return