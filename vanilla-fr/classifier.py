import numpy as np
import pdb
import logging

logging.basicConfig(filename="logs", level=logging.DEBUG)

class PCA:
    """ Class to abstract implementations of major classification algorithms. """
    def __init__(self):
        """ Do nothing in initialization"""
        self.cov = None
        self.eigen_vals = None
        self.eigen_vecs = None
        self.eigen_face_space = None
        self.selected_eigen_vecs = None
        self.test_projection = None

    def fit(self, A, n_components):
        """ Fits the PCA with the given feature vector and the number of components """
        # Compute the inner feature covariance for simplifying computation
        self.cov = np.matrix(A.T) * np.matrix(A)
        self.cov /= self.cov.shape[0]

        self.eigen_vals, self.eigen_vecs = np.linalg.eig(self.cov)
        self.eigen_vals = np.abs(self.eigen_vals)
        self.eigen_vecs = self.eigen_vecs.T
        self.sort_indices = self.eigen_vals.argsort()[::-1]
        self.eigen_vals = self.eigen_vals[self.sort_indices[0:n_components]]
        self.eigen_vecs = self.eigen_vecs[self.sort_indices[0:n_components]]
        self.eigen_vecs = self.eigen_vecs.T

        logging.debug("PCA: Printing shape of eigenvectors {0} and eigenvalues {1}".format(self.eigen_vecs.shape, self.eigen_vals.shape))
        logging.debug("PCA: Printing the eigenvalues, {0}".format(self.eigen_vals))
        logging.debug("PCA: Printing the eigenvectors, {0}".format(self.eigen_vecs))

        # TODO: Conduct slicing.
        self.selected_eigen_vecs = np.matrix(A) * np.matrix(self.eigen_vecs)
        norms = np.linalg.norm(self.selected_eigen_vecs, axis=0)
        self.selected_eigen_vecs /= norms
        self.eigen_face_space = np.matrix(self.selected_eigen_vecs.T) * np.matrix(A)
        return self.selected_eigen_vecs, self.eigen_face_space

    def transform(self, y):
        """ Transforms the given test data with the developed model"""
        self.test_projection = np.matrix(self.selected_eigen_vecs.T) * np.matrix(y).T
        return self.test_projection
        
