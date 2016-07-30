#!/usr/bin/python
import numpy as np
from time import time
from sklearn.decomposition import PCA
import cv2

""" Script for EigenFace Method of Face Recognition """


NUM_IMGS     = 10
dims         = (100, 100)

def train():
    """ Get data, train, get the Eigenvalues and store them."""
    face_matrix = np.array([ np.resize(np.array(cv2.imread("data/ROLL ("+str(num)+")/Regular/W (2).jpg", cv2.IMREAD_GRAYSCALE)), dims).ravel() for num in range(1, NUM_IMGS+1) ])
    print "The dimensions of the face matrix are: {0}".format(face_matrix.shape)

    mean = np.mean(face_matrix, axis=0)
    print mean.shape
    face_matrix -= mean

    pca = PCA(n_components=8)
    pca.fit(face_matrix.T)

    print "Displaying the EigenValues selected: {0}".format(pca.explained_variance_ratio_)
    eigen_vals = pca.explained_variance_ratio_
    eigen_vecs = pca.components_

    features = np.dot(eigen_vecs, face_matrix)
    print features.shape
    return features, mean

def test(features, mean):
    """ Acquire a new image and get the data. """
    test_image = np.resize(np.array(cv2.imread("", cv2.IMREAD_GRAYSCALE)), dims).ravel()

    test_image -= mean

    test_face = np.dot(features, test_image)

    # TODO: Compute the features for all other people and then conduct a nearest neighbour.


if __name__ == "__main__":
    features, mean = train()
    test(features, mean)
    print "We are done."
