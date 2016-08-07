#!/usr/bin/python
import numpy as np
from time import time
from sklearn.decomposition import PCA
import cv2
import pdb
import csv

""" Script for EigenFace Method of Face Recognition """

"""
TODO:
Some helper functions need to be written for better analysis.
1. An automated function for displaying the set of images in a subplot, followed by the graphify function.
2. Compute the eigenvectors and analyse them -- compare, visualize and save them.
"""


NUM_IMGS     = 40
dims         = (100, 100)

"""
Some helper functions.
"""
def display_imgs(face_matrix):
    """
    TODO: Reshape, quantise and display the images in a subplot fashion.
    http://stackoverflow.com/questions/17111525/how-to-show-multiple-images-in-one-figure
    """

def eigen_logger(eigen_vals, eigen_vecs):
    """
    TODO: Identify the class of photos and place the eigenvalues and eigenvectors in a fairly understandable
    data structure. Also, try to plot the eigenvalue projections across all data fitted hitherto.
    Things to save: del A, del (eig_vals A), del (eig_vecs A)
    """
    pdb.set_trace()
    with open("eigen_vals.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(eigen_vals.tolist())
    with open("eigen_vecs.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(eigen_vecs.tolist())

def train(tilt_idx):
    """ Get data, train, get the Eigenvalues and store them."""
    face_matrix = np.array([ np.resize(np.array(cv2.imread("data/ROLL ("+str(num)+")/Regular/W ("+str(tilt_idx)+").jpg", cv2.IMREAD_GRAYSCALE), dtype='float64'), dims).ravel() for num in range(1, NUM_IMGS+1) ], dtype='float64')
    print "The dimensions of the face matrix are: {0}".format(face_matrix.shape)

    mean = np.mean(face_matrix, axis=0)
    print mean.shape
    face_matrix -= mean

    print face_matrix.shape
    cov = np.matrix(face_matrix) * np.matrix(face_matrix.T)
    cov /= NUM_IMGS
    print cov.shape
    eigen_vals, eigen_vecs = np.linalg.eig(cov)
    eigen_vals = np.abs(eigen_vals)
    sort_indices = eigen_vals.argsort()[::-1]
    eigen_vals = eigen_vals[sort_indices[0:10]]
    eigen_vecs = eigen_vecs[sort_indices[0:10]]

    print eigen_vecs.shape, eigen_vals.shape
    print "Haha"
    print eigen_vecs, eigen_vals

    # TODO: Conduct slicing.

    eigen_vecs = np.matrix(eigen_vecs) * np.matrix(face_matrix)
    norms = np.linalg.norm(eigen_vecs, axis=0)
    eigen_vecs /= norms

    weights = np.matrix(face_matrix) * np.matrix(eigen_vecs.T)
    print weights.shape
    #pdb.set_trace()
    return eigen_vals, eigen_vecs, weights, mean

def test(tilt_idx, eigen_vecs, weights, mean):
    """ Acquire a new image and get the data. """
    test_image = np.resize(np.array(cv2.imread("data/ROLL (9)/Regular/W ("+str(tilt_idx-1)+").jpg", cv2.IMREAD_GRAYSCALE), dtype='float64'), dims).ravel()

    test_image -= mean

    # TODO: Compute the features for all other people and then conduct a nearest neighbour.
    print eigen_vecs.shape, test_image.shape
    proj_weights = np.dot(eigen_vecs, test_image)

    
    similarity_feat = np.linalg.norm(weights - proj_weights, axis=1)
    print similarity_feat
    detected_idx = np.argmin(similarity_feat)
    print "Detected face is of serial no. {0}".format(detected_idx+1)
    ##pdb.set_trace()

def multi_runner():
    """
    Runs the training and test for all the different tilted faces. Returns a list of lists of eigenvalues and eigenvectors.
    """
    eigenvals, eigenvecs = [], []
    for tilt_idx in range(3, 8):
        tmp_eigen_vals, tmp_eigen_vecs, tmp_weights, tmp_mean = train(tilt_idx)
        test(tilt_idx, tmp_eigen_vecs, tmp_weights, tmp_mean)
        eigenvals.append(tmp_eigen_vals)
        eigenvecs.append(tmp_eigen_vecs)
    eigenvecs = np.array(eigenvecs)
    eigenvals = np.array(eigenvals)
    print eigenvals, eigenvecs
    return eigenvals, eigenvecs


if __name__ == "__main__":
    eigen_vals, eigen_vecs = multi_runner()
    eigen_logger(eigen_vals, eigen_vecs)
    print "We are done."
