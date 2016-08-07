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


NUM_IMGS         = 10
IMGS_PER_PERSON  = 2
NUM_PEOPLE       = NUM_IMGS / IMGS_PER_PERSON
dims             = (100, 100)

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
    with open("eigen_vals.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(eigen_vals)
    with open("eigen_vecs.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(eigen_vecs)

def import_training_set():
    """ Get the face matrix here. """
    face_matrix = np.array([ np.resize(np.array(cv2.imread("data/ROLL ("+str(num)+")/Regular/W ("+str(tilt_idx)+").jpg", cv2.IMREAD_GRAYSCALE), dtype='float64'), dims).ravel() for num in range(1, NUM_IMGS+1) ], dtype='float64')
    face_matrix = face_matrix.T
    print "The dimensions of the face matrix are: {0}".format(face_matrix.shape)

    mean = np.mean(face_matrix, axis=1)
    print "The dimensions of the mean face are: {0}".format(mean)

    # TODO: Make a way to print / imwrite this average image
    face_matrix -= mean

    return face_matrix

def lda(eigen_face):
    """ Computes the LDA in the specified subspace provided. """
    class_means = np.zeros(eigen_face.shape[0], NUM_PEOPLE)
    within_class_cov = np.zeros(eigen_face.shape[0], eigen_face.shape[0])
    between_class_cov = np.zeros(eigen_face.shape[0], eigen_face.shape[0])
    for i in range(NUM_PEOPLE):
        class_means[:,i] = np.mean(eigen_face[:,i*IMGS_PER_PERSON:i*IMGS_PER_PERSON+IMGS_PER_PERSON], axis=1)

    overall_mean = np.mean(class_means, axis=1)
    for i in range(NUM_PEOPLE):
        class_mean_i = class_means[:, i]
        class_mat = eigen_face[:, i*IMGS_PER_PERSON:(i+1)*IMGS_PER_PERSON] - class_means
        within_class_cov += np.matrix(class_mat) * np.matrix(class_mat.T)

        between_class_cov += np.matrix(class_mean_i - overall_mean) * np.matrix((class_mean_i - overall_mean).T)

    print "Dimensions of within class scatter matrix are {0}".format(within_class_cov.shape)
    print "Dimensions of between class scatter matrix are {0}".format(between_class_cov.shape)

    eigen_vals, eigen_vecs = np.linalg.eig(np.matrix(np.linalg.inv(within_class_cov)) * np.matrix(between_class_cov))

    # TODO: Determine the dimensions of the eigenvector and choose the appropriate projection.



def pca(X, A):
    """ Computes the PCA of:
        X: covariance matrix
        A: difference matrix
    """
    eigen_vals, eigen_vecs = np.linalg.eig(cov)
    eigen_vals = np.abs(eigen_vals)
    sort_indices = eigen_vals.argsort()[::-1]
    eigen_vals = eigen_vals[sort_indices]
    eigen_vecs = eigen_vecs[sort_indices]
    print eigen_vecs.shape, eigen_vals.shape
    print eigen_vecs, eigen_vals
    # TODO: Conduct slicing.
    selected_eigen_vecs = np.matrix(A) * np.matrix(eigen_vecs)
    eigen_face_space = np.matrix(selected_eigen_vecs.T) * np.matrix(A)
    return eigen_face_space



def train(tilt_idx):
    """ Get data, train, get the Eigenvalues and store them."""
    face_matrix = import_training_set()

    print face_matrix.shape
    cov = np.matrix(face_matrix.T) * np.matrix(face_matrix)
    cov /= NUM_IMGS
    print cov.shape

    eigen_face_space = pca(face_matrix, cov)

    eigen_vecs = np.matrix(eigen_vecs) * np.matrix(face_matrix)
    norms = np.linalg.norm(eigen_vecs, axis=0)
    eigen_vecs /= norms

    weights = np.matrix(face_matrix) * np.matrix(eigen_vecs.T)
    print weights.shape
    #pdb.set_trace()
    return eigen_vals, eigen_vecs, weights, mean

def test(tilt_idx, eigen_vecs, weights, mean):
    """ Acquire a new image and get the data. """
    test_image = np.resize(np.array(cv2.imread("data/ROLL (9)/Regular/W ("+str(tilt_idx)+").jpg", cv2.IMREAD_GRAYSCALE), dtype='float64'), dims).ravel()

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
    for tilt in range(5):
        tmp_eigen_vals, tmp_eigen_vecs, tmp_weights, tmp_mean = train(tilt_idx)
        test(tilt_idx, tmp_eigen_vecs, tmp_weights, tmp_mean)
        eigenvals.append(tmp_eigen_vals)
        eigenvecs.append(tmp_eigen_vecs)
    return eigen_vals, eigen_vecs


if __name__ == "__main__":
    eigen_vals, eigen_vecs = multi_runner()
    eigen_logger(eigen_vals, eigen_vecs)
    print "We are done."
