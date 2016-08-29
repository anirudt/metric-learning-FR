#!/usr/bin/python
import numpy as np
from time import time
from sklearn.decomposition import PCA
import cv2
import pdb
import csv
from classifier import PCA, LDA

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
    #pdb.set_trace()
    with open("eigen_vals.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(eigen_vals.tolist())
    with open("eigen_vecs.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(eigen_vecs.tolist())

def nearest_neighbour(projs, test_proj):
    distances = np.zeros((projs.shape[1], 1))
    # Alternatively, you could also try the following line.
    # distances = np.linalg.norm(projs - test_proj, axis=1)
    for col in range(projs.shape[1]):
        distances[col] = np.linalg.norm((projs[:, col] - test_proj))
    print "Closest neighbour is {0}".format(distances)
    return np.argmin(distances)

def import_training_set():
    """ Get the face matrix here. """
    face_matrix = np.array([ np.resize(np.array(cv2.imread("data/ROLL ("+str(num)+")/Regular/W ("+str(tilt_idx)+").jpg", cv2.IMREAD_GRAYSCALE), dtype='float64'), dims).ravel() for num in range(1, NUM_IMGS+1) for tilt_idx in range(2,3)], dtype='float64')
    labels = [num for num in range(1, NUM_IMGS+1) for i in range(IMGS_PER_PERSON)]
    print "labels = ", labels
    face_matrix = face_matrix.T
    print "The dimensions of the face matrix are: {0}".format(face_matrix.shape)

    mean = np.mean(face_matrix, axis=1)
    print "The dimensions of the mean face are: {0}".format(mean.shape)

    # TODO: Make a way to print / imwrite this average image
    for col in range(face_matrix.shape[1]):
        face_matrix[:, col] = face_matrix[:, col] - mean

    return face_matrix, mean, labels

def train(tilt_idx):
    """ Get data, train, get the Eigenvalues and store them."""
    face_matrix, mean_face, labels = import_training_set()

    print face_matrix.shape
    """
    cov = np.matrix(face_matrix.T) * np.matrix(face_matrix)
    cov /= NUM_IMGS
    print cov.shape
    """
    pca = PCA()
    selected_eigen_vecs_pca, eigen_face_space = pca.fit(face_matrix, NUM_IMGS)

    # TODO: Return something
    lda = LDA()
    lda_projection, selected_eigen_vecs_lda = lda.fit(eigen_face_space, NUM_PEOPLE, NUM_IMGS)

    return pca, lda, lda_projection, mean_face, selected_eigen_vecs_pca, selected_eigen_vecs_lda

def test(tilt_idx, lda, lda_projection, mean_face, pca, selected_eigen_vecs_pca, selected_eigen_vecs_lda):
    """ Acquire a new image and get the data. """
    test_image = np.resize(np.matrix(cv2.imread("data/ROLL (8)/Regular/W ("+str(tilt_idx-1)+").jpg", cv2.IMREAD_GRAYSCALE), dtype='float64'), dims).ravel()
    test_image = test_image.T

    test_image -= mean_face

    # PCA-Transform the image
    #pdb.set_trace()
    print selected_eigen_vecs_pca.T.shape, test_image.shape
    pca_test_proj = pca.transform(test_image)

    # LDA-Transform the PCA subspace
    lda_test_proj = lda.transform(pca_test_proj)

    # Trying out the nearest neighbour for classification
    detected_idx = nearest_neighbour(lda_projection, lda_test_proj)

    print "Detected face is of serial no. {0}".format(detected_idx+1)
    ##pdb.set_trace()

def multi_runner():
    """
    Runs the training and test for all the different tilted faces. Returns a list of lists of eigenvalues and eigenvectors.
    """
    eigenvals, eigenvecs = [], []
    for tilt_idx in range(3, 8):
        pca, lda, lda_projection, mean_face, selected_eigen_vecs_pca, selected_eigen_vecs_lda = train(tilt_idx)
        test(tilt_idx, lda, lda_projection, mean_face, pca, selected_eigen_vecs_pca, selected_eigen_vecs_lda)
        """
        eigenvals.append(tmp_eigen_vals)
        eigenvecs.append(tmp_eigen_vecs)
    eigenvecs = np.array(eigenvecs)
    eigenvals = np.array(eigenvals)
    print eigenvals, eigenvecs
    return eigenvals, eigenvecs
    """
    return 0


if __name__ == "__main__":
    multi_runner()
    #eigen_logger(eigen_vals, eigen_vecs)
    print "We are done."
