#!/usr/bin/python
import numpy as np
from time import time
from sklearn.decomposition import PCA
import cv2
import pdb
import csv
import classifier
import dataPorter

""" Script for EigenFace Method of Face Recognition """

"""
TODO:
Some helper functions need to be written for better analysis.
1. An automated function for displaying the set of images in a subplot, followed by the graphify function.
2. Compute the eigenvectors and analyse them -- compare, visualize and save them.
"""


NUM_IMGS         = 40
IMGS_PER_PERSON  = 4
NUM_PEOPLE       = NUM_IMGS / IMGS_PER_PERSON

"""
Acceptable tuples include:
KGP_DB = (20, 2, (100, 100))
ATT_DB = (40, 4, (92, 112))
"""


str2classifier = {"pca": classifier.PCA(),
                  "lda": classifier.LDA(),
                  "lbp": classifier.LBP()}

str2traindatabase = { "KGP": dataPorter.import_custom_training_set,
                      "ATT": dataPorter.import_att_training_set}

str2testdatabase = { "KGP": dataPorter.import_custom_testing_set,
                     "ATT": dataPorter.import_att_testing_set}

str2ravelling = {"pca": 'ravel',
                 "pcalda": 'ravel',
                 "lbp": 'unravel'}

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

""" Hybrid Classifiers Definition """
def pca_lda(face_matrix, pca, lda, labels):
    print "shape ", face_matrix.shape
    selected_eigen_vecs_pca, eigen_face_space = pca.fit(face_matrix, NUM_IMGS)

    # TODO: Return something
    selected_eigen_vecs_lda, lda_projection = lda.fit(eigen_face_space, labels)

    return [2, pca, lda, selected_eigen_vecs_pca, selected_eigen_vecs_lda, lda_projection]

def train(classifier_str, database):
    """ Get data, train, get the Eigenvalues and store them."""
    face_matrix, labels = str2traindatabase[database](NUM_PEOPLE, IMGS_PER_PERSON, str2ravelling[classifier_str])

    print face_matrix.shape

    if classifier_str in str2classifier:
        model = str2classifier[classifier_str]
        trained_bundle = model.fit(face_matrix, labels)
        return [1, model, trained_bundle]
        # TODO: Handle this trained_bundle in a standard way
    else:
        if classifier_str == "pcalda":
            pca = str2classifier[classifier_str[0:3]]
            lda = str2classifier[classifier_str[3:]]
            trained_bundle = pca_lda(face_matrix, pca, lda, labels)
            return trained_bundle
        # Add more hybrid varieties here. If they are standalone 
        # classifier_strs, make a class out of it.

def test(person, tilt_idx, trained_bundle, classifier_str, database):
    """ Acquire a new image and get the data. """
    test_image = str2testdatabase[database](person, tilt_idx, str2ravelling[classifier_str])
    
    c = 0
    if trained_bundle[0] == 1:
        model = trained_bundle[1]
        test_proj, space = model.transform(test_image)

        # FIXME: Remove this once the LBP has an option to give its space key
        if hasattr(space, 'shape'):
            detected_idx = classifier.nearest_neighbour(space, test_proj)
            print detected_idx
            detected_idx = int((detected_idx/IMGS_PER_PERSON)+1)
            print "Detected argmin: ", detected_idx
        else:
            detected_idx = test_proj
            print space
        print detected_idx
        #pdb.set_trace()

    elif trained_bundle[0] == 2:
        model1 = trained_bundle[1]
        test_proj1, _ = model1.transform(test_image)

        model2 = trained_bundle[2]
        test_proj2, space = model2.transform(test_proj1)

        detected_idx = classifier.nearest_neighbour(space, test_proj2)
        detected_idx = int((detected_idx-1)/IMGS_PER_PERSON)+1
        print detected_idx

    # Log the accuracy metrics.
    return [person, classifier_str, detected_idx]
    
    #print "Detected face is of serial no. {0}".format((detected_idx+2)/IMGS_PER_PERSON)

def multi_runner(classifier_str, database):
    """
    Runs the training and test for all the different tilted faces. Returns a list of lists of eigenvalues and eigenvectors.
    """
    g = open("results.csv", 'ab')
    a = open("accuracy.csv", "wb")
    wr = csv.writer(g)
    acc = csv.writer(a)
    eigenvals, eigenvecs = [], []
    start, end = 9, 11
    trained_bundle = train(classifier_str, database)
    for person in range(1,NUM_PEOPLE+1):
        c = 0
        for tilt_idx in range(start, end):
            val = test(person, tilt_idx, trained_bundle, classifier_str, database)
            print person
            wr.writerow(val)
            if val[2] == val[0]:
                c+=1
        accuracy = c*100.0/(end-start)
        tmp = [classifier_str, database, person, accuracy]
        acc.writerow(tmp)


    return 0


def main(classifier_str, database):
    global NUM_PEOPLE, NUM_IMGS, IMGS_PER_PERSON, dims
    if database == "KGP":
        (NUM_PEOPLE, NUM_IMGS, IMGS_PER_PERSON, dims) = (10, 20, 2, (100, 100))
    elif database == "ATT":
        (NUM_PEOPLE, NUM_IMGS, IMGS_PER_PERSON, dims) = (10, 80, 8, (92, 112))
    print "sizes are ",(NUM_IMGS, IMGS_PER_PERSON, dims) 

    multi_runner(classifier_str, database)
    return 1

if __name__ == '__main__':
    main("pca", "ATT")
