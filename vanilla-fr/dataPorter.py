import cv2
import numpy as np

def import_custom_training_set(NUM_PEOPLE, IMGS_PER_PERSON):
    """ This function provides mean corrected images from the custom IIT KGP dataset
        and provides labels. """

    face_matrix = np.array([ np.resize(np.array(cv2.imread("data/ROLL ("+str(num)+")/Regular/W ("+str(tilt_idx)+").jpg", cv2.IMREAD_GRAYSCALE), dtype='float64'), (100, 100)).ravel() for num in range(1, NUM_PEOPLE+1) for tilt_idx in range(2,4)], dtype='float64')
    labels = [num for num in range(1, NUM_PEOPLE+1) for i in range(IMGS_PER_PERSON)]
    print "labels = ", labels
    face_matrix = face_matrix.T
    print "The dimensions of the face matrix are: {0}".format(face_matrix.shape)

    mean = np.mean(face_matrix, axis=1)
    print "The dimensions of the mean face are: {0}".format(mean.shape)

    # TODO: Make a way to print / imwrite this average image
    for col in range(face_matrix.shape[1]):
        face_matrix[:, col] = face_matrix[:, col] - mean

    return face_matrix, mean, labels

def import_custom_testing_set(tilt_idx):
   return np.resize(np.matrix(cv2.imread("data/ROLL (8)/Regular/W ("+str(tilt_idx-1)+").jpg", cv2.IMREAD_GRAYSCALE), dtype='float64'), (100, 100)).ravel()


def import_att_training_set(NUM_PEOPLE_ATT, IMGS_PER_PERSON_ATT):
    """ This function provides mean corrected images from the AT&T face dataset.
    Credits: AT&T Laboratories Cambridge."""

    face_matrix = np.array([ np.array(cv2.imread("~/Datasets/orl_faces/s"+str(num)+"/"+str(idx)+".pgm", cv2.IMREAD_GRAYSCALE), dtype=np.float64).ravel() for num in range(1, NUM_PEOPLE_ATT+1) for idx in range(1, 8) ], dtype=np.float64)
    labels = [num for num in range(1, NUM_PEOPLE_ATT+1) for _ in range(IMGS_PER_PERSON_ATT)]
    face_matrix = face_matrix.T
    mean = np.mean(face_matrix, axis = 1)

    for col in range(face_matrix.shape[1]):
        face_matrix[:, col] = face_matrix[:, col] - mean

    return face_matrix, mean, labels
