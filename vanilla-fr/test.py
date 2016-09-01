import cv2
import numpy as np
import dataPorter
import cv2
import pdb

NUM_IMGS         = 20
IMGS_PER_PERSON  = 2
NUM_PEOPLE       = NUM_IMGS / IMGS_PER_PERSON
dims             = (100, 100)

face_matrix, labels = dataPorter.import_custom_training_set(NUM_PEOPLE, IMGS_PER_PERSON)
pdb.set_trace()
lbp = cv2.createLBPHFaceRecognizer()
lbp.train(face_matrix, labels)
