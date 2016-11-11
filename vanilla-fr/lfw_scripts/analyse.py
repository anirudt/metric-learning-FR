#! /usr/bin/python

import sys
import matplotlib.pyplot as plt
import numpy as np

def getAlgoBounds():
    args = sys.argv[1:]

    for idx, f in enumerate(args[:-2]):
        data = np.genfromtxt(f, delimiter=',')
        print data.shape
        colnames = data.dtype.names
        all_max = np.max(data, axis=1)
        print f, np.mean(all_max), np.std(all_max)

        print data.shape

    lda_dat = np.genfromtxt(args[-1], delimiter=',')
    svm_dat = np.genfromtxt(args[-2], delimiter=',')
    print "svm ", np.mean(svm_dat), np.std(svm_dat) 
    print "lda ", np.mean(lda_dat), np.std(lda_dat) 

if __name__ == '__main__':
    getAlgoBounds()
