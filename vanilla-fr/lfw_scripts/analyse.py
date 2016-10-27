#! /usr/bin/python

import sys
import matplotlib.pyplot as plt
import numpy as np

def getAlgoBounds():
    args = sys.argv[1:]

    for idx, f in enumerate(args):
        data = np.genfromtxt(f, delimiter=',')
        print data.shape
        colnames = data.dtype.names
        if idx == 0:
            consolidated = data
        else:
            consolidated = np.concatenate((consolidated, data))

    # Calculate stats
    print consolidated.shape

    all_max = np.max(consolidated[:, 0:-1], axis=1)
    print "ensemble", all_max, np.mean(all_max), np.std(all_max)

    print "svm", np.mean(consolidated[:,-1]), np.std(consolidated[:,-1])

    print np.sum(all_max > consolidated[:,-1])

if __name__ == '__main__':
    getAlgoBounds()
