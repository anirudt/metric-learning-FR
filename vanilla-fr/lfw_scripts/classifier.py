import numpy as np
import pdb, os, scipy
import cv2
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from skimage.feature import local_binary_pattern
from modshogun import RealFeatures, MulticlassLabels
from modshogun import LMNN as shogun_LMNN
import matplotlib.pyplot as plt
from metric_learn import ITML_Supervised, SDML_Supervised, LSML_Supervised
from sklearn.neighbors.nearest_centroid import NearestCentroid
import operator
from threading import Thread

logging.basicConfig(filename="logs", level=logging.DEBUG)

# Helper Functions
def nearest_neighbour(projs, test_proj):
    distances = np.zeros((projs.shape[1], 1))
    # Alternatively, you could also try the following line.
    # distances = np.linalg.norm(projs - test_proj, axis=1)
    for col in range(projs.shape[1]):
        distances[col] = np.linalg.norm((projs[:, col] - test_proj))
    print "Neighbours at {0}".format(distances)
    print "Closest neighbour: {0}".format(np.argmin(distances))
    return np.argmin(distances)

def sk_nearest_neighbour_proba(centroids, X_test_single):
    """ Wrapper over sklearn's nearest neighbor. """
    # TODO: Compute a softmax over the distances between each of them
    probs = np.zeros(centroids.shape[0])
    for cent in xrange(centroids.shape[0]):
        probs[cent] = np.exp(-1*np.linalg.norm(centroids[cent,:] - X_test_single))
    
    probs /= np.sum(probs)
    return probs

def sk_nearest_neighbour(X_train, y_train, X_test, y_test):
    """ Wrapper over sklearn's nearest neighbor. """
    clf = NearestCentroid()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    c = np.sum(y_pred == y_test)
    accuracy = c * 100.0 / len(y_test)
    return accuracy, y_pred

class PCA:
    """ Class to abstract implementations of PCA. """
    def __init__(self):
        """ Do nothing in initialization"""
        self.cov = None
        self.eigen_vals = None
        self.eigen_vecs = None
        self.A_space = None
        self.selected_eigen_vecs = None
        self.test_projection = None
        self.mean = []

    def fit(self, A, labels):
        """ Fits the PCA with the given feature vector and the number of components """
        A = A.T
        
        num_people = np.max(labels)
        num_points = np.shape(labels)[0]/num_people
        self.mean = np.mean(A, axis=1)
        print "The dimensions of the mean face are: {0}".format(self.mean.shape)

        # TODO: Make a way to print / imwrite this average image
        for col in range(A.shape[1]):
            A[:, col] = A[:, col] - self.mean

        n_components = int(A.shape[1])
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
        # Need to return values to be used in cascaded classifier systems
        # (_,feats) = self.A_space.shape
        #self.new_space = np.zeros((num_people, num_points, feats))
        #for i in xrange(num_people):
        #  for j in xrange(num_points):
        #    self.new_space[i,j,:] = self.A_space[i*num_points+j,:]
            
        # TODO: Change the following return to the reshaped vals
        self.A = A
        return self.selected_eigen_vecs, self.A_space

    def transform(self, y):
        """ Transforms the given test data with the developed model"""
        y = y.T
        for col in range(y.shape[1]):
            y[:, col] = y[:, col] - self.mean
        self.test_projection = np.matrix(self.selected_eigen_vecs.T) * np.matrix(y)
        self.A_space = np.matrix(self.selected_eigen_vecs.T) * np.matrix(self.A)
        return self.test_projection, self.A_space
        
class LDA:
    """ Class to abstract implementation of LDA"""
    def __init__(self):
        """ Computes the LDA in the specified subspace provided. """
        self.eigen_vals = None
        self.eigen_vecs = None
        self.lda_projection = None
        self.test_proj = None
    
    def fit(self, A, labels):
        """ Check if you really need to specify the n_classes as another argument"""
        n_classes = np.max(labels)
        n_components = labels.shape[0]
        num_imgs = A.shape[1]
        imgs_per_person = num_imgs / n_classes
        class_means = np.zeros((A.shape[0], n_classes))
        within_class_cov = np.zeros((A.shape[0], A.shape[0]))
        between_class_cov = np.zeros((A.shape[0], A.shape[0]))
        for i in range(n_classes):
            class_means[:,i] = np.mean(A[:,i*imgs_per_person:i*imgs_per_person+imgs_per_person], axis=1).ravel()

        overall_mean = np.mean(class_means, axis=1).ravel()
        for i in range(n_classes):
            class_mean_i = class_means[:, i]
            class_mat = np.zeros((A.shape[0], imgs_per_person))
            #class_mat = np.matrix(A[:, i*imgs_per_person:(i+1)*imgs_per_person]) - class_mean_i
            for j in range(i*imgs_per_person, (i+1)*imgs_per_person):
                class_mat[:, j-i*imgs_per_person] = A[:, j].ravel() - class_mean_i
            within_class_cov += np.matrix(class_mat) * np.matrix(class_mat.T)

            diff_mat = (class_mean_i - overall_mean).reshape((A.shape[0], 1))
            between_class_cov += np.matrix(diff_mat) * np.matrix(diff_mat.T)

        within_class_cov /= 1.0*n_classes
        between_class_cov /= 1.0*n_classes
        logging.debug("Dimensions of within class scatter matrix are {0}".format(within_class_cov.shape))
        logging.debug("Dimensions of between class scatter matrix are {0}".format(between_class_cov.shape))

        self.eigen_vals, self.eigen_vecs = np.linalg.eig(np.matrix(np.linalg.inv(within_class_cov)) * np.matrix(between_class_cov))
        #pdb.set_trace()
        # TODO: Select only some components based on some selection theory
        sort_indices = np.abs(self.eigen_vals).argsort()[::-1]
        self.eigen_vecs = self.eigen_vecs.T
        #pdb.set_trace()

        # TODO: In case you wish to remove certain LDA components, do them here.
        self.eigen_vals = self.eigen_vals[sort_indices]
        self.eigen_vecs = self.eigen_vecs[sort_indices]
        self.eigen_vecs = self.eigen_vecs.T
        print self.eigen_vecs.T.shape, A.shape

        self.lda_projection = np.matrix(self.eigen_vecs.T) * np.matrix(A)

        logging.debug("The dimensions of the LDA projection are {0}".format(self.lda_projection.shape))

        # Need to return the values to be used in cascaded classifier systems
        return self.eigen_vecs, self.lda_projection

    def transform(self, y):
        """ Function to apply given test data on the created LDA model """
        print "Sizes are ", self.eigen_vecs.T.shape, y.shape
        self.test_proj = np.matrix(self.eigen_vecs.T) * np.matrix(y)
        pdb.set_trace()
        return self.test_proj, self.lda_projection

class LBP:
    """Class to abstract implementation of LBP"""
    def __init__(self):
        self.radius = 2
        self.n_points = 16
        self.model = cv2.createLBPHFaceRecognizer(self.radius,
                self.n_points)

    def fit(self, features, labels, kparts=2):
        nsamples = features.shape[0]
        subset_size = nsamples/kparts
        for j in xrange(kparts):
            if j == 0:
              self.model.train(features[0:subset_size][:], labels[0:subset_size])
            else:
              self.model.update(features[j*subset_size:(j+1)*subset_size][:], labels[j*subset_size:(j+1)*subset_size])

    def transform(self, y_test):
        """ Uses a nearest neighbour to find the class label """
        return self.model.predict(y_test)

    def save(self, filename):
        return self.model.save(filename)

    def load(self, filename):
        return self.model.load(filename)

    def update(self, features, labels):
        return self.model.update(features, labels)

class LMNN:
    """Class to abstract implementation of LMNN."""
    def __init__(self, k=3, min_iter=50, max_iter=1000, learn_rate=1e-7,
                 regularization=0.50, convergence_tol=0.001, use_pca=False):
        self.k = k
        self.eigenvecs = None
        self.space = None
        self.use_pca = use_pca
        self.metric_model = None
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.convergence_tol = convergence_tol
        self.regularization = regularization
        print self.k, self.min_iter, self.max_iter, self.learn_rate, self.regularization, self.convergence_tol
        
    def fit(self, feats, labels):
        self.X_tr = feats
        self.y_train = labels

        feats = feats.astype(np.float64)
        feat = RealFeatures(feats.T)
        self.metric_model = shogun_LMNN(feat, MulticlassLabels(labels.astype(np.float64)), self.k)
        self.metric_model.set_maxiter(self.max_iter)
        self.metric_model.set_regularization(self.regularization)
        self.metric_model.set_obj_threshold(self.convergence_tol)
        self.metric_model.set_stepsize(self.learn_rate)

        self.metric_model.train()

        stats = self.metric_model.get_statistics()
        #pdb.set_trace()
        #plt.plot(stats.obj.get())
        #plt.grid(True)
        #plt.show()
        self.linear_transform = self.metric_model.get_linear_transform()

        #self.projected_data = np.dot(self.linear_transform, feats.T)
        #norms = np.linalg.norm(self.projected_data, axis=0)
        #self.projected_data /= norms
        # Fit the data with PCA first.
        # pdb.set_trace()
        return self

    def transform(self, y):
        # On the projection in the resultant space, apply LMNN.
        lk = np.dot(self.linear_transform,y.T)
        #lk = lk/np.linalg.norm(lk, axis=0)
        
        return lk.T

    def predict_proba(self, X_te):
        """Predicts the probabilities of each of the test samples"""
        test_samples = X_te.shape[0]
        self.X_tr = self.transform(self.X_tr)
        clf = NearestCentroid()
        clf.fit(self.X_tr, self.y_train)
        centroids = clf.centroids_
        probabilities = np.zeros((test_samples, centroids.shape[0]))
        for sample in xrange(test_samples):
            probabilities[sample] = sk_nearest_neighbour_proba(centroids, X_te[sample, :])
        return probabilities

class ITML:
    def __init__(self, num_constraints=200):
        self.metric_model = ITML_Supervised(num_constraints)

    def fit(self, features, labels):
        """Fits the model to the prescribed data."""
        return self.metric_model.fit(features, labels)

    def transform(self, y):
        """Transforms the test data according to the model"""
        return self.metric_model.transform(y)

    def predict_proba(self, X_te):
        """Predicts the probabilities of each of the test samples"""
        test_samples = X_te.shape[0]
        self.X_tr = self.transform(self.X_tr)
        clf = NearestCentroid()
        clf.fit(self.X_tr, self.y_train)
        centroids = clf.centroids_
        probabilities = np.zeros((test_samples, centroids.shape[0]))
        for sample in xrange(test_samples):
            probabilities[sample] = sk_nearest_neighbour_proba(centroids, X_te[sample, :])
        return probabilities

class LSML:
    def __init__(self):
        self.metric_model = LSML_Supervised(num_constraints=200)
        self.X_tr = None
        self.y_train = None
        self.X_te = None

    def fit(self, X_tr, y_train):
        """Fits the model to the prescribed data."""
        self.X_tr = X_tr
        self.y_train = y_train
        return self.metric_model.fit(X_tr, y_train)

    def transform(self, X):
        """Transforms the test data according to the model"""
        return self.metric_model.transform(X)

    def predict_proba(self, X_te):
        """Predicts the probabilities of each of the test samples"""
        test_samples = X_te.shape[0]
        self.X_tr = self.transform(self.X_tr)
        clf = NearestCentroid()
        clf.fit(self.X_tr, self.y_train)
        centroids = clf.centroids_
        probabilities = np.zeros((test_samples, centroids.shape[0]))
        for sample in xrange(test_samples):
            probabilities[sample] = sk_nearest_neighbour_proba(centroids, X_te[sample, :])
        return probabilities

class RCA:
    def __init__(self):
        self.metric_model = SDML_Supervised(num_constraints=200)
        self.X_tr = None
        self.y_train = None
        self.X_te = None

    def fit(self, X_tr, y_train):
        """Fits the model to the prescribed data."""
        self.X_tr = X_tr
        self.y_train = y_train
        return self.metric_model.fit(X_tr, y_train)

    def transform(self, X):
        """Transforms the test data according to the model"""
        return self.metric_model.transform(X)

    def predict_proba(self, X_te):
        """Predicts the probabilities of each of the test samples"""
        test_samples = X_te.shape[0]
        self.X_tr = self.transform(self.X_tr)
        clf = NearestCentroid()
        clf.fit(self.X_tr, self.y_train)
        centroids = clf.centroids_
        probabilities = np.zeros((test_samples, centroids.shape[0]))
        for sample in xrange(test_samples):
            probabilities[sample] = sk_nearest_neighbour_proba(centroids, X_te[sample, :])
        return probabilities

class NCA:
    def __init__(self):
        self.metric_model = SDML_Supervised(num_constraints=200)
        self.X_tr = None
        self.y_train = None
        self.X_te = None

    def fit(self, X_tr, y_train):
        """Fits the model to the prescribed data."""
        self.X_tr = X_tr
        self.y_train = y_train
        return self.metric_model.fit(X_tr, y_train)

    def transform(self, X):
        """Transforms the test data according to the model"""
        return self.metric_model.transform(X)

    def predict_proba(self, X_te):
        """Predicts the probabilities of each of the test samples"""
        test_samples = X_te.shape[0]
        self.X_tr = self.transform(self.X_tr)
        clf = NearestCentroid()
        clf.fit(self.X_tr, self.y_train)
        centroids = clf.centroids_
        probabilities = np.zeros((test_samples, centroids.shape[0]))
        for sample in xrange(test_samples):
            probabilities[sample] = sk_nearest_neighbour_proba(centroids, X_te[sample, :])
        return probabilities

class SDML:
    def __init__(self):
        self.metric_model = SDML_Supervised(num_constraints=200)
        self.X_tr = None
        self.y_train = None
        self.X_te = None

    def fit(self, X_tr, y_train):
        """Fits the model to the prescribed data."""
        self.X_tr = X_tr
        self.y_train = y_train
        return self.metric_model.fit(X_tr, y_train)

    def transform(self, X):
        """Transforms the test data according to the model"""
        return self.metric_model.transform(X)

    def predict_proba(self, X_te):
        """Predicts the probabilities of each of the test samples"""
        test_samples = X_te.shape[0]
        self.X_tr = self.transform(self.X_tr)
        clf = NearestCentroid()
        clf.fit(self.X_tr, self.y_train)
        centroids = clf.centroids_
        probabilities = np.zeros((test_samples, centroids.shape[0]))
        for sample in xrange(test_samples):
            probabilities[sample] = sk_nearest_neighbour_proba(centroids, X_te[sample, :])
        return probabilities

class LDML:
    def __init__(self):
        self.metric_model = None
        self.X_tr = None
        self.y_train = None
        self.X_te = None
        self.L = None

    def fit(self, X_tr, y_train):
        """ Fits the LDML model 
        Steps include:
            1. Write the data to a .mat file. 
            2. Call the Matlab script to a Matlab wrapper
            which calls ldml_learn and read the written matrix back. 
            3. Returns the X_tr transformed matrix. """
        self.X_tr = X_tr
        self.y_train = y_train
        np.savetxt('X_tr.mat', X_tr)
        np.savetxt('y_train.mat', y_train)

        os.system('matlab -nodesktop < ldml_wrap_learn.m')

        # Read the transformation back and store it into self.L
        self.L = scipy.io.loadmat("L.mat")

    def transform(self, y):
        if y is None:
            y = self.X_tr
        return y.dot(self.L.T)

    def predict_proba(self, X_te):
        """Predicts the probabilities of each of the test samples. 
        Ensure that the X_te passed to this function is transformed
        before sending it here. """
        test_samples = X_te.shape[0]
        self.X_tr = self.transform(self.X_tr)
        clf = NearestCentroid()
        clf.fit(self.X_tr, self.y_train)
        centroids = clf.centroids_
        probabilities = np.zeros((test_samples, centroids.shape[0]))
        for sample in xrange(test_samples):
            probabilities[sample] = sk_nearest_neighbour_proba(centroids, X_te[sample, :])
        return probabilities

class MLThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None
    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
    def join(self):
        Thread.join(self)
        return self._return
