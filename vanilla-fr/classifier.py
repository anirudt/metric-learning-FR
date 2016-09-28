import numpy as np
import pdb
import cv2
import logging
from skimage.feature import local_binary_pattern
from modshogun import RealFeatures, MulticlassLabels
from modshogun import LMNN as shogun_LMNN
import matplotlib.pyplot as plt
from metric_learn import ITML_Supervised

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
        self.A_space = np.matrix(self.selected_eigen_vecs.T) * np.matrix(A)
        # Need to return values to be used in cascaded classifier systems
        # (_,feats) = self.A_space.shape
        #self.new_space = np.zeros((num_people, num_points, feats))
        #for i in xrange(num_people):
        #  for j in xrange(num_points):
        #    self.new_space[i,j,:] = self.A_space[i*num_points+j,:]
            
        # TODO: Change the following return to the reshaped vals
        return self.selected_eigen_vecs, self.A_space

    def transform(self, y):
        """ Transforms the given test data with the developed model"""
        y = y.T
        y = y - self.mean
        self.test_projection = np.matrix(self.selected_eigen_vecs.T) * np.matrix(y).T
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
    def __init__(self, k=3, use_pca=False):
        self.k = k
        self.eigenvecs = None
        self.space = None
        self.space_model = PCA()
        self.use_pca = use_pca
        self.metric_model = None
        
    def fit(self, feats, labels):
        self.eigenvecs, self.space = self.space_model.fit(feats, labels)
        feat = RealFeatures(self.space)
        self.metric_model = shogun_LMNN(feat, MulticlassLabels(labels.astype(np.float64)), self.k)
        self.metric_model.set_maxiter(1000)
        self.metric_model.set_regularization(0.50)
        self.metric_model.set_obj_threshold(0.001)
        self.metric_model.set_stepsize(1e-7)

        #pdb.set_trace()
        L = np.eye(self.space.shape[1])
        self.metric_model.train(L)

        stats = self.metric_model.get_statistics()
        #plt.plot(stats.obj.get())
        #plt.grid(True)
        #plt.show()
        self.linear_transform = self.metric_model.get_linear_transform()

        self.projected_data = np.dot(self.linear_transform, self.space)
        norms = np.linalg.norm(self.projected_data, axis=0)
        self.projected_data /= norms
        # Fit the data with PCA first.
        # pdb.set_trace()
        return self.eigenvecs, self.projected_data

    def transform(self, y):
        # Transform using PCA first.
        test_proj, _ = self.space_model.transform(y)

        # On the projection in the resultant space, apply LMNN.
        lk = np.dot(self.linear_transform, test_proj)
        lk = lk/np.linalg.norm(lk, axis=0)

        return lk, self.projected_data
class ITML:
    def __init__(self, num_constraints=200):
        self.space_model = PCA()
        self.metric_model = ITML_Supervised(num_constraints)

    def fit(self, feats, labels):
        """Fits the model to the prescribed data."""
        pdb.set_trace()
        self.eigenvecs, self.space = self.space_model.fit(feats, labels)
        pdb.set_trace()
        self.metric_model.fit(self.space.T, labels)

    def transform(self, y):
        """Transforms the test data according to the model"""
        test_proj, _ = self.space_model.transform(y)
        pdb.set_trace()
        return self.metric_model.transform(y)
