import unittest
import eigenfaces
import classifier

class FRTestCase(unittest.TestCase):
    def pcaTest(self):
        self.AssertCase(eigenfaces("pca"), 1)

    def ldaTest(self):
        self.AssertCase(eigenfaces("lda"), 1)

    def pcaLdaTest(self):
        self.AssertCase(eigenfaces("pcalda"), 1)

    def lbpTest(self):
        self.AssertCase(eigenfaces("lbp"), 1)
