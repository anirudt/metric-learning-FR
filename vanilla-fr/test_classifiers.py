import unittest
import eigenfaces
import classifier

class FRTestCase(unittest.TestCase):
    def pcaTest(self):
        self.AssertCase(eigenfaces.main("pca", "ATT"), 1)

    def ldaTest(self):
        self.AssertCase(eigenfaces.main("lda", "ATT"), 1)

    def pcaLdaTest(self):
        self.AssertCase(eigenfaces.main("pcalda", "ATT"), 1)

    def lbpTest(self):
        self.AssertCase(eigenfaces.main("lbp", "ATT"), 1)
