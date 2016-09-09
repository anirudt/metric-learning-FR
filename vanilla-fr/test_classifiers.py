import unittest
import eigenfaces
import classifier

class FRTestCase(unittest.TestCase):
    def test_pcaTest(self):
        self.assertEqual(eigenfaces.main("pca", "ATT"), 1)
        #self.assertEqual(eigenfaces.main("pca", "KGP"), 1)

    def test_pcaLdaTest(self):
        self.assertEqual(eigenfaces.main("pcalda", "ATT"), 1)
        #self.assertEqual(eigenfaces.main("pcalda", "KGP"), 1)

    def test_lbpTest(self):
        self.assertEqual(eigenfaces.main("lbp", "ATT"), 1)
        #self.assertEqual(eigenfaces.main("lbp", "KGP"), 1)

if __name__ == '__main__':
    unittest.main()
