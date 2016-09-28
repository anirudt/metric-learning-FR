from modshogun import LMNN, RealFeatures, MulticlassLabels
import numpy as np
import pdb
from metric_learn import ITML_Supervised

def test_LMNN():
    X = np.eye(80)
    Y = np.array([i for j in range(4) for i in range(20)])
    feats = RealFeatures(X.T)
    labs = MulticlassLabels(Y.astype(np.float64))
    arr = LMNN(feats, labs, 2)
    arr.train()
    L = arr.get_linear_transform()
    X_proj = np.dot(L, X.T)
    test_x = np.eye(80)[0:20:]
    test = RealFeatures(test_x.T)
    test_proj = np.dot(L, test_x.T)
    pdb.set_trace()

def test_ITML():
    X = np.random.rand(40, 40)
    Y = np.array([i for j in range(2) for i in range(20)])
    itml = ITML_Supervised(num_constraints=200)
    itml.fit(X, Y)
    pdb.set_trace()

if __name__ == "__main__":
    test_ITML()
