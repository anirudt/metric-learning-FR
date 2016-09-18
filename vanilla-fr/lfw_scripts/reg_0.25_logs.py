
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======


Total dataset size:
n_samples: 1288
n_features: 1850
n_classes: 7
Extracting the top 150 eigenfaces from 966 faces
done in 0.423s
Projecting the input data on the eigenfaces orthonormal basis
done in 0.057s
Trying LMNN
accuracy = %s 82.298136646
                   precision    recall  f1-score   support

     Ariel Sharon       0.75      0.92      0.83        13
     Colin Powell       0.88      0.85      0.86        60
  Donald Rumsfeld       0.69      0.67      0.68        27
    George W Bush       0.91      0.88      0.90       146
Gerhard Schroeder       0.69      0.80      0.74        25
      Hugo Chavez       0.68      0.87      0.76        15
       Tony Blair       0.67      0.61      0.64        36

      avg / total       0.83      0.82      0.82       322

[[ 12   0   1   0   0   0   0]
 [  0  51   1   5   0   0   3]
 [  2   1  18   2   1   1   2]
 [  0   2   5 129   4   3   3]
 [  1   0   0   1  20   1   2]
 [  1   0   0   0   0  13   1]
 [  0   4   1   4   4   1  22]]
Fitting the classifier to the training set
done in 8.417s
Best estimator found by grid search:
SVC(C=1000.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Predicting people's names on the test set
accuracy = %s 83.2298136646
done in 0.048s
                   precision    recall  f1-score   support

     Ariel Sharon       0.71      0.92      0.80        13
     Colin Powell       0.83      0.83      0.83        60
  Donald Rumsfeld       0.77      0.63      0.69        27
    George W Bush       0.87      0.95      0.90       146
Gerhard Schroeder       0.90      0.72      0.80        25
      Hugo Chavez       0.80      0.80      0.80        15
       Tony Blair       0.72      0.58      0.65        36

      avg / total       0.83      0.83      0.83       322

[[ 12   0   1   0   0   0   0]
 [  0  50   1   8   0   0   1]
 [  2   1  17   4   1   1   1]
 [  0   3   2 138   0   1   2]
 [  1   0   0   2  18   1   3]
 [  1   1   0   0   0  12   1]
 [  1   5   1   7   1   0  21]]
Without the LMNN structure
Fitting the classifier to the training set
done in 26.621s
Best estimator found by grid search:
SVC(C=1000.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.005, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Predicting people's names on the test set
accuracy = %s 86.0248447205
done in 0.088s
                   precision    recall  f1-score   support

     Ariel Sharon       0.85      0.85      0.85        13
     Colin Powell       0.84      0.90      0.87        60
  Donald Rumsfeld       0.89      0.63      0.74        27
    George W Bush       0.85      0.99      0.91       146
Gerhard Schroeder       0.90      0.72      0.80        25
      Hugo Chavez       1.00      0.60      0.75        15
       Tony Blair       0.86      0.67      0.75        36

      avg / total       0.87      0.86      0.85       322

[[ 11   1   1   0   0   0   0]
 [  0  54   0   6   0   0   0]
 [  0   1  17   7   1   0   1]
 [  0   1   0 144   0   0   1]
 [  1   0   1   3  18   0   2]
 [  1   4   0   1   0   9   0]
 [  0   3   0   8   1   0  24]]
