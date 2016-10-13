## What?
This is a Face Recognition learning repo for research.

## Immediate TODOs:
- [ ] Conduct parameter tuning.
- [ ] Log the params for the algorithms.

## TODO:
- [X] Conduct an ensemble classifier on all of the metric learning based techniques.
  Partly done, though hard voting is also required to be implemented.
- [ ] Port the code to my machine, make using this
```
cmake -DCMAKE_INSTALL_PREFIX="$HOME/shogun-install" -DPythonModular=ON -DBUNDLE_EIGEN=ON ..
```
- [X] Port LFW: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html
- [X] Test why the 40 img dataset gave a segfault: Done, this is because we fed a k > number of points per label.
- [X] Do the preprocessing as conducted by them, and try out kNN with PCA on Euclidean and Mahalanobis distance, using
the `get_distance()` API given by Shogun.
- [X] Get LMNN to work.
      Edit: Needs to be made ready for our metric learning algorithms.
- [ ] Modify the existing NN function to give a kNN, for better comparison.
- [X] Mangle the data in the format required by the metric-learn module and feed it for results.
  Edit: Doing this inside the classful implementation itself.

## Working Algorithms:
- ITML
- LMNN
- LSML
- SDML
- LDML
- NCA
- RCA
