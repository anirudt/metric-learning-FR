## What?
This is a Face Recognition learning repo for research.

## TODO:
- [X] Port LFW: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html
- [X] Test why the 40 img dataset gave a segfault: Done, this is because we fed a k > number of points per label.
- [X] Do the preprocessing as conducted by them, and try out kNN with PCA on Euclidean and Mahalanobis distance, using
the `get_distance()` API given by Shogun.
- [X] Get LMNN to work.
- [ ] Get ITML to work.
- [ ] Conduct parameter tuning.
      Edit: Needs to be made ready for our metric learning algorithms.
- [ ] Modify the existing NN function to give a kNN, for better comparison.
- [X] Mangle the data in the format required by the metric-learn module and feed it for results.
  Edit: Doing this inside the classful implementation itself.
