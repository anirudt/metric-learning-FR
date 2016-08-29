# Overall Idea:
The idea is to use matrix perturbation theory to find the minimum size of the training set, and thus the minimum number of views. Try to estimate the bound of the intraclass separability. Go through materials for this and try to develop Perturbation Bounds.
Read the newest version of the Face Recognition paper for better clarity. One possible ideology can be, assimilate as many views as necessary, and try training them, reduce the number of images, and keep plotting the accuracy as a function of the amount of training images that you take. This could lead to a fair amount of knowledge about the amount of information that you need to uniquely characterise an individual.

Use a single image for each individual, and try to modify it for simulating several scenarios and try to interpret the relative recognition accuracies amongst them.

# Immediate Next Day TODOs:
- Add integration testing methodologies.
- Testing of PCA, LDA system separately using nearest neighbour methods.
- Get preprocessing work using SIFT.
- Integration Testing scripts.
- Implementations of SVM, kNN.
- and LBP. Implementation of LMNN for a first hand impression of how good this distance metric learning method is. Also, *consolidate all evidence* to prove that these methods are different from inherent similar techniques like the LDA or SVM.

# Things Done:
- PCA, LDA classful implementation.
- LBP Classful implementation.

# TODO:
- Prior Art establishment in the areas of *Face Recognition*, *Matrix Perturbation*, *Facial Deformation* and *Random Matrix Theory*.
- Look for combinations of PCA and Perturbation Theory as well.
- Search for relevant Facial Deformation Datasets.
- Repositize papers accordingly.
- Read through the requisite chapters for understanding the relevant material.
- Establish differences between the template matching algorithm by Karhunen and Louvre.
- Preprocessing module should have:
  - Face Detector System
  - Place eyes at the same level, etc
  - Pose Estimation definitely needs to be done, and binning a test image. Refer the pose.pdf for this.

# Papers:
- All @ papers/
- EigenFaces paper - (download from GMail)
- (This paper)[paper/bounds_docs.pdf] gives a thorough understanding of the eigen value perturbation bounds in case of an varied cases.
- (rigorous_perturbations.pdf) gives a good understanding of the mathematical treatment of the perturbation bounds like Bauer Fike, etc. along with an example.

# DONE:
- Read through Wikipedia pages of EigenFaces and Matrix Perturbation Theory.
- Went through the Research paper of EigenFaces by Turk and Pentland.
- Went through rot1.pdf and the Testing/Experiment section of the EigenFaces Paper.
- Went through pose.pdf, the View-Specific EigenSpaces concept.

# Idea:
- Start a primal understandable working version of EigenFaces in Matlab and Python. Try C for showing your prowess.
- Categorize the face deformations and start devloping a separate module for each one of them
- Start analysing for multi-resolution and extending a similar idea for it.


# Unanswered Questions:
- Over what parameter would we have to develop a PDF of the eigenvalues for the disturbances caused?

# Resources:
- Unit testing: http://docs.python-guide.org/en/latest/writing/tests/
