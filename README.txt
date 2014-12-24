## ------------------------------------------------------------------------------------
##    README File for GMM and SPLICE Training using Multiprocessing on Large Datasets
##    Copyright (C) 2014,  D S Pavan Kumar
## ------------------------------------------------------------------------------------

gmm.py:
*) The GMM class contains routines to train diagonal covariance GMMs on large datasets using Expectation Maximisation (EM) algorithm and to compute likelihoods of feature vectors.
*) Most other GMM implementations load the entire database into Python memory, so they cannot process large databases.
*) This script processes file by file and supports multiprocessing, and hence is optimal for large databases.
*) System's CPU count is used as the default number of parallel computing processes.
*) The current implementation is written for and tested on the features of a Speech database.
*) Data are specified as a list of feature files, where each file consists of a set of feature vectors (of a speech utterance).
*) Text, HTK and Sphinx file formats for features are supported. In case of text format, each line of the file is considered as a feature vector, and the vector size is expected to be invariant across all the files.
*) If there are issues reading HTK/Sphinx files, please check the endianness of the features. This code assumes that the endianness matches with that of the system.
*) Currently the script supports training mixtures in the powers of 2.

Training process:
The training starts with building single mixture models. The mixtures are then repeatedly doubled followed by three iterations of EM until the desired mixture count is reached. Three iterations of retraining are performed at the end. Mixture doubling is performed as follows: the heaviest component (largest variance) of each mixture is identified (say v), and its mean component m is split into (m + 0.2v) and (m - 0.2v). All other components of means of the splits remain the same. The variances are just copied to both the splits and the weights are halved. Retraining adjusts the parameters.

splice.py
*) The SPLICE class contains routines to trains SPLICE matrices from stereo data and to apply the SPLICE normalisation on test features.

More documentation and scripts will be added in the next version.
