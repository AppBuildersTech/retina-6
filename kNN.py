import random
import numpy as np
from data_utils import load_CIFAR10


import os
import sys
lib_path = os.path.abspath(os.path.join('classifiers'))
sys.path.append(lib_path)


from kNearest import KNearestClassifier

cifar10_dir = 'datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)


# taking  a small subset while testing
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

print len(X_train)
print len(X_test)

classifier = KNearestClassifier()
classifier.train(X_train, y_train)


dists_two_loops = classifier.predict(X_test, num_loops=2)
dists_one_loop = classifier.predict(X_test, num_loops=1)
dists_no_loops = classifier.predict(X_test, num_loops=0)

print dists_two_loops
print dists_one_loop 
print dists_no_loops

y_test_pred_two = classifier.predict_labels(dists_two_loops, k=1)
y_test_pred_one = classifier.predict_labels(dists_one_loop, k=1)
y_test_pred_none = classifier.predict_labels(dists_no_loops, k=1)