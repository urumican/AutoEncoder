import gzip
import cPickle
import numpy as np
import theano
import theano.tensor as T
import random

examples_per_labels = 10


# Load the pickle file for the MNIST dataset.
dataset = 'mnist.pkl.gz'

f = gzip.open(dataset, 'rb')
train_set, dev_set, test_set = cPickle.load(f)
f.close()

#train_set contains 2 entries, first the X values, second the Y values
train_x, train_y = train_set
dev_x, dev_y = dev_set
test_x, test_y = test_set

print 'Train: ', train_x.shape
print 'Dev: ', dev_x.shape
print 'Test: ', test_x.shape

examples = []
examples_labels = []
examples_count = {}

for idx in xrange(train_x.shape[0]):
    label = train_y[idx]

    if label not in examples_count:
        examples_count[label] = 0

    if examples_count[label] < examples_per_labels:
        arr = train_x[idx]
        examples.append(arr)
        examples_labels.append(label)
        examples_count[label]+=1

train_subset_x = np.asarray(examples)
train_subset_y = np.asarray(examples_labels)

print "Train Subset: ",train_subset_x.shape