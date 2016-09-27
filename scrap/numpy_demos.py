import numpy as np

# image data after flattening
array1=np.array([1.1, 2.2, 3.3])
array2=np.array([1, 2, 3])

#differences
print array1 - array2

#squares
print np.square(array1 - array2)

#sum of squares
print np.sum(np.square(array1 - array2))

#square root of sum of squares
print np.sqrt(np.sum(np.square(array1 - array2)))




# creating a set of train images data
train_array1 = np.random.randint(255, size=5)
train_array2 = np.random.randint(255, size=5)
train_array3 = np.random.randint(255, size=5)
train_array4 = np.random.randint(255, size=5)
train_array5 = np.random.randint(255, size=5)


test_array1 = np.random.randint(255, size=5)
test_array2 = np.random.randint(255, size=5)
test_array3 = np.random.randint(255, size=5)



train_data = np.asarray([train_array1, train_array2, train_array3, train_array4, train_array5])
test_data = np.asarray([test_array1, test_array2, test_array3])

print "Distance Using two loops"
num_test = np.array(test_data).shape[0]
num_train = np.array(train_data).shape[0]
dist_two_loops = np.zeros((num_test, num_train))        

for i in xrange(num_test):
    for j in xrange(num_train):
        dist_two_loops[i,j] = np.sqrt(np.sum(np.square(test_data[i] - train_data[j])))

print dist_two_loops


print "Distance Using one loop"
dist_one_loop = np.zeros((num_test, num_train))        
for i in xrange(num_test):
    dist_one_loop[i,:] = np.sqrt(np.sum(np.square(train_data - test_data[i]), axis=1)) # broadcasting

print dist_one_loop


print "distance without loops"
print "test data is", test_data
test_sum = np.sum(np.square(test_data), axis=1) # num_test x 1
train_sum = np.sum(np.square(train_data), axis=1) # num_test x 1

inner_product = np.dot(test_data, train_data.T)
print "Inner product is", inner_product


# is the same as (a-b)^2
# a^2 - 2*a*b  + b*2
np.sqrt(-2 * inner_product + test_sum.reshape(-1,1) + train_sum)
