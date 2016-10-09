import numpy as np
import datetime


class KNearestClassifier:
    """
    """
    
    def __init__(self, ):
        pass

    def train(self, X, Y):
        print type(X)
        print type(Y)
        self.X_train = X
        self.Y_train = Y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0 :
            time1 = datetime.datetime.now()
            dists = self.compute_distances_no_loops(X)
            time2 = datetime.datetime.now()
        elif num_loops == 1:
            time1 = datetime.datetime.now()
            dists = self.compute_distances_one_loop(X)
            time2 = datetime.datetime.now()
        elif num_loops == 2:
            time1 = datetime.datetime.now()
            dists = self.compute_distances_two_loops(X)
            time2 = datetime.datetime.now()
        else :
            print "not right"

        print "Experiment Done, Time taken : ", time2 - time1


    def compute_distances_no_loops(self, X):
        """
        
        Arguments:
        - `self`:
        - `X`:
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        test_sum = np.sum(np.square(X), axis=1)
        train_sum = np.sum(np.square(self.X_train), axis=1)

        inner_product = np.dot(X, self.X_train.T)
        return np.sqrt(-2 * inner_product + test_sum.reshape(-1,1) + train_sum)

        
    def compute_distances_one_loop(self, X):
        """
        
        Arguments:
        - `self`:
        - `X`:
        """
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis=1)) # broadcasting
        return dists
        

    def compute_distances_two_loops(self, X):
        """
        
        Arguments:
        - `self`:
        - `X`:
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))        

        for i in xrange(num_test):
            for j in xrange(num_train):
                print "Calculating distance - Test ID : " , i, " Train ID : ", j
                dists[i, j] = np.sqrt(np.sum(np.square(X[i, : ] - self.X_train[j,:])))
        return dists


    def predict_labels(self, dists, k):
        """
        
        Arguments:
        - `dists`:
        - `k`:
        """
        
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in xrange(num_test):
            closest_y = []
            y_indices = np.argsort(dists[i,:], axis=0)
            closest_y = self.Y_train[y_indices[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred