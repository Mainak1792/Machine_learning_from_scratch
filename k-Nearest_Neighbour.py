import numpy as np
from collections import Counter

def e_distance(x1,x2): #define_the_eucledian_distance
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self,k=3): #cluster_to_be_defined_by_user_default_3
        self.k=k

    def fit(self,X,y): #Dependent_and_independent_variables
        self.X_train = X
        self.y_train = y

    def predict(self, X): #predict_function
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self,x):
        distances = [e_distance(x,X_train) for X_train in self.X_train] #Compute distances between x and all examples in the training set
        k_idx = np.argsort(distances)[:self.k] # Sort by distance and return indices of the first k neighbors
        k_n_l= [self.y_train[i] for i in k_idx] # Extract the labels of the k nearest neighbor training samples
        most_common= Counter(k_n_l).most_common(1) # return the most common class label
        return most_common[0][0]


