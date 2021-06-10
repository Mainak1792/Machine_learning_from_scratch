import numpy as np
class LogisticRegression:
    def __init__(self,learning_rate=0.01, n__iters=1000):
        self.lr=learning_rate
        self.n_iters= n__iters
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
             linear_model= np.dot(X,self.weights)+self.bias
             y_predicted = self.sigmoid(linear_model)

             dw= (1/n_samples)*np.dot(X.T,(y_predicted-y))
             db= (1/n_samples)*np.sum(y_predicted-y)

             self.weights-=self.lr*dw
             self.bias-=self.lr*db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias #calculate_the_linear_model
        y_predicted = self._sigmoid(linear_model) #calculate_the_sigmoid_function
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted] #Binary_classification
        return np.array(y_predicted_cls) 


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
