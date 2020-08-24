import numpy as np
from planar_utils import sigmoid

np.random.seed(1)                  # For consistentency

class Model:
    def __init__(self, n_x, n_h, n_y, learning_rate = 1):
        
        # Check if n_a need to be a member variable
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y

        self.learning_rate = learning_rate

        self.W1 = np.random.randn(self.n_h, self.n_x) * 0.01
        self.b1 = np.zeros((self.n_h, 1))

        self.W2 = np.random.randn(self.n_y, self.n_h) * 0.01
        self.b2 = np.zeros((self.n_y, 1))

    def forward_propagation(self, X):
        """
        Argument:
        X -- input data of size (n_x, m)
 
        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        
        # Implement Forward Propagation to calculate A2 (probabilities)
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = sigmoid(Z2)
        
        assert(A2.shape == (1, X.shape[1]))
        
        cache = {"Z1": Z1,  "A1": A1, "Z2": Z2, "A2": A2}
        
        return A2, cache

    def compute_cost(self, A2, Y):
        """
        Computes the cross-entropy cost 
        
        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, m)
        Y -- "true" labels vector of shape (1, m)
        
        Returns:
        cost -- cross-entropy cost 
        """
        
        m = Y.shape[1]  # number of example

        # Compute the cross-entropy cost
        logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
        cost = np.squeeze(-np.sum(logprobs) / m)
        # print(cost)
        assert(isinstance(cost, float))
        
        return cost

    def backward_propagation(self, cache, X, Y):
        """
        Implement the backward propagation 
        
        Arguments:
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]
            
        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache['A1']
        A2 = cache['A2']
        
        # Backward propagation: calculate dW1, db1, dW2, db2. 
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.dot(self.W2.T, dZ2)*(1-np.power(A1,2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.W1 -=  self.learning_rate * dW1
        self.b1 -=  self.learning_rate * db1
        self.W2 -=  self.learning_rate * dW2
        self.b2 -=  self.learning_rate * db2

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
        return grads

    def fit(self, X, Y, num_iterations = 10000, print_cost=False):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations
        
        Returns: (Not explicitly but already in object Weights and Bias)
         parameters learnt by the model. They can then be used to predict.
        """
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):
            A2, cache = self.forward_propagation(X)
        
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = self.compute_cost(A2, Y)
    
            # Backpropagation and Gradient Descent.
            grads = self.backward_propagation(cache, X, Y)
            
            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print (f"Cost after iteration {i}: {cost}")

    def predict(self, X):
        """
        Using the learned parameters, predicts a class for each example in X
        
        Arguments:
        X -- input data of size (n_x, m)
        
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        
        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        A2, cache = self.forward_propagation(X)
        predictions = np.around(A2)

        
        return predictions

    def accuracy(self, X, Y):
        C = 0 
        W = 0
        predictions = self.predict(X)
        for i in range(Y.shape[1]):
            if predictions[0, i] == Y[0, i]:
                C += 1 
            else:
                W += 1 
        
        accuracy = 100 * (C / (C + W))
        return accuracy

