import numpy as np

class NeuralNetworkXor:
    """
    Xor Neural Network initialization
    -> Initialize a neural network with 1 input layer, 1 hidden layer, 1 output layer
    @param x : Trainig data set
    @param y : Expected results
    @param learning_rate : Learning rate of our Neural Network
                            (0.075 fit the best here)
    @param nb_epoch : Number of epoch to train the Neural Network
    @param inputs, hidden, outputs : Number of nodes inside each layer
    """
    def __init__(self, x, y, learning_rate=0.075, nb_epoch=10000, inputs=2, hidden=2, outputs=1):
        print("Neural Network Initialization ...\n")
        np.random.seed(0)
        
        self.train_data = x
        self.expected_res = y
        self.lr = learning_rate
        self.nb_iteration = nb_epoch
        self.losses = []

        # # Weights and biases to get the hidden layer (input -> hidden)
        self.hidden_weights = np.random.uniform(size=(inputs, hidden))
        self.hidden_biases = np.zeros((1, hidden))

        # Weights and biases to get the output layer (hidden -> output)
        self.output_weights = np.random.uniform(size=(hidden, outputs))
        self.output_biases = np.zeros((1, outputs))

    """
    Neurone Activation function and its derivative
    -> Compute the Sigmoid and Sigmoid derivative function to every value
    @param X : value to compute (a matrix, np.array, in our case)
    @return : value computed
    """
    def neuroneActivation(self, X):
        return 1 / (1 + np.exp(-X))
    
    def neuroneActivationDerivative(self, X):
        return X * (1 - X)

    """
    Back propagation function
    -> Compute the back propagation algorithm to update each layer values
    @param V : Forward propagation values of each layer
    @param eT: Error term (MSE fo us)
    """
    def backPropagation(self, V, eT):
        # Compute the gradient to modify the layer's data
        d_output = eT * self.neuroneActivationDerivative(V["oA"])
        d_hidden = d_output @ self.output_weights.T * self.neuroneActivationDerivative(V["hA"])

        gradient_hidden = self.train_data.T @ d_hidden
        gradient_output = V["hA"].T @ d_output

        # Update the weights of the hidden and output layer
        self.hidden_weights += self.lr * gradient_hidden
        self.output_weights += self.lr * gradient_output

        # Update the biases of the hidden and output layer
        self.hidden_biases += self.lr * np.sum(d_hidden, axis=0, keepdims=True)
        self.output_biases += self.lr * np.sum(d_output, axis=0, keepdims=True)

    """
    Forward propagation function
    -> Compute the foward propagation algorithm to update each layer values
    @param X : Given input layer values (a training set for training or 
                an input for a result)
    @return : Activated output layer and an object containing every layer after
                and before activation
    """
    def forwardPropagation(self, X):
        # Activation of the hidden layer
        hidden = np.dot(X, self.hidden_weights) + self.hidden_biases
        A_hidden = self.neuroneActivation(hidden)

        # Activation of the output layer
        output = np.dot(A_hidden, self.output_weights) + self.output_biases
        A_output = self.neuroneActivation(output)

        # Return the result of all the neurones activated
        values = {"h": hidden, "hA": A_hidden, "o": output, "oA": A_output}
        return A_output, values
    
    """
    Loss Function
    -> Compute the mean squared error (MSE)
    @param R : Results that the forward propagation gave us
    @return : MSE of R
    """
    def lossFunction(self, R):
        # Compute the squared error
        loss = 0.5 * (self.expected_res - R) ** 2
        self.losses.append(np.sum(loss))

        # Return the error term
        return self.expected_res - R

    """
    Training function
    -> Train the Neural Network using the back and forward propagation
    """
    def train(self):
        for _ in range(self.nb_iteration):
            R, V = self.forwardPropagation(self.train_data)
            errorTerm = self.lossFunction(R)
            self.backPropagation(V, errorTerm)

    """
    Prediction function
    -> Predict a result following the weights and biases of each layer
    @param X : matrix (np.array) of the two input values
    @return : the convergence to O or 1
    """
    def predict(self, X):
        res, _ = self.forwardPropagation(X)
        print(f"-----------\nProbality : {res}")
        return (res > 0.5).astype(int)