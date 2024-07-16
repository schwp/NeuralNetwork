import numpy as np

class NeuralNetworkXor:
    def __init__(self, x, y, learning_rate=0.01, nb_epoch=10000, inputs=2, hidden=3, outputs=1):
        print("Neural Network Initialization ...\n")
        
        self.train_data = x
        self.expected_res = y
        self.lr = learning_rate
        self.nb_iteration = nb_epoch
        self.losses = []

        # # Weights and biases to get the hidden layer (input -> hidden)
        self.hidden_weights = np.random.uniform(size=(inputs, hidden))
        self.hidden_biases = np.random.uniform(size=(1, hidden))

        # Weights and biases to get the output layer (hidden -> output)
        self.output_weights = np.random.uniform(size=(hidden, outputs))
        self.output_biases = np.random.uniform(size=(1, outputs))

    def neuroneActivation(self, X):
        return 1 / (1 + np.exp(-X))
    
    def neuroneActivationDerivative(self, X):
        return X * (1 - X)

    def backPropagation(self, V, eT):
        # Compute the gradient to modify the layer's data
        gradient_hidden = self.train_data.T @ (((eT * self.neuroneActivationDerivative(V["oA"])) * self.output_weights.T) * self.neuroneActivationDerivative(V["hA"]))
        gradient_output = V["oA"].T @ (eT * self.neuroneActivationDerivative(V["oA"]))

        # Update the weights of the hidden and output layer
        self.hidden_weights += self.lr * gradient_hidden
        self.output_weights += self.lr * gradient_output

        # Update the biases of the hidden and output layer
        self.hidden_biases += np.sum(self.lr * ((eT * self.neuroneActivationDerivative(V["oA"])) * self.output_weights.T) * self.neuroneActivationDerivative(V["hA"]), axis=0)
        self.output_biases += np.sum(self.lr * eT * self.neuroneActivationDerivative(V["oA"]), axis=0)

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
    
    def lossFunction(self, R):
        # Compute the squared error
        loss = 0.5 * (self.expected_res - R) ** 2
        self.losses.append(np.sum(loss))

        # Return the error term
        return self.expected_res - R

    def train(self):
        for i in range(self.nb_iteration):
            R, V = self.forwardPropagation(self.train_data)
            errorTerm = self.lossFunction(R)
            self.backPropagation(V, errorTerm)

    def predict(self, X):
        res, _ = self.forwardPropagation(X)
        return (res >= 0.5).astype(int)