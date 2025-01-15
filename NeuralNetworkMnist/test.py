from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot
from NeuralNetworkMnist import NeuralNetworkMnist

# For plotting and visualization pupose
def plot_images(arr):
    for i in range(9):  
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(arr[i], cmap='gray')
        pyplot.savefig("figures.png")

if __name__ == "__main__":
    print("=== MNIST NEURAL NETWORK TEST ===")
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    plot_images(train_X)

    X = []
    for im in train_X:
        imFlatten = im.flatten()
        binaryImage = (imFlatten > 0).astype(int)
        X.append(binaryImage)
    X = np.array(X)

    Y = []
    for res in train_y:
        arr = [0] * 10
        arr[res] = 1
        Y.append(arr)
    Y = np.array(Y)
    
    nn = NeuralNetworkMnist(X[:1000], Y[:1000])
    nn.train()
    
    # write the tests for number to guess
    for i in range(10):
        print(f'expected: {test_y[i]} | got: {nn.predict(test_X[i].flatten())}')
