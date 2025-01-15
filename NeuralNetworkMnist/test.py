from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot
from NeuralNetworkMnist import NeuralNetworkMnist

# For plotting and visualization pupose
def plot_images(arr, image_name):
    for i in range(9):  
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(arr[i], cmap='gray')
        pyplot.savefig(image_name)

if __name__ == "__main__":
    print("=== MNIST NEURAL NETWORK TEST ===")
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    plot_images(train_X, "train_images.png")
    plot_images(test_X, "test_images.png")

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
    
    nn = NeuralNetworkMnist(X[:10000], Y[:10000], 0.0001)
    nn.train()

    correct = 0
    total = len(test_y)
    for i in  range(total):
        if test_y[i] == nn.predict(test_X[i].flatten())[0]:
            correct += 1

    print(f'Coverage: {round(correct * 100 / total)}% ({correct}/{total})')
