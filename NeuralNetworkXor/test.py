from NeuralNetworkXor import NeuralNetworkXor
import numpy as np

NB_INPUTS = 2
NB_HIDDEN = 4
NB_OUTPUTS = 1

if __name__ == "__main__":
    print("=== XOR NEURAL NETWORK TEST ===")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    y_and = np.array([[0], [0], [0], [1]])
    y_or = np.array([[0], [1], [1], [1]])
    y_nand = np.array([[1], [1], [1], [0]])

    Xor = NeuralNetworkXor(X, y_xor)

    Xor.train()

    for a, b in X:
        print(f'Performing {a} ^ {b} -> {1 if Xor.predict([a, b]) > 0.5 else 0}')
