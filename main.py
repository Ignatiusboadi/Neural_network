from ANN import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

var = 0.2
n = 800
class_0_a = var * np.random.randn(n//4, 2)
class_0_b = var * np.random.randn(n//4, 2) + (2, 2)

class_1_a = var * np.random.randn(n//4, 2) + (0, 2)
class_1_b = var * np.random.randn(n//4, 2) + (2, 0)

X = np.concatenate([class_0_a, class_0_b, class_1_a, class_1_b], axis =0)
Y = np.concatenate([np.zeros((n//2, 1)), np.ones((n//2, 1))])

rand_perm = np.random.permutation(n)

X = X[rand_perm, :]
Y = Y[rand_perm, :]

X = X.T
Y = Y.T

# train test split
ratio = 0.8
X_train = X[:, :int(n*ratio)]
Y_train = Y[:, :int(n*ratio)]

X_test = X[:, int(n*ratio):]
Y_test = Y[:, int(n*ratio):]

plt.scatter(X_train[0, :], X_train[1, :], c=Y_train[0, :])
plt.show()

alpha = 0.001
n_epochs = 10000
h0, h1, h2 = 2, 10, 1
neural_net = NeuralNetwork(h0, h1)
neural_net.fit(X_train, Y_train, X_test, Y_test, n_epochs, alpha)
