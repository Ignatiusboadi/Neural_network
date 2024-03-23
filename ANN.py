import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, layer1=2, layer2=10):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = 1
        self.W1 = None
        self.W2 = None
        self.b1 = None
        self.b2 = None
        self.A1 = None
        self.A2 = None
        self.Z1 = None
        self.Z2 = None

    def init_params(self):
        xav_mean_layer1 = 0
        xav_std_layer1 = np.sqrt(2/(self.layer1 + self.layer2))
        xav_mean_layer2 = 0
        xav_std_layer2 = np.sqrt(2/(self.layer2 + self.layer3))
        self.W1 = np.random.normal(xav_mean_layer1, xav_std_layer1, (self.layer2, self.layer1))
        self.W2 = np.random.normal(xav_mean_layer2, xav_std_layer2, (self.layer3, self.layer2))
        self.b1 = np.random.normal(xav_mean_layer1, xav_std_layer1, (self.layer2, 1))
        self.b2 = np.random.normal(xav_mean_layer2, xav_std_layer2, (self.layer3, 1))
        # return W1, W2, b1, b2

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def loss(self, y_pred: np.ndarray, Y: np.ndarray) -> float:
        return -np.sum(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))/(Y.shape[1])

    def forward_pass(self, X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        # return A2, Z2, A1, Z1

    def backward_pass(self, X, Y):
        # Your code here
        dl_dA2 = (self.A2 - Y) / (self.A2 * (1 - self.A2))
        dA2_dZ2 = self.d_sigmoid(self.Z2)
        dZ2_dW2 = self.A1.T

        self.dW2 = (dl_dA2 * dA2_dZ2) @ dZ2_dW2
        self.db2 = dl_dA2 @ dA2_dZ2.T

        dZ2_dA1 = self.W2
        dA1_dZ1 = self.d_sigmoid(self.Z1)
        dZ1_dW1 = X.T

        self.dW1 = (dZ2_dA1.T * (dl_dA2 * dA2_dZ2) * dA1_dZ1) @ dZ1_dW1
        self.db1 = ((dl_dA2 * dA2_dZ2) @ (dZ2_dA1.T * dA1_dZ1).T).T

        # return dW1, dW2, db1, db2

    def accuracy(self, y_pred, y):
        # Your code here
        y = y.reshape((-1, 1))
        y_pred = y.reshape((-1, 1))
        return np.mean((y == y_pred))

    def predict(self, X):
        self.forward_pass(X)
        return (self.A2 >= 0.5).astype(float)

    def update(self, alpha):
        # Your code here
        self.W1 -= alpha * self.dW1
        self.W2 -= alpha * self.dW2
        self.b1 -= alpha * self.db1
        self.b2 -= alpha * self.db2