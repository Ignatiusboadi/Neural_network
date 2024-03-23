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
        y = y.reshape((-1, 1))
        y_pred = y_pred.reshape((-1, 1))
        return np.mean((y == y_pred))

    def predict(self, X):
        self.forward_pass(X)
        return (self.A2 >= 0.5).astype(float)

    def update(self, alpha):
        self.W1 -= alpha * self.dW1
        self.W2 -= alpha * self.dW2
        self.b1 -= alpha * self.db1
        self.b2 -= alpha * self.db2

    def plot_decision_boundary(self):
        x = np.linspace(-0.5, 2.5, 100)
        y = np.linspace(-0.5, 2.5, 100)
        xv, yv = np.meshgrid(x, y)
        X_ = np.stack([xv, yv], axis=0)
        X_ = X_.reshape(2, -1)
        self.forward_pass(X_)
        plt.figure()
        plt.scatter(X_[0, :], X_[1, :], c=self.A2)
        plt.show()

    def fit(self, X_train, Y_train, X_test, Y_test, n_epochs=10000, alpha=0.01):
        self.init_params()
        train_loss = []
        test_loss = []
        for i in range(n_epochs):
            self.forward_pass(X_train)
            self.backward_pass(X_train, Y_train)
            self.update(alpha)

            train_loss.append(self.loss(self.A2, Y_train))
            self.forward_pass(X_test)
            test_loss.append(self.loss(self.A2, Y_test))

            if i % 1000 == 0:
                self.plot_decision_boundary()

        plt.plot(train_loss)
        plt.plot(test_loss)

        y_pred = self.predict(X_train)
        train_accuracy = self.accuracy(y_pred, Y_train)
        print("train accuracy :", train_accuracy)

        y_pred = self.predict(X_test)
        test_accuracy = self.accuracy(y_pred, Y_test)
        print("test accuracy :", test_accuracy)
