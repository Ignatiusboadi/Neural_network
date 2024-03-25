# Neural Network Implementation from Scratch
This project is an implementation of a neural network with two layers. Both layers are lined with sigmoid activation
functions. 

### Overview
The project has two files:

- ANN.py: This file contains an implementation of a class for neural networks with two layers both lined with sigmoid
activations.
- main.py: This file contains test data and examples. When executed plots of the training and loss will be made. Training
and test accuracies will be displayed in the terminal.

### Installation and Requirements 
To use this project, clone the repository and install the required dependencies using pip:

```Copy code
git clone https://github.com/Ignatiusboadi/Neural_network.git
cd Neural_network
pip install -r requirements.txt
```

This project requires the following dependencies:

- NumPy
- Matplotlib

## Example Usage
To train the neural network on a given dataset, where the units in the first layer are 2 and 10 in the second layer
with 1 output, initialize the network as below. You can then apply the fit method after which the object can be used
on new data. See `main.py` for more examples.
```Copy code
alpha = 0.001
n_epochs = 10000
h0, h1, h2 = 2, 10, 1
neural_net = NeuralNetwork(h0, h1)
neural_net.fit(X_train, Y_train, X_test, Y_test, n_epochs, alpha)
```

After installation, you can run the code below in the terminal you cloned the project into to execute the main.py file.

```Copy code
python main.py
```

## Contributing
Contributions to this project are welcome! If you'd like to contribute, please fork the repository,
make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
