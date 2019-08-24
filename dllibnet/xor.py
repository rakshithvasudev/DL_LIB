"""
XOR isn't separable linearly. Let's see if the framework can converge.
"""

import numpy as np
from dllibnet.train import train
from dllibnet.nn import NeuralNet
from dllibnet.layers import Linear, Tanh

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]])

neural_nn = NeuralNet(
    [
        Linear(input_size=2, output_size=2),
        Tanh(),
        Linear(input_size=2, output_size=2)
     ])

train(neural_nn,inputs, targets)

for x, y in zip(inputs, targets):
    predicted = neural_nn.forward(x)
    print(x, predicted, y)
