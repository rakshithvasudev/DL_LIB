"""
Neural nets are made of layers. They feed forward and backward propagate gradients.

for instance : input -> linear (matmul) ->  Activation (Tanh) -> linear -> output
"""
from typing import Dict, Callable

import numpy as np
from dllibnet.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward propagate.
        :param inputs:
        :return:
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backprop gradients.
        :param grad:
        :return:
        """
        raise NotImplementedError


class Linear(Layer):
    """
    FF & BP is done here.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs are (batch_size, input_size)
        # outputs are (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        y = wx + b
        :param inputs:
        :return:
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then partial derivatives
        dy/da = f'(x) * b ,
         dy/db = f'(x) * a ,
         dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        then dy/db = a.T @ f'(x)
        then dy/dc = f'(x) @ b.T

        :param grad:
        :return:
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    Applies non linearity elementwise.
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def tanh(self, x: Tensor) -> Tensor:
        return np.tanh(x)

    def tanh_prime(self, x: Tensor) -> Tensor:
        y = self.tanh(x)
        return 1 - np.square(y)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) & x = g(z)
        then dy/dz = f'(x) * g'(z)

        :param grad:
        :return:
        """
        return self.f_prime(self.inputs) * grad

class Tanh(Activation):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_prime)
