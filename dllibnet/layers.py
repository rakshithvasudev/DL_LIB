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
    Applies non linearity elementwise
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
