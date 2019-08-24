"""
Optimizer adjusts parameters based on Gradients during backprop.
"""

from dllibnet.nn import NeuralNet


class Optimizer:
    def step(self, NeuralNet):
        raise NotImplemented


class SGD(Optimizer):
    def __init__(self, lr: float = 0.1) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
