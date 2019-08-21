"""
A loss functions helps adjust the models prediction by quantifying how far of it is from the ground truth.
"""
import numpy as np
from dllibnet.tensor import Tensor
from dllibnet.basic_utils.utils import tensor_diff


class Loss:
    def Loss(self, predicted: Tensor, actual: Tensor) -> float:
        """
        Base Loss function
        :param predicted: Model's predicted tensor
        :param actual: Actual ground truth (Target)
        :return: loss value
        """
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """
        Gradient vector partial derivatives of the loss function wrt to each of the predicted values.
        :param predicted:
        :param actual:
        :return:
        """
        raise NotImplementedError

    class MSE(Loss):
        """
        Mean squared error loss, mean of the squared error.
        """

        def Loss(self, predicted: Tensor, actual: Tensor) -> float:
            assert predicted.shape == actual.shape
            return np.sum(np.square(tensor_diff(predicted - actual))) / predicted.shape[0]

        def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
            return 2 * tensor_diff(predicted, actual)
