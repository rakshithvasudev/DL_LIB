
from dllibnet.tensor import Tensor
import numpy as np


def tensor_diff(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Subtracts 2 tensors.
    :param tensor1: Tensor 1
    :param tensor2: Tensor 2
    :return: Subtracts 2 from 1
    """

    return tensor1 - tensor2
