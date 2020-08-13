import numpy as np
import torch
from torch.autograd import Function


# https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_, on_):
        ctx.lambda_ = lambda_
        ctx.on_ = on_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        on_ = ctx.on_
        lambda_ = grads.new_tensor(lambda_)
        if on_:
            dx = -lambda_ * grads
        else:
            dx = lambda_ * grads
        return dx, None, None

# on turns the layer on if true; if off, turns it off and becomes identity for both forward and backwards pass
class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1, on=True):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_
        self.on = on

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_, self.on)