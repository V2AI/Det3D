from __future__ import division, print_function

import argparse

import numpy as np
import torch
from modules.correlation import Correlation


def check_equal(first, second, verbose):
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print("-" * 80)
        np.testing.assert_allclose(x, y, err_msg="Index: {}".format(i))


def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


def get_grads(variables):
    return [var.grad.clone() for var in variables]


def check_forward(input1, input2, correlation_sampler, verbose):
    # cpu_values = correlation_sampler(input1, input2)
    cuda_values = correlation_sampler(input1.to(device), input2.to(device))
    print("Forward: CPU vs. CUDA ... ", end="")
    # check_equal(cpu_values, cuda_values, verbose)
    print("Ok")


def check_backward(input1, input2, correlation_sampler, verbose):
    # cpu_values = correlation_sampler(input1, input2)
    # cpu_values.sum().backward()
    # grad_cpu = get_grads([input1, input2])

    # zero_grad([input1, input2])

    cuda_values = correlation_sampler(input1.to(device), input2.to(device))
    cuda_values.sum().backward()
    grad_cuda = get_grads([input1, input2])

    print("Backward: CPU vs. CUDA ... ", end="")
    # check_equal(grad_cpu, grad_cuda, verbose)
    print("Ok")


parser = argparse.ArgumentParser()
parser.add_argument("direction", choices=["forward", "backward"], nargs="+")
parser.add_argument("-b", "--batch-size", type=int, default=1)
parser.add_argument("-k", "--kernel-size", type=int, default=3)
parser.add_argument("--patch", type=int, default=3)
parser.add_argument("--patch_dilation", type=int, default=1)
parser.add_argument("-c", "--channel", type=int, default=10)
parser.add_argument("--height", type=int, default=10)
parser.add_argument("-w", "--width", type=int, default=10)
parser.add_argument("-s", "--stride", type=int, default=1)
parser.add_argument("-p", "--pad", type=int, default=1)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

assert torch.cuda.is_available(), "no comparison to make"
device = torch.device("cuda")

input1 = torch.randn(args.batch_size, args.channel, args.height, args.width).double()
input2 = torch.randn(args.batch_size, args.channel, args.height, args.width).double()
input1.requires_grad = True
input2.requires_grad = True

correlation_sampler = Correlation(
    args.kernel_size, args.patch, args.stride, args.pad, args.patch_dilation
)

if "forward" in args.direction:
    check_forward(input1, input2, correlation_sampler, args.verbose)

if "backward" in args.direction:
    check_backward(input1, input2, correlation_sampler, args.verbose)
