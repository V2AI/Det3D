import math
from queue import Queue

import torch
import torch.cuda.comm as comm
import torch.distributed as dist
import torch.nn.functional as F
from IPython import embed
from torch.autograd.function import once_differentiable
from torch.nn.modules.batchnorm import _BatchNorm

from . import syncbn_gpu


class DistributedSyncBNFucntion(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        training=True,
        momentum=0.1,
        eps=1e-5,
        sync=True,
    ):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.sync = sync

        if ctx.training:
            ex, exs = syncbn_gpu.batch_norm_collect_statistics(x)

            world_size = dist.get_world_size()
            if ctx.sync:
                # ex_all  = torch.empty(world_size, ex.size(0), dtype=ex.dtype, device=ex.device)
                # exs_all = torch.empty(world_size, ex.size(0), dtype=ex.dtype, device=ex.device)
                # ex_l    = [ex_all.narrow(0, i, 1) for i in range(world_size)]
                # exs_l   = [exs_all.narrow(0, i, 1) for i in range(world_size)]
                # dist.all_gather(ex_l, ex)
                # dist.all_gather(exs_l, exs)

                # ex = ex_all.mean(0)
                # exs = exs_all.mean(0)

                ex_all_reduce = dist.all_reduce(ex, async_op=True)
                exs_all_reduce = dist.all_reduce(exs, async_op=True)

                ex_all_reduce.wait()
                exs_all_reduce.wait()
                ex /= world_size
                exs /= world_size

            n = x.numel() / x.shape[1] * world_size
            ctx.cf = n / (n - 1)  # correction factor
            var = (exs - ex ** 2) * ctx.cf
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * ex)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var)

            ctx.mark_dirty(running_mean, running_var)

            y = syncbn_gpu.batch_norm_transform_input(
                x, gamma, beta, ex, exs, ctx.eps, ctx.cf
            )

            ctx.save_for_backward(x, ex, exs, gamma, beta)

        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_ouput):
        x, ex, exs, gamma, beta = ctx.saved_tensors

        (
            grad_gamma,
            grad_beta,
            grad_ex,
            grad_exs,
        ) = syncbn_gpu.batch_norm_collect_grad_statistics(
            x, grad_ouput, gamma, ex, exs, ctx.eps, ctx.cf
        )

        if ctx.training:
            if ctx.sync:
                world_size = dist.get_world_size()
                # grad_ex_all  = torch.empty(world_size, grad_ex.size(0), dtype=grad_ex.dtype, device=grad_ex.device)
                # grad_exs_all = torch.empty(world_size, grad_ex.size(0), dtype=grad_ex.dtype, device=grad_ex.device)
                # grad_ex_l    = [grad_ex_all.narrow(0, i, 1) for i in range(world_size)]
                # grad_exs_l   = [grad_exs_all.narrow(0, i, 1) for i in range(world_size)]
                # dist.all_gather(grad_ex_l, grad_ex)
                # dist.all_gather(grad_exs_l, grad_exs)

                # grad_ex = grad_ex_all.mean(0)
                # grad_exs = grad_exs_all.mean(0)
                grad_ex_all_reduce = dist.all_reduce(grad_ex, async_op=True)
                grad_exs_all_reduce = dist.all_reduce(grad_exs, async_op=True)

                grad_gamma_all_reduce = dist.all_reduce(grad_gamma, async_op=True)
                grad_beta_all_reduce = dist.all_reduce(grad_beta, async_op=True)

                grad_ex_all_reduce.wait()
                grad_exs_all_reduce.wait()

                grad_gamma_all_reduce.wait()
                grad_beta_all_reduce.wait()

                grad_ex /= world_size
                grad_exs /= world_size

        grad_input = syncbn_gpu.batch_norm_input_backward(
            x, grad_ouput, gamma, ex, exs, grad_ex, grad_exs, ctx.eps, ctx.cf
        )

        return grad_input, grad_gamma, grad_beta, None, None, None, None, None, None


class DistributedSyncBN(_BatchNorm):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.01,
        affine=True,
        track_running_stats=True,
        sync=True,
    ):
        super(DistributedSyncBN, self).__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.sync = sync

    def forward(self, x):
        if self.training and self.sync:
            return DistributedSyncBNFucntion.apply(
                x,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_var,
                self.training,
                self.momentum,
                self.eps,
                self.sync,
            )
        else:
            exponential_average_factor = 0.0

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(
                            self.num_batches_tracked
                        )
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor,
                self.eps,
            )


if __name__ == "__main__":
    pass
