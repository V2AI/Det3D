import numpy as np
import torch
from IPython import embed
from RoI import RoIFunction as RRoIFunc
from RoI import RotateRoIAlign as RRoI
from torch.autograd import Variable, gradcheck

fmap = torch.from_numpy(np.ones((128, 128)) * 0)
fmap[54 : 54 + 8, 54 : 54 + 16] = 1
fmap[54 + 8 : 54 + 16, 54 : 54 + 16] = 10
fmap = fmap.type(torch.float32)
# fmap = torch.randn((128, 128))
fmap = fmap.resize_(1, 1, 128, 128)
# rois = torch.tensor([[52.0, 52.0, 8, 8, 0.0],[100.0, 100,8,8,0]])
# remember that rois: [n, x, y, w, h , theta]
rois = torch.tensor([[0.0, 54.0, 54, 16, 8, 0.2], [0.0, 54.0, 54, 16, 8, -0.2]])

scale = 1.0
pooled_height = 2
pooled_width = 2
ratio = 2

# make variables
variables = [fmap, rois, pooled_height, pooled_width, scale, ratio]
for i, var in enumerate(variables):
    if i < 2:
        req_grad = i == 0
        var = var.cuda()
        variables[i] = Variable(var.double(), requires_grad=req_grad)


if __name__ == "__main__":
    res = gradcheck(RRoIFunc.apply, variables, eps=1e-6, atol=1e-4)
    print("RRoI check result:{}".format(res))
