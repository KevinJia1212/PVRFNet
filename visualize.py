import torch
from torch import nn
from torchviz import make_dot
import graphviz
from torch.autograd import Variable
from model64_v1_2 import Net
from torchsummary import summary

net = Net(num_classes=250)
x = Variable(torch.randn(1, 3, 64, 64))
# p = dict(net.named_parameters())
vis_graph = make_dot(net(x), params=dict(net.named_parameters()))
# vis_graph = make_dot(net(x))
vis_graph.view()