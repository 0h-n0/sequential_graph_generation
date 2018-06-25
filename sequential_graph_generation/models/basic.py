import math

import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

def init_feedforward_weights(dnn: nn.Module,
                             init_mean=0,
                             init_std=1,
                             init_xavier: bool=True,
                             init_normal: bool=True,
                             init_gain: float=1.0):
    for name, p in dnn.named_parameters():
        if 'bias' in name:
            p.data.zero_()
        if 'weight' in name:            
            if init_xavier:
                if init_normal:
                    nn.init.xavier_normal(p.data, init_gain)
                else:
                    nn.init.xavier_uniform(p.data, init_gain)
            else:
                if init_normal:
                    nn.init.normal(p.data, init_gain)
                else:
                    nn.init.uniform(p.data, init_gain)


class GraphLinear(torch.nn.Module):
    """Graph Linear layer.

    This function assumes its input is 3-dimensional.
    Differently from :class:`chainer.functions.linear`, it applies an affine
    transformation to the third axis of input `x`.

    .. seealso:: :class:`torch.nn.Linear`
    """
    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 nonlinearity='sigmoid',
                 init_mean=0,
                 init_std=1,
                 init_xavier: bool=True,
                 init_normal: bool=True,
                 init_gain = None,
                 dropout=0.0,
                 bias=True,
    ):
        super(GraphLinear, self).__init__()

        self.linear = torch.nn.Linear(in_features,
                                      out_features,
                                      bias)
        self.out_features = out_features
        self.nonlinearity = nonlinearity
        
        if not init_gain and nonlinearity is not None:
            init_gain = torch.nn.init.calculate_gain(nonlinearity)
        else:
            init_gain = 1
        
        init_feedforward_weights(self.linear,
                                 init_mean,
                                 init_std,
                                 init_xavier,
                                 init_normal,
                                 init_gain)

    def __call__(self, x):
        # (minibatch, atom, ch)
        s0, s1, s2 = x.size()
        x = x.view(s0 * s1, s2)
        x = self.linear(x)
        x = x.view(s0, s1, self.out_features)
        return x
