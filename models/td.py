""" Targeted dropout implementation """
import torch
import torch.nn as nn
import numpy as np
import genotypes as gt
import ops

class TargetedMixedOp(nn.Module):
    """ Targeted mixed operation """
    def __init__(self, C, stride, td_type, td_rate=0.90, drop_rate=0.75):
        super().__init__()

        self.td_type   = td_type   # unit/weight
        self.td_rate   = td_rate   # gamma
        self.drop_rate = drop_rate # alpha

        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = ops.OPS[primitive](C, stride, affine=False)
            self._ops.append(op)
    

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))