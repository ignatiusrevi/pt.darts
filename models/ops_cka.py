""" Operations """
import torch
import torch.nn as nn
import genotypes as gt
import torch.nn.functional as F


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: \
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), # 5x5
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine), # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine),
    'res_blck_3x3': lambda C, stride, affine: ResBlock(C, C, stride, affine=affine)
}


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x

def random_shuffle(x):
    batch_size, num_channels, height, width = x.data.size()
    indices = torch.randperm(num_channels)
    x = x[:,indices]

def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, C_in, C_out, stride=1, affine=True):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(C_out, affine=affine)
        self.conv2 = nn.Conv2d(C_out, C_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(C_out, affine=affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or C_in != self.expansion*C_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(C_in, self.expansion*C_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*C_out, affine=affine)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MixedOpCKA(nn.Module):
    """ Mixed operation """
    def __init__(self, C, stride, td_choice='weight', td_rate=0.50, drop_rate=0.50, C_reduction=4):
        super().__init__()
        self.td_choice   = td_choice
        self.C_reduction = C_reduction
        self.max_pool    = nn.MaxPool2d(2,2)

        if td_choice == 'unit':
            self.td_rate, self.drop_rate = 0.75, 0.90 # default td unit
        else: 
            self.td_rate, self.drop_rate = td_rate, drop_rate
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C // C_reduction, stride, affine=False) # PC-channel reduction
            # if 'pool' in primitive:
            #     op = nn.Sequential(op, nn.BatchNorm2d(C // C_reduction, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        random_shuffle(x)
        
        channel = x.shape[1]
        x_temp_1 = x[ :, :channel//self.C_reduction, :, :]
        x_temp_2 = x[ :, channel//self.C_reduction:, :, :]

        if self.td_choice == 'unit':
            x = self.targeted_unit_dropout(x, weights)
        elif self.td_choice == 'weight':
            weights = self.targeted_weight_dropout(x, weights)
        else:
            raise ValueError("Targeted Dropout must be either 'unit' or 'weight'")

        sum_temp_1 = sum(w * op(x_temp_1) for w, op in zip(weights, self._ops))

        # reduction cell needs pooling before concat
        if sum_temp_1.shape[2] == x.shape[2]:
            result = torch.cat([sum_temp_1, x_temp_2], dim=1)
        else:
            result = torch.cat([sum_temp_1, self.max_pool(x_temp_2)], dim=1)
        # result = channel_shuffle(result, self.C_reduction)

        return result


    def targeted_weight_dropout(self, x, weights):
        """
        Implementation of mixedop targeted weight dropout
        Args:
            x: input
            weights: weight for each operation
        """
        if self.training and self.td_rate > 0. and self.drop_rate > 0.:
            norm        = torch.abs(weights)
            idx         = int(self.td_rate * int(weights.shape[0]))
            sorted_w, _ = torch.sort(norm)
            threshold   = sorted_w[idx]
            mask        = (weights < threshold)
            mask        = ((1. - self.drop_rate) < torch.rand(weights.shape[0])).cuda() * mask
            weights     = (~mask).type(torch.cuda.FloatTensor) * weights

        return weights


    def targeted_unit_dropout(self, x, weights):
        """
        Implementation of mixedop targeted unit dropout
        Args:
            x: input
            weights: weight for each operation
        """
        if self.training and self.td_rate > 0. and self.drop_rate > 0.:
            re_x            = x.view(-1, x.shape[-1]) # wrt columns
            norm            = torch.norm(re_x, dim=0)
            idx             = int(self.td_rate * int(re_x.shape[1]))
            sorted_norms, _ = torch.sort(norm)
            threshold       = sorted_norms[idx]
            mask            = (norm < threshold).expand([re_x.shape[0], -1])

            mask = ((1. - self.drop_rate) < torch.rand(re_x.shape)).cuda() * mask
            x = ((~mask).type(torch.cuda.FloatTensor) * re_x).view(x.shape)

        return x
