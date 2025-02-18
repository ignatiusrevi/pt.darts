""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size (1/2) and double (2x) channels in 1/3 and 2/3 layer.
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, weights_normal, weights_reduce, beta_normal, beta_reduce):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            # weights = weights_reduce if cell.reduction else weights_normal
            if cell.reduction:
                weights = weights_reduce
                n, start = 3, 2
                # beta_weights = F.softmax(beta_reduce[0:2], dim=-1)
                beta_weights = [F.softmax(beta, dim=-1) for beta in beta_reduce[0:2]]
                for i in range(self.n_nodes-1):
                    end = start + n
                    # tw2 = F.softmax(beta_reduce[start:end], dim=-1)
                    tw2 = [F.softmax(beta, dim=-1) for beta in beta_reduce[start:end]]
                    start = end
                    n += 1
                    # beta_weights = torch.cat([beta_weights, tw2], dim=0)
                    beta_weights = beta_weights + tw2
            else:
                weights = weights_normal
                n, start = 3, 2
                # beta_weights = F.softmax(beta_normal[0:2], dim=-1)
                beta_weights = [F.softmax(beta, dim=-1) for beta in beta_normal[0:2]]
                for i in range(self.n_nodes-1):
                    end = start + n
                    # tw2 = F.softmax(beta_normal[start:end], dim=-1)
                    tw2 = [F.softmax(beta, dim=-1) for beta in beta_normal[start:end]]
                    start = end
                    n += 1
                    # beta_weights = torch.cat([beta_weights, tw2], dim=0)
                    beta_weights = beta_weights + tw2
            s0, s1 = s1, cell(s0, s1, weights, beta_weights)
        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=3,
                 device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas, betas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()
        # PC-channel reduction - beta parameters
        self.beta_normal = nn.ParameterList()
        self.beta_reduce = nn.ParameterList()

        # for each node in a cell (identical no. for both normal and reduction cells)
        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            # PC-channel reduction - beta parameters
            self.beta_normal.append(nn.Parameter(1e-3*torch.randn(i+2)))
            self.beta_reduce.append(nn.Parameter(1e-3*torch.randn(i+2)))

        # setup alphas, betas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n or 'beta' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        # Single-GPU support
        if len(self.device_ids) == 1:
            return self.net(x, weights_normal, weights_reduce, self.beta_normal, self.beta_reduce)

        # Multi-GPU support
        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies, wreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)
    
    def print_beta(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### BETA #######")
        logger.info("# Beta - normal")
        for beta in self.beta_normal:
            logger.info(F.softmax(beta, dim=-1))

        logger.info("\n# Beta - reduce")
        for beta in self.beta_reduce:
            logger.info(F.softmax(beta, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        n, start = 3, 2
        weightsr2 = [F.softmax(beta, dim=-1) for beta in self.beta_reduce[0:2]]
        weightsn2 = [F.softmax(beta, dim=-1) for beta in self.beta_normal[0:2]]

        for i in range(self.n_nodes):
            end = start + n
            tw2 = [F.softmax(beta, dim=-1) for beta in self.beta_reduce[start:end]]
            tn2 = [F.softmax(beta, dim=-1) for beta in self.beta_normal[start:end]]
            start = end
            n += 1
            weightsr2 = weightsr2 + tw2
            weightsn2 = weightsn2 + tn2

        # normalize edge-leve beta parameters
        gene_normal = gt.parse(self.alpha_normal, weightsn2, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, weightsr2, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
