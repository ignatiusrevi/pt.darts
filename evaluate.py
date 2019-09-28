""" Evaluate models """
import time
import torch
import torch.nn as nn
import numpy as np
import utils
from IPython.core.debugger import set_trace
import genotypes as gt
from models import ops
# from config import EvaluateConfig

# Supernet
from models.search_cnn_cka  import SearchCNNControllerCKA
# ChildNet
from models.augment_cnn_cka import AugmentCNNCKA

import scripts.cka as cka
import scripts.cca as cca

device_ids = [0]
device = torch.device('cuda:0')

def main():
    # Breakdown the Composition of DARTS
    # childnet = torch.load('./results/02-random-beta-childnet/best.pth.tar', map_location=torch.device(device))
    # childnet.module.cells[0].dag[0][0]
    # supernet = torch.load('./results/02-random-beta-supernet/best.pth.tar', map_location=torch.device(device))
    # supernet.net.cells[0].dag[0][0]._ops[5].net[1]

    input_size, input_channels, n_classes, train_data = utils.get_data('CIFAR10', './data/', cutout_length=0, validation=False)

    net_crit = nn.CrossEntropyLoss().to(device)
    supernet = SearchCNNControllerCKA(C_in=input_channels, C=16, n_classes=n_classes, n_layers=8, criterion=net_crit, device_ids=device_ids)
    childnet = AugmentCNNCKA(input_size=input_size, C_in=input_channels, C=16, n_classes=10, n_layers=16, auxiliary=True, genotype=gt.from_str("Genotype(normal=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], [('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], [('sep_conv_3x3', 1), ('sep_conv_5x5', 0)], [('sep_conv_3x3', 0), ('sep_conv_5x5', 1)]], normal_concat=range(2, 6), reduce=[[('sep_conv_3x3', 1), ('max_pool_3x3', 0)], [('max_pool_3x3', 2), ('sep_conv_5x5', 0)], [('dil_conv_5x5', 3), ('sep_conv_5x5', 1)], [('max_pool_3x3', 4), ('sep_conv_5x5', 0)]], reduce_concat=range(2, 6))"))
    childnet = nn.DataParallel(childnet, device_ids=device_ids).to(device)

    utils.load(supernet, './results/02-random-beta-supernet/weights.pt')
    utils.load(childnet, './results/02-random-beta-l16-init16-childnet/weights.pt')
    supernet.to(device)
    childnet.to(device)

    supernet.eval()
    childnet.eval()

    n_cells = len(supernet.net.cells)
    n_ops   = 8
    for i in range(n_cells): # 8 Cells
        # Normal Cell
        if i not in [n_cells//3, 2*n_cells//3]:
            # DAG 0
            for k in range(2):
                for j in range(n_ops):
                    if j != 3: # sep_conv_3x3
                        supernet.net.cells[i].dag[0][k]._ops[j] = ops.Zero(stride=1)

            # DAG 1
            for k in range(2):
                for j in range(n_ops):
                    if j != 3: # sep_conv_3x3
                        supernet.net.cells[i].dag[1][k]._ops[j] = ops.Zero(stride=1)
            for j in range(n_ops):
                supernet.net.cells[i].dag[1][2]._ops[j]         = ops.Zero(stride=1)

            # DAG 2
            for j in range(n_ops):
                if j != 4: # sep_conv_5x5
                    supernet.net.cells[i].dag[2][0]._ops[j] = ops.Zero(stride=1)
            for j in range(n_ops):
                if j != 3: # sep_conv_3x3
                    supernet.net.cells[i].dag[2][1]._ops[j] = ops.Zero(stride=1)
            for k in range(2,4):
                for j in range(n_ops):
                    supernet.net.cells[i].dag[2][k]._ops[j] = ops.Zero(stride=1)

            # DAG 3
            for j in range(n_ops):
                if j != 3: # sep_conv_3x3
                    supernet.net.cells[i].dag[3][0]._ops[j] = ops.Zero(stride=1)
            for j in range(n_ops):
                if j != 4: # sep_conv_5x5
                    supernet.net.cells[i].dag[3][1]._ops[j] = ops.Zero(stride=1)
            for k in range(2,5):
                for j in range(n_ops):
                    supernet.net.cells[i].dag[3][k]._ops[j] = ops.Zero(stride=1)
        # Reduction Cell
        else: 
            # DAG 0
            for j in range(n_ops):
                if j != 1: # max_pool_3x3
                    supernet.net.cells[i].dag[0][0]._ops[j] = ops.Zero(stride=2)
            for j in range(n_ops):
                if j != 3: # sep_conv_3x3
                    supernet.net.cells[i].dag[0][1]._ops[j] = ops.Zero(stride=2)

            # DAG 1
            for j in range(n_ops):
                if j != 4: # sep_conv_5x5
                    supernet.net.cells[i].dag[1][0]._ops[j] = ops.Zero(stride=2)
            for j in range(n_ops):
                supernet.net.cells[i].dag[1][1]._ops[j]     = ops.Zero(stride=2)
            for j in range(n_ops):
                if j != 1: # max_pool_3x3
                    supernet.net.cells[i].dag[1][2]._ops[j] = ops.Zero(stride=1)

            # DAG 2
            for j in range (n_ops):
                if j != 4: # sep_conv_5x5
                    supernet.net.cells[i].dag[2][1]._ops[j] = ops.Zero(stride=2)
            for j in range(n_ops):
                if j != 6: # dil_conv_5x5
                    supernet.net.cells[i].dag[2][3]._ops[j] = ops.Zero(stride=1)
            w = [0, 2]
            for k in w:
                if k == 0:
                    stride = 2
                else:
                    stride = 1
                for j in range(n_ops):
                    supernet.net.cells[i].dag[2][k]._ops[j] = ops.Zero(stride=stride)

            # DAG 3
            for j in range(n_ops):
                if j != 4: # sep_conv_5x5
                    supernet.net.cells[i].dag[3][0]._ops[j] = ops.Zero(stride=2)
            for j in range(n_ops):
                if j != 1: # max_pool_3x3
                    supernet.net.cells[i].dag[3][4]._ops[j] = ops.Zero(stride=1)
            for k in range(1,4):
                if k == 1:
                    stride = 2
                else:
                    stride = 1
                for j in range(n_ops):
                    supernet.net.cells[i].dag[3][k]._ops[j] = ops.Zero(stride=stride)

    # Test Image Batch
    img = torch.cat([train_data[i][0].view(-1, 3, input_size, input_size) for i in range(64)], dim=0)
    img = img.to(device, non_blocking=True)

    # Evaluate Forward Pass
    with torch.no_grad():
        supernet(img) # len(supernet.net.outpts)
        childnet(img) # len(childnet.module.outputs)

        sout = [x.view(64,-1).cpu().numpy() for x in supernet.net.outputs]
        cout = [x.view(64,-1).cpu().numpy() for x in childnet.module.outputs]

        lcka_arr   = {}
        rbfcka_arr = {}
        lcka_arr_debiased = {}
        cca_arr    = {}

        for i, s in enumerate(sout):
            print('Itr:', i)
            lcka_tmp   = {}
            rbfcka_tmp = {}
            lcka_tmp_debiased = {}
            cca_tmp    = {}
            for j, c in enumerate(cout):
                if s.shape == c.shape:
                    # Linear CKA
                    lcka = cka.feature_space_linear_cka(s, c)
                    lcka_tmp[(i,j)] = round(lcka, 4)
                    # Non-Linear CKA
                    rbfcka = cka.cka(cka.gram_rbf(s, 0.5), cka.gram_rbf(c, 0.5))
                    rbfcka_tmp[(i,j)] = round(rbfcka, 4)
                    # Linear CKA Debiased
                    lcka_debiased = cka.feature_space_linear_cka(s, c, debiased=True)
                    lcka_tmp_debiased[(i,j)] = round(lcka_debiased, 4)
                    # CCA
                    rcca = cca.cca(s, c)
                    cca_tmp[(i,j)] = round(rcca, 4)
            lcka_arr[i]   = lcka_tmp
            rbfcka_arr[i] = rbfcka_tmp
            lcka_arr_debiased[i] = lcka_tmp_debiased
            cca_arr[i]    = cca_tmp

    print('Linear CKA: ', lcka_arr)
    print()
    print('RBF CKA: ',    rbfcka_arr)
    print()
    print('LinearCKAD: ', lcka_arr_debiased)
    print()
    print('CCA: ',        cca_arr)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print("Total Evaluation Time: %ds" % duration)
