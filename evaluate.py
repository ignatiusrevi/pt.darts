""" Evaluate models """
import time
import torch
import torch.nn as nn
import numpy as np
import utils
from IPython.core.debugger import set_trace
import genotypes as gt
# from config import EvaluateConfig

# Supernet
from models.search_cnn_cka  import SearchCNNControllerCKA
# ChildNet
from models.augment_cnn_cka import AugmentCNNCKA

import scripts.cka as cka
import scripts.cca as cca

device_ids = [3]
device = torch.device('cuda:3')

def main():
    # Breakdown the Composition of DARTS
    # childnet = torch.load('./results/02-random-beta-childnet/best.pth.tar', map_location=torch.device(device))
    # childnet.module.cells[0].dag[0][0]
    # supernet = torch.load('./results/02-random-beta-supernet/best.pth.tar', map_location=torch.device(device))
    # supernet.net.cells[0].dag[0][0]._ops[5].net[1]

    input_size, input_channels, n_classes, train_data = utils.get_data('CIFAR10', './data/', cutout_length=0, validation=False)

    net_crit = nn.CrossEntropyLoss().to(device)
    supernet = SearchCNNControllerCKA(C_in=input_channels, C=16, n_classes=n_classes, n_layers=8, criterion=net_crit, device_ids=device_ids)
    childnet = AugmentCNNCKA(input_size=input_size, C_in=input_channels, C=16, n_classes=10, n_layers=8, auxiliary=True, genotype=gt.from_str("Genotype(normal=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], [('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], [('sep_conv_3x3', 1), ('sep_conv_5x5', 0)], [('sep_conv_3x3', 0), ('sep_conv_5x5', 1)]], normal_concat=range(2, 6), reduce=[[('sep_conv_3x3', 1), ('max_pool_3x3', 0)], [('max_pool_3x3', 2), ('sep_conv_5x5', 0)], [('dil_conv_5x5', 3), ('sep_conv_5x5', 1)], [('max_pool_3x3', 4), ('sep_conv_5x5', 0)]], reduce_concat=range(2, 6))"))
    childnet = nn.DataParallel(childnet, device_ids=device_ids).to(device)

    utils.load(supernet,    './results/02-random-beta-supernet/weights.pt')
    utils.load(childnet, './results/02-random-beta-l8-init16-childnet/weights.pt')
    supernet.to(device)
    childnet.to(device)

    supernet.eval()
    childnet.eval()

    # Test Image Batch
    img = torch.cat([train_data[i][0].view(-1, 3, input_size, input_size) for i in range(64)], dim=0)
    img = img.to(device, non_blocking=True)

    # Evaluate Forward Pass
    with torch.no_grad():
        supernet(img) #  len(supernet.net.outpts)
        childnet(img) # len(childnet.module.outputs)

        sout = [x.view(64,-1).cpu().numpy() for x in supernet.net.outputs]
        cout = [x.view(64,-1).cpu().numpy() for x in childnet.module.outputs]

        lcka_arr   = {}
        rbfcka_arr = {}
        lcka_arr_debiased = {}
        cca_arr = {}

        assert len(sout) == len(cout) == 8, "Invalid Length"

        for i, s in enumerate(sout):
            lcka_tmp   = {}
            rbfcka_tmp = {}
            lcka_tmp_debiased = {}
            cca_tmp = {}
            for j, c in enumerate(cout):
                if s.shape == c.shape:
                    # Linear CKA
                    lcka = cka.feature_space_linear_cka(s, c)
                    lcka_tmp[(i,j)] = lcka
                    # Non-Linear CKA
                    rbfcka = cka.cka(cka.gram_rbf(s, 0.5), cka.gram_rbf(c, 0.5))
                    rbfcka_arr[(i,j)] = rbfcka
                    # Linear CKA Debiased
                    lcka_debiased = cka.feature_space_linear_cka(s, c, debiased=True)
                    lcka_tmp_debiased[(i,j)] = lcka_debiased
                    # CCA
                    cca = cca.cca(s, c)
                    cca_tmp[(i,j)] = cca
            lcka_arr[i]   = lcka_tmp
            rbfcka_arr[i] = rbfcka_tmp
            lcka_arr_debiased[i] = lcka_tmp_debiased
            cca_arr[i] = cca_tmp

    # lcka_arr: [0.8647181751790686, 0.8420941606313009, 0.7348263199116554, 0.6825569398218329, 0.6052602758501613, 0.6565296056176754, 0.5687178121624982, 0.32996000264800635]

    print('LCKA_ARR: ',   lcka_arr)
    print()
    print('RBFCKA_ARR: ', rbfcka_arr)
    print()
    print('LCKA_ARR_DEBIASED: ', lcka_arr_debiased)
    print()
    print('CCA_ARR: ', cca_arr)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print("Total Evaluation Time: %ds", duration)
