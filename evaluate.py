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
    utils.load(childnet, './results/02-random-beta-l8-init16-64-childnet/weights.pt')
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
        cca_arr    = {}

        assert len(sout) == len(cout) == 8, "Invalid Length"

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
                    lcka_tmp[(i,j)] = lcka
                    # Non-Linear CKA
                    rbfcka = cka.cka(cka.gram_rbf(s, 0.5), cka.gram_rbf(c, 0.5))
                    rbfcka_tmp[(i,j)] = rbfcka
                    # Linear CKA Debiased
                    lcka_debiased = cka.feature_space_linear_cka(s, c, debiased=True)
                    lcka_tmp_debiased[(i,j)] = lcka_debiased
                    # CCA
                    rcca = cca.cca(s, c)
                    cca_tmp[(i,j)] = rcca
            lcka_arr[i]   = lcka_tmp
            rbfcka_arr[i] = rbfcka_tmp
            lcka_arr_debiased[i] = lcka_tmp_debiased
            cca_arr[i]    = cca_tmp

    '''
    CKA Supernet-Childnet(l8-96)

    Linear CKA:  {0: {(0, 0): 0.8676586279085245, (0, 1): 0.7565095329657149}, 
                  1: {(1, 0): 0.8661395744028203, (1, 1): 0.8426195938851828}, 
                  2: {(2, 2): 0.7425798581576286, (2, 3): 0.6855975175884164, (2, 4): 0.6316171799014005}, 
                  3: {(3, 2): 0.7501621356799125, (3, 3): 0.6914640705553381, (3, 4): 0.6298844593349988}, 
                  4: {(4, 2): 0.7145043912392024, (4, 3): 0.6597830729290587, (4, 4): 0.6116518868392024}, 
                  5: {(5, 5): 0.65794550719517, (5, 6): 0.590893061805165, (5, 7): 0.337006734763566}, 
                  6: {(6, 5): 0.6271008919353144, (6, 6): 0.567877113925489, (6, 7): 0.3200083821810128}, 
                  7: {(7, 5): 0.6327680617331396, (7, 6): 0.5670474095009975, (7, 7): 0.33072531128700555}}

    RBF CKA:  {0: {(0, 0): 0.9532823, (0, 1): 0.94264835}, 
               1: {(1, 0): 0.9640795, (1, 1): 0.96091485}, 
               2: {(2, 2): 0.93316513, (2, 3): 0.9163666, (2, 4): 0.9031203}, 
               3: {(3, 2): 0.93338275, (3, 3): 0.9175467, (3, 4): 0.90480024}, 
               4: {(4, 2): 0.93372726, (4, 3): 0.91679937, (4, 4): 0.9058656}, 
               5: {(5, 5): 0.92865086, (5, 6): 0.91035444, (5, 7): 0.6153425}, 
               6: {(6, 5): 0.9202162, (6, 6): 0.9026651, (6, 7): 0.6136792}, 
               7: {(7, 5): 0.92136294, (7, 6): 0.90367335, (7, 7): 0.61616653}}

    Linear CKA Debiased:  {0: {(0, 0): 0.885327339812747, (0, 1): 0.7154878791514093}, 
                           1: {(1, 0): 0.7443802828155672, (1, 1): 0.7884092163386466}, 
                           2: {(2, 2): 0.5550788548212844, (2, 3): 0.3650349685611187, (2, 4): 0.16435105797124883}, 
                           3: {(3, 2): 0.5666212075527898, (3, 3): 0.35166861541900796, (3, 4): 0.1387820541183383}, 
                           4: {(4, 2): 0.502686526984809, (4, 3): 0.3251720781505099, (4, 4): 0.14053831961919408}, 
                           5: {(5, 5): 0.35115299798192284, (5, 6): 0.22168325175527742, (5, 7): 0.10040204964218279}, 
                           6: {(6, 5): 0.32045882436561274, (6, 6): 0.21119623505173354, (6, 7): 0.09082882653199824}, 
                           7: {(7, 5): 0.3516602459526517, (7, 6): 0.22639508783100373, (7, 7): 0.11297295795243746}}

    CCA:  {0: {(0, 0): 0.0009765625, (0, 1): 0.0009765625}, 
           1: {(1, 0): 0.0009765625, (1, 1): 0.0009765625}, 
           2: {(2, 2): 0.0019531247671693633, (2, 3): 0.001953125, (2, 4): 0.001953125}, 
           3: {(3, 2): 0.0019531247671693633, (3, 3): 0.001953125, (3, 4): 0.0019531247671693633}, 
           4: {(4, 2): 0.001953125, (4, 3): 0.001953125, (4, 4): 0.001953125}, 
           5: {(5, 5): 0.00390625, (5, 6): 0.00390625, (5, 7): 0.00390625}, 
           6: {(6, 5): 0.00390625, (6, 6): 0.00390625, (6, 7): 0.00390625}, 
           7: {(7, 5): 0.00390625, (7, 6): 0.0039062495343387266, (7, 7): 0.00390625}}
    '''


    '''
    CKA Supernet-Childnet(l8-64)

    Linear CKA:  {0: {(0, 0): 0.8352792912331861, (0, 1): 0.8199967605418861}, 
                  1: {(1, 0): 0.8362026413701265, (1, 1): 0.8741494162047883}, 
                  2: {(2, 2): 0.7776490929971002, (2, 3): 0.6820612764001182, (2, 4): 0.6133510036788681}, 
                  3: {(3, 2): 0.7730648720125691, (3, 3): 0.6786249617607284, (3, 4): 0.6138471231296045}, 
                  4: {(4, 2): 0.760404721835937, (4, 3): 0.6725530999113611, (4, 4): 0.6103022170089057}, 
                  5: {(5, 5): 0.6656843741045557, (5, 6): 0.6121197475441963, (5, 7): 0.33010644344834544}, 
                  6: {(6, 5): 0.6372809345420658, (6, 6): 0.5881148595116308, (6, 7): 0.3158136454068329}, 
                  7: {(7, 5): 0.6316309979970047, (7, 6): 0.5890806087139465, (7, 7): 0.32243734769066607}}

    RBF CKA:  {0: {(0, 0): 0.9380381, (0, 1): 0.9417867}, 
               1: {(1, 0): 0.95218676, (1, 1): 0.96400076}, 
               2: {(2, 2): 0.93573093, (2, 3): 0.9173023, (2, 4): 0.89951605}, 
               3: {(3, 2): 0.9317879, (3, 3): 0.912457, (3, 4): 0.89612216}, 
               4: {(4, 2): 0.9354652, (4, 3): 0.91679776, (4, 4): 0.90077627}, 
               5: {(5, 5): 0.9303177, (5, 6): 0.91685504, (5, 7): 0.6462964}, 
               6: {(6, 5): 0.9223904, (6, 6): 0.9098717, (6, 7): 0.6423473}, 
               7: {(7, 5): 0.9212247, (7, 6): 0.9085942, (7, 7): 0.64660907}}

    Linear CKA Debiased:  {0: {(0, 0): 0.8622394920037112, (0, 1): 0.8036679381343795}, 
                           1: {(1, 0): 0.6993161229373346, (1, 1): 0.8078501244821658}, 
                           2: {(2, 2): 0.6780569735232216, (2, 3): 0.4436534227766567, (2, 4): 0.15887891450882968}, 
                           3: {(3, 2): 0.6387506329972906, (3, 3): 0.3863549546719028, (3, 4): 0.12575258646080664}, 
                           4: {(4, 2): 0.6542701516284081, (4, 3): 0.4547483357236482, (4, 4): 0.1827990811191325}, 
                           5: {(5, 5): 0.38348927002752803, (5, 6): 0.29931177002892134, (5, 7): 0.08945373807451043}, 
                           6: {(6, 5): 0.347074363982186, (6, 6): 0.2754397257276325, (6, 7): 0.08167289018615821}, 
                           7: {(7, 5): 0.3489929414329208, (7, 6): 0.28784922525737994, (7, 7): 0.09516807867790027}}

    CCA:  {0: {(0, 0): 0.0009765625, (0, 1): 0.0009765625}, 
           1: {(1, 0): 0.0009765625, (1, 1): 0.0009765625}, 
           2: {(2, 2): 0.001953125, (2, 3): 0.001953125, (2, 4): 0.001953125}, 
           3: {(3, 2): 0.001953125, (3, 3): 0.001953125, (3, 4): 0.001953125}, 
           4: {(4, 2): 0.001953125, (4, 3): 0.001953125, (4, 4): 0.001953125}, 
           5: {(5, 5): 0.00390625, (5, 6): 0.00390625, (5, 7): 0.00390625}, 
           6: {(6, 5): 0.00390625, (6, 6): 0.00390625, (6, 7): 0.00390625}, 
           7: {(7, 5): 0.00390625, (7, 6): 0.00390625, (7, 7): 0.00390625}}
    '''

    print('Linear CKA: ', lcka_arr)
    print()
    print('RBF CKA: ', rbfcka_arr)
    print()
    print('Linear CKA Debiased: ', lcka_arr_debiased)
    print()
    print('CCA: ', cca_arr)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print("Total Evaluation Time: %ds" % duration)
