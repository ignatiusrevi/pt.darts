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

    utils.load(supernet, './results/02-random-beta-supernet/weights.pt')
    utils.load(childnet, './results/02-random-beta-l8-init16-64-childnet/weights.pt')
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

    print('Linear CKA: ',          lcka_arr)
    print()
    print('RBF CKA: ',             rbfcka_arr)
    print()
    print('Linear CKA Debiased: ', lcka_arr_debiased)
    print()
    print('CCA: ',                 cca_arr)

    '''
    CKA-l8-64

    Linear CKA:  {0: {(0, 0): 0.8221820902528569, (0, 1): 0.7959822037025007}, 
                  1: {(1, 0): 0.8274592257830808, (1, 1): 0.856917200966611}, 
                  2: {(2, 2): 0.6960931618581306, (2, 3): 0.6017430549911925, (2, 4): 0.5279309369217731}, 
                  3: {(3, 2): 0.67992665295559,   (3, 3): 0.5915330415381924, (3, 4): 0.5228086846494047}, 
                  4: {(4, 2): 0.6636434733118876, (4, 3): 0.56370938549373,   (4, 4): 0.4933790190065167}, 
                  5: {(5, 5): 0.5863233246388775, (5, 6): 0.5009669349294872, (5, 7): 0.25403426432396686}, 
                  6: {(6, 5): 0.5935558179477539, (6, 6): 0.5064339170887164, (6, 7): 0.26143514782747995}, 
                  7: {(7, 5): 0.6178393216272278, (7, 6): 0.5328352780599712, (7, 7): 0.2832606121358926}}

    RBF CKA:  {0: {(0, 0): 0.9369706,  (0, 1): 0.9349373}, 
               1: {(1, 0): 0.95096314, (1, 1): 0.95934284}, 
               2: {(2, 2): 0.9084164,  (2, 3): 0.88648003, (2, 4): 0.8691506}, 
               3: {(3, 2): 0.9019167,  (3, 3): 0.8798763,  (3, 4): 0.8631861}, 
               4: {(4, 2): 0.890997,   (4, 3): 0.86624753, (4, 4): 0.84993774}, 
               5: {(5, 5): 0.901176,   (5, 6): 0.8790282,  (5, 7): 0.61724025}, 
               6: {(6, 5): 0.90688753, (6, 6): 0.8844269,  (6, 7): 0.6222117}, 
               7: {(7, 5): 0.9143759,  (7, 6): 0.8927816,  (7, 7): 0.6306835}}

    Linear CKA Debiased:  {0: {(0, 0): 0.8763763306476804,  (0, 1): 0.8239008440347815}, 
                           1: {(1, 0): 0.7100704656489435,  (1, 1): 0.8157964081345358}, 
                           2: {(2, 2): 0.600659084237,      (2, 3): 0.3740293104134287,  (2, 4): 0.08814751148338276}, 
                           3: {(3, 2): 0.556521835394171,   (3, 3): 0.3403056447316868,  (3, 4): 0.07802932620781039}, 
                           4: {(4, 2): 0.6063326225368809,  (4, 3): 0.3851066458176257,  (4, 4): 0.10685310205549492}, 
                           5: {(5, 5): 0.2805799105278847,  (5, 6): 0.15534315720253075, (5, 7): 0.009402592064172726}, 
                           6: {(6, 5): 0.2873745231401529,  (6, 6): 0.15927681901056243, (6, 7): 0.0179668016805328}, 
                           7: {(7, 5): 0.31229564988876773, (7, 6): 0.18668740341191267, (7, 7): 0.038255377523626004}}

    CCA:  {0: {(0, 0): 0.0009765623835846816, (0, 1): 0.0009765625}, 
           1: {(1, 0): 0.0009765623835846816, (1, 1): 0.0009765623835846816}, 
           2: {(2, 2): 0.001953125,           (2, 3): 0.0019531247671693633, (2, 4): 0.001953125}, 
           3: {(3, 2): 0.001953125,           (3, 3): 0.0019531247671693633, (3, 4): 0.001953125}, 
           4: {(4, 2): 0.001953125,           (4, 3): 0.001953125,           (4, 4): 0.001953125}, 
           5: {(5, 5): 0.00390625,            (5, 6): 0.00390625,            (5, 7): 0.00390625}, 
           6: {(6, 5): 0.0039062495343387266, (6, 6): 0.00390625,            (6, 7): 0.00390625}, 
           7: {(7, 5): 0.0039062495343387266, (7, 6): 0.00390625,            (7, 7): 0.00390625}}
    '''

    '''
    CKA-l8-96

    Linear CKA:  {0: {(0, 0): 0.8469042209264193, (0, 1): 0.7355716258370412}, 
                  1: {(1, 0): 0.8486632334576409, (1, 1): 0.821867159214105}, 
                  2: {(2, 2): 0.6962152097370453, (2, 3): 0.611906655100392,  (2, 4): 0.5810545826842675}, 
                  3: {(3, 2): 0.6824830945133084, (3, 3): 0.6076439812995248, (3, 4): 0.5713601210879795}, 
                  4: {(4, 2): 0.6573661724222462, (4, 3): 0.5769111754630705, (4, 4): 0.5572163659938212}, 
                  5: {(5, 5): 0.6136534989500113, (5, 6): 0.5700009966022486, (5, 7): 0.28364539136465067}, 
                  6: {(6, 5): 0.6129652474518361, (6, 6): 0.5736528307071961, (6, 7): 0.28731120122181764}, 
                  7: {(7, 5): 0.635658388675098,  (7, 6): 0.5934457281852811, (7, 7): 0.3054343411135469}}

    RBF CKA:  {0: {(0, 0): 0.9424678,  (0, 1): 0.9299834}, 
               1: {(1, 0): 0.9533989,  (1, 1): 0.95156693}, 
               2: {(2, 2): 0.9033178,  (2, 3): 0.8854578,  (2, 4): 0.87429136}, 
               3: {(3, 2): 0.89078563, (3, 3): 0.8721595,  (3, 4): 0.8613814}, 
               4: {(4, 2): 0.89600295, (4, 3): 0.8775259,  (4, 4): 0.8680786}, 
               5: {(5, 5): 0.9133115,  (5, 6): 0.89925253, (5, 7): 0.60912174}, 
               6: {(6, 5): 0.9172234,  (6, 6): 0.9037878,  (6, 7): 0.6159175}, 
               7: {(7, 5): 0.9216837,  (7, 6): 0.9084512,  (7, 7): 0.6201703}}

    Linear CKA Debiased:  {0: {(0, 0): 0.9032852490846589, (0, 1): 0.7368490848913454}, 
                           1: {(1, 0): 0.7897397787871898, (1, 1): 0.7882532299751058}, 
                           2: {(2, 2): 0.6207442427472173, (2, 3): 0.3817807163404143,  (2, 4): 0.26469472188004245}, 
                           3: {(3, 2): 0.5891362638999464, (3, 3): 0.3828813610942579,  (3, 4): 0.25118219800364955}, 
                           4: {(4, 2): 0.5856954677904356, (4, 3): 0.36914745745920396, (4, 4): 0.283373813768603}, 
                           5: {(5, 5): 0.3171694315834956, (5, 6): 0.23697598364630196, (5, 7): 0.053035873477024334}, 
                           6: {(6, 5): 0.3085984519993863, (6, 6): 0.23795202890617095, (6, 7): 0.05649286582743998}, 
                           7: {(7, 5): 0.3329098555948916, (7, 6): 0.25615046728325896, (7, 7): 0.07150357253388395}}

    CCA:  {0: {(0, 0): 0.0009765625, (0, 1): 0.0009765625}, 
           1: {(1, 0): 0.0009765625, (1, 1): 0.0009765623835846816}, 
           2: {(2, 2): 0.001953125,  (2, 3): 0.001953125,           (2, 4): 0.001953125}, 
           3: {(3, 2): 0.001953125,  (3, 3): 0.001953125,           (3, 4): 0.001953125}, 
           4: {(4, 2): 0.001953125,  (4, 3): 0.0019531247671693633, (4, 4): 0.001953125}, 
           5: {(5, 5): 0.00390625,   (5, 6): 0.00390625,            (5, 7): 0.00390625}, 
           6: {(6, 5): 0.00390625,   (6, 6): 0.00390625,            (6, 7): 0.00390625}, 
           7: {(7, 5): 0.00390625,   (7, 6): 0.00390625,            (7, 7): 0.00390625}}
    '''

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print("Total Evaluation Time: %ds" % duration)
