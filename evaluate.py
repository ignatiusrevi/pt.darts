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

    Linear CKA:  {0: {(0, 0): 0.8328535760767377, (0, 1): 0.7788916022689075}, 
                  1: {(1, 0): 0.8079274954285709, (1, 1): 0.7963110169057394}, 
                  2: {(2, 2): 0.7799838905546352, (2, 3): 0.663976403648518, (2, 4): 0.6002347976821842}, 
                  3: {(3, 2): 0.7634077638950499, (3, 3): 0.649255971609716, (3, 4): 0.5902767301160605}, 
                  4: {(4, 2): 0.7694462149344811, (4, 3): 0.643674621854729, (4, 4): 0.5815169127676262}, 
                  5: {(5, 5): 0.6395715289188378, (5, 6): 0.562095052666216, (5, 7): 0.3018609424642878}, 
                  6: {(6, 5): 0.7545886649030376, (6, 6): 0.679090813054925, (6, 7): 0.36491994642353776}, 
                  7: {(7, 5): 0.726663672848422,  (7, 6): 0.650431395970726, (7, 7): 0.3665040948360681}}

    RBF CKA:     {0: {(0, 0): 0.9406557, (0, 1): 0.9321029}, 
                  1: {(1, 0): 0.9326793, (1, 1): 0.93127054}, 
                  2: {(2, 2): 0.9248978, (2, 3): 0.8966035, (2, 4): 0.88344526}, 
                  3: {(3, 2): 0.9172881, (3, 3): 0.8884339, (3, 4): 0.876322}, 
                  4: {(4, 2): 0.9156036, (4, 3): 0.8836428, (4, 4): 0.87148094}, 
                  5: {(5, 5): 0.8944931, (5, 6): 0.8783169, (5, 7): 0.61382633}, 
                  6: {(6, 5): 0.9487136, (6, 6): 0.9337911, (6, 7): 0.6522991}, 
                  7: {(7, 5): 0.9379861, (7, 6): 0.9220003, (7, 7): 0.650306}}

    Linear CKA Debiased:  {0: {(0, 0): 0.9359392049790847, (0, 1): 0.8037465625658307}, 
                           1: {(1, 0): 0.8756431241037442, (1, 1): 0.8629399056679641}, 
                           2: {(2, 2): 0.7792649408593321, (2, 3): 0.5484618977769247,  (2, 4):  0.2678058515697928}, 
                           3: {(3, 2): 0.7339382696305976, (3, 3): 0.49177801238799773, (3, 4): 0.23629907345554466}, 
                           4: {(4, 2): 0.822311050188199,  (4, 3): 0.60326601797061,    (4, 4): 0.31574449832316454}, 
                           5: {(5, 5): 0.4290127223119876, (5, 6): 0.28897451322552664, (5, 7): 0.08535510059644492}, 
                           6: {(6, 5): 0.4662279444487991, (6, 6): 0.33444309445339276, (6, 7): 0.09053164013378423}, 
                           7: {(7, 5): 0.4592687860354769, (7, 6): 0.3260504393529013,  (7, 7): 0.11483459764259811}}

    CCA:         {0: {(0, 0): 0.0009765625, (0, 1): 0.0009765625}, 
                  1: {(1, 0): 0.0009765625, (1, 1): 0.0009765623835846816}, 
                  2: {(2, 2): 0.001953125,  (2, 3): 0.001953125, (2, 4): 0.001953125}, 
                  3: {(3, 2): 0.0019531247, (3, 3): 0.001953125, (3, 4): 0.001953125}, 
                  4: {(4, 2): 0.0019531247, (4, 3): 0.001953125, (4, 4): 0.001953125}, 
                  5: {(5, 5): 0.00390625,   (5, 6): 0.00390625,  (5, 7): 0.00390625}, 
                  6: {(6, 5): 0.00390625,   (6, 6): 0.00390625,  (6, 7): 0.00390625}, 
                  7: {(7, 5): 0.0039062495, (7, 6): 0.00390625,  (7, 7): 0.0039062495343387266}}
    '''

    '''
    CKA-l8-96

    Linear CKA:  {0: {(0, 0): 0.8657758977307071, (0, 1): 0.7252689215550284}, 
                  1: {(1, 0): 0.8491905398582124, (1, 1): 0.7578524777478035}, 
                  2: {(2, 2): 0.7715327715403418, (2, 3): 0.664866761124665,  (2, 4): 0.6182232825721212}, 
                  3: {(3, 2): 0.7685066624187927, (3, 3): 0.6538225713375204, (3, 4): 0.6158398752247314}, 
                  4: {(4, 2): 0.7628401630474201, (4, 3): 0.6426905348291893, (4, 4): 0.6026920845123085}, 
                  5: {(5, 5): 0.6493959906329707, (5, 6): 0.5794062915343754, (5, 7): 0.2629509687279048}, 
                  6: {(6, 5): 0.7629123911805326, (6, 6): 0.6952849142079661, (6, 7): 0.34566161292731185}, 
                  7: {(7, 5): 0.7432488804765783, (7, 6): 0.6713959271893811, (7, 7): 0.34862670452569067}}

    RBF CKA:     {0: {(0, 0): 0.9517884, (0, 1): 0.9338336}, 
                  1: {(1, 0): 0.9452221, (1, 1): 0.93357855}, 
                  2: {(2, 2): 0.9255855, (2, 3): 0.9008065, (2, 4): 0.88937455}, 
                  3: {(3, 2): 0.9201559, (3, 3): 0.8942654, (3, 4): 0.88470256}, 
                  4: {(4, 2): 0.9115309, (4, 3): 0.8820146, (4, 4): 0.8718834}, 
                  5: {(5, 5): 0.8978233, (5, 6): 0.8825315, (5, 7): 0.5962776}, 
                  6: {(6, 5): 0.95061713, (6, 6): 0.936476, (6, 7): 0.63451517}, 
                  7: {(7, 5): 0.94182336, (7, 6): 0.926734, (7, 7): 0.63344204}}

    Linear CKA Debiased:  {0: {(0, 0): 0.9503798217220855, (0, 1): 0.7627867280149697}, 
                           1: {(1, 0): 0.9223125310781016, (1, 1): 0.8295214486592853}, 
                           2: {(2, 2): 0.7934688849511303, (2, 3): 0.5123290981763224, (2, 4): 0.32550234550082374}, 
                           3: {(3, 2): 0.7908826173935941, (3, 3): 0.4858611836370285, (3, 4): 0.3234154170239845}, 
                           4: {(4, 2): 0.8445338415258139, (4, 3): 0.5499516001605483, (4, 4): 0.3737229761519269}, 
                           5: {(5, 5): 0.448804614528437,  (5, 6): 0.3006718396923722, (5, 7): 0.03556187650964222}, 
                           6: {(6, 5): 0.4773907133781866, (6, 6): 0.3294766452032794, (6, 7): 0.06415958478895567}, 
                           7: {(7, 5): 0.4791735467871810, (7, 6): 0.3258298547250243, (7, 7): 0.08517673774175248}}

    CCA:         {0: {(0, 0): 0.0009765625, (0, 1): 0.0009765625}, 
                  1: {(1, 0): 0.0009765625, (1, 1): 0.0009765625}, 
                  2: {(2, 2): 0.001953125,  (2, 3): 0.001953124767, (2, 4): 0.001953125}, 
                  3: {(3, 2): 0.0019531247, (3, 3): 0.001953124767, (3, 4): 0.001953125}, 
                  4: {(4, 2): 0.001953125,  (4, 3): 0.001953125,    (4, 4): 0.001953125}, 
                  5: {(5, 5): 0.00390625,   (5, 6): 0.00390625,     (5, 7): 0.00390625}, 
                  6: {(6, 5): 0.00390625,   (6, 6): 0.00390625,     (6, 7): 0.00390625}, 
                  7: {(7, 5): 0.00390625,   (7, 6): 0.00390625,     (7, 7): 0.0039062495343387266}}
    '''
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print("Total Evaluation Time: %ds" % duration)
