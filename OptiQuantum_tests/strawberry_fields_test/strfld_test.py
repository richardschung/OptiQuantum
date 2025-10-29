import strawberryfields as sf

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from pytictoc import TicToc
from strawberryfields.decompositions_new import rectangular_symmetric, triangular_MZ

sys.path.append('../../')
from neuroptica.layers_new import ClementsLayer,ReckLayer, flipped_ReckLayer
from neuroptica.components_new import MZI_delay

def main():

    N = 6
    
    #Make Haar-random matrix
    matrix = scipy.stats.ortho_group.rvs(N)

    (tlist, localv, discard) = rectangular_symmetric(matrix)
    tlist = np.array(tlist)

    #Rearrange tlist to be compatible with neuroptica
    tlist_straw = []
    tlist_idx = 0

    #Arrange tlist into list of diagonals
    #N-1 diagonals
    for i in range(N-1):
        diag_list = []
        #Each diagonal up to N/2 has 2*i+1 elements
        if i < N/2:
            for j in range(2*i+1):
                diag_list.append(tlist[tlist_idx])
                tlist_idx += 1
        #Each diagonal after N/2 has 2*((N-1)-i) elements 
        else:
            for j in range(2*((N-1)-i)):
                diag_list.append(tlist[tlist_idx])
                tlist_idx += 1
        tlist_straw.append(diag_list)

    #Convert diagonals to columns
    tlist_neur = []
    tlist_temp = tlist_straw.copy()

    #Take first t from each diagonal then remove previous column
    for i in range(N):
        #Index of entry within column, not index of column
        col_idx = 0
        #Length of column is different for even and odd columns
        if i % 2 == 0:
            col_len = N/2
        else:
            col_len = N/2 - 1

        #Index of diagonal within matrix
        diag_num = 0

        #Remove first entries of first col_len non-empty diagonals
        while col_idx < col_len:
            if tlist_temp[diag_num]:
                tlist_neur.append(tlist_temp[diag_num][0])
                tlist_temp[diag_num] = tlist_temp[diag_num][1:]
                col_idx += 1
            diag_num += 1

    print(np.round(tlist,2))

    tlist = np.array(tlist_neur)
    print(np.round(tlist,2))
    print(localv)
    localv = np.log(localv)/1j
    print(np.exp(1j*localv))
    
    
    sz = np.shape(tlist)
    phase_data = [(tlist[i,2],tlist[i,3]) for i in range(sz[0])]

    network = ClementsLayer(N, phases=phase_data, include_phase_shifter_layer=True,shifter_phases=localv)
    for layer in network.mesh.layers:
        for mzi in layer:
            if isinstance(mzi,MZI_delay):
                print((mzi.m,mzi.theta))
    print(matrix)
    print(np.round(np.abs(network.mesh.get_transfer_matrix()),2))
    print(np.round(network.mesh.get_transfer_matrix(),2))
    print(np.round(network.mesh.get_transfer_matrix()-matrix,2))

if __name__ == '__main__':
    main()
