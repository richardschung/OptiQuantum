import strawberryfields as sf

from functools import reduce

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from pytictoc import TicToc
from strawberryfields.decompositions_new import rectangular_symmetric, triangular_symmetric, triangular_MZ, mach_zehnder, triangular, Ti, mach_zehnder_inv, T

sys.path.append('../../')
from neuroptica.layers_new import ClementsLayer,ReckLayer, flipped_ReckLayer
from neuroptica.component_layers_new import MZIDelayLayer, PhaseShifterLayer
from neuroptica.components_new import MZI_delay

def main():
    N = 8

    #Make Haar-random matrix
    #matrix = scipy.stats.ortho_group.rvs(N)
    matrix = np.zeros([N,N])
    matrix[0,:] = 1
    print(matrix)

    (tlist, localv, discard) = triangular_symmetric(matrix)
##    #Sanity check for triangular matrix
##    #Working combination: triangular,
##    print(tlist)
##    print(tlist[0])
##    #tlist = list(reversed(tlist))
##    print(localv)
##
##    #checkT = np.eye(4)
##    #checkTi = np.eye(4)
##    check = np.eye(4)
##    for t in tlist:
##        print(t[0])
##        #checkT = T(*t)@checkT
##        #checkTi = Ti(*t)@checkTi
##        #check = mach_zehnder_inv(*t)@check
##        check = mach_zehnder(*t)@check
##        #print(np.round(checkT-checkTi,2))
##    #check = checkT
##    print(matrix)
##    print(np.round(check,2))
##    print(np.round(np.abs(check)-np.abs(matrix),2))
##    check = (np.eye(N)*localv)@check
##    print(np.round(check,2))
##    print(np.round(np.abs(check-matrix),2))
##    return
    #Do not reverse
    print(tlist)
    #tlist = np.array(list(reversed(tlist)))
    #tlist = np.array(tlist)
    return

    #Rearrange tlist to be compatible with neuroptica
    tlist_straw = []
    tlist_idx = 0

    #Arrange tlist into list of diagonals
    #N-1 diagonals
    for i in range(N-1):
        diag_list = []
        #Each diagonal has i+1 elements
        for j in range(i+1):
            diag_list.append(tlist[tlist_idx])
            tlist_idx += 1
        tlist_straw.append(diag_list)

    #Convert diagonals to columns
    tlist_neur = []
    tlist_temp = tlist_straw.copy()

    #Take first t from each diagonal then remove previous column
    for i in range(2*N-2):
        #Reverse indices to reverse column direction
        for j in reversed(range(min(i,len(tlist_temp)))):
            if tlist_temp[j]:
                tlist_neur.append(tlist_temp[j][0])
                tlist_temp[j] = tlist_temp[j][1:]

    print(np.round(tlist,2))

    tlist = np.array(tlist_neur)
    print(np.round(tlist,2))
    print(localv)
    localv = np.log(localv)/1j
    print(np.exp(1j*localv))

##    ps_layer = PhaseShifterLayer(N)
##    for i in range(len(ps_layer.phase_shifters)):
##        ps_layer.phase_shifters[i].phi = localv[i]
    
    sz = np.shape(tlist)
    phase_data = [(tlist[i,2],tlist[i,3]) for i in range(sz[0])]

    network = flipped_ReckLayer(N, phases=phase_data, include_phase_shifter_layer=True,shifter_phases=localv)
##    np.insert(network.mesh.layers,0,ps_layer)
    for layer in network.mesh.layers:
        for mzi in layer:
            if isinstance(mzi,MZI_delay):
                print([mzi.m,mzi.n,mzi.theta,mzi.phi])

    print(matrix)
    print(np.round(np.abs(network.mesh.get_transfer_matrix()),2))
    print(np.round(network.mesh.get_transfer_matrix(),2))
    print(np.round(network.mesh.get_transfer_matrix()-matrix,2))

if __name__ == '__main__':
    main()
