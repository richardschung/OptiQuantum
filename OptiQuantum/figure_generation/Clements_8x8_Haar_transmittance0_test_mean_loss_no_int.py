'''Program to test HOM interferometry simulation using Neuroptica layers.
Current layer to test is an MZI mesh with all components in bar state.
'''
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from pytictoc import TicToc
from strawberryfields.decompositions_new import rectangular_symmetric

sys.path.append('../../')
from neuroptica.layers_new import ClementsLayer
from neuroptica.component_layers_new import MZIDelayLayer, PhaseShifterLayer
from neuroptica.components_new import MZI_delay

sys.path.append('../')
from convert_decomposition import strawberryfields_to_neuroptica_clements
from metrics import fidelity, visibility, transmittance

def main():
    timer = TicToc()

    #Mesh size
    N = 8

    n_Haars = 1
    
    transmittances = np.zeros([N-1,n_Haars])
    out1 = 0
    out2 = 1
##    out1 = N-2
##    out2 = N-1
    
    timer.tic()

    for haar_num in range(n_Haars):

        #Make Haar-random matrix
        #matrix = scipy.stats.ortho_group.rvs(N)

        #Load Haar random matrix
        matrix = np.loadtxt(f'figures_set1/test_Haar_8x8.txt')

        (tlist, localv, discard) = rectangular_symmetric(matrix)
        tlist = strawberryfields_to_neuroptica_clements(tlist,N)
        tlist = np.array(tlist)
        localv = np.log(localv)/1j

        sz = np.shape(tlist)
        phase_data = [(tlist[i,2],tlist[i,3]) for i in range(sz[0])]

        network = ClementsLayer(N, phases=phase_data, include_phase_shifter_layer=True,shifter_phases=localv,loss_diff=0.02)
        mesh = network.mesh
        
        for n in range(N-1):
            transmittances[n,haar_num] = transmittance(mesh, n, n+1, out1, out2)
            #print(fidels[i,j])

    #Save Data
    np.savetxt(f'figures_set1/transmittance_1_2_no_err_Clements_1Haar_new_loss.txt',transmittances)
    timer.toc()

    
if __name__ == '__main__':
    main()
