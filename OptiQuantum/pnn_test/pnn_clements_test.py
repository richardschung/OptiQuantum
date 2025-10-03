import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from pytictoc import TicToc
from pnn.methods import decompose_clements, reconstruct_clements

sys.path.append('../../')
from neuroptica.layers_new import ClementsLayer

def main():
    N = 4
    matrix = np.eye(N)
    matrix[0,0] = 0
    matrix[0,1] = 1
    matrix[1,0] = 1
    matrix[1,1] = 0
    print('test1')
    [phis, thetas, alphas] = decompose_clements(matrix, block='mzi')
    print('test2')
    print(phis/np.pi)
    print('test3')
    sz = np.shape(phis)
    print(sz)
    phi_data = np.zeros(sz)
    print(np.shape(phi_data))
    print(phis[0:sz[0]:2])
    phi_data[0:int(np.ceil(sz[0]/2))] = phis[0:sz[0]:2]
    phi_data[int(np.ceil(sz[0]/2)):sz[0]] = phis[1:sz[0]:2]
    
    phis_flat = phi_data.T.flatten()
    thetas_flat = thetas.flatten()
    print(thetas_flat/np.pi)
    print(phis_flat/np.pi)
    print('test4')
    phase_data = [(thetas_flat[i]*2,phis_flat[i]) for i in range(len(phis_flat))]
    print(phase_data)
    network = ClementsLayer(N, phases=phase_data)
    print('test')
    print(np.round(np.diag(np.exp(1j*alphas))@network.mesh.get_transfer_matrix(),2))
    print(matrix)
    for layer in network.mesh.layers:
        for mzi in layer.mzis:
            print(np.round(mzi.get_transfer_matrix(),2))
    print(np.round(reconstruct_clements(phis,thetas,alphas, block='mzi'),2))
    print(np.exp(1j*alphas))

if __name__ == '__main__':
    main()
