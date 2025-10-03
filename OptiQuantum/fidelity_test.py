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

sys.path.append('../')
from neuroptica.component_layers_new import MZIDelayLayer, OpticalMeshNew

def visibility(mesh, idx_in1, idx_in2, idx_out1, idx_out2):
    '''Function for the integrand used in the HOM interferometry simulation.
    :param mesh: The Neuroptica mesh under simulation
    :param idx_in1: Waveguide index of the first input path
    :param idx_in2: Waveguide index of the second input path
    :param idx_out1: Waveguide index of the first output path
    :param idx_out2: Waveguide index of the second output path
    '''
    #Get transfer matrix
    transfer = mesh.get_transfer_matrix()

    #Evaluate visiblity
    term1 = transfer[idx_in1,idx_out2]*(transfer[idx_in1,idx_out2].conj())*transfer[idx_in2,idx_out1]*(transfer[idx_in2,idx_out1].conj())
    term2 = transfer[idx_in1,idx_out1]*(transfer[idx_in1,idx_out1].conj())*transfer[idx_in2,idx_out2]*(transfer[idx_in2,idx_out2].conj())
    term3 = (transfer[idx_in1,idx_out1].conj())*(transfer[idx_in2,idx_out2].conj())*transfer[idx_in1,idx_out2]*transfer[idx_in2,idx_out1]
    term4 = (transfer[idx_in1,idx_out2].conj())*(transfer[idx_in2,idx_out1].conj())*transfer[idx_in1,idx_out1]*transfer[idx_in2,idx_out2]

    coinc_max = term1 + term2
    coinc_min = term1 + term2 + term3 + term4
    
    visib = (coinc_max-coinc_min)/coinc_max
    
    return visib

def fidelity(mesh):
    #Get ideal matrix U and lossy/uncertain matrix T
    U = mesh.get_transfer_matrix(add_uncertainties=False, add_loss=False)
    T = mesh.get_transfer_matrix()

    #Calculate hermitian conjugates
    U_H = U.conj().T
    T_H = T.conj().T

    print(np.round(T,2))
    print(np.round(np.abs(T),2))
    print(np.trace(U_H@T))
    print(np.round(U_H@T,2))
    print(np.trace(U_H@U))
    print(np.round(U_H@U,2))
    print(np.trace(T_H@T))
    print(np.round(T_H@T,2))

    #Calculate fidelity
    fidel = abs(np.trace(U_H@T)/np.sqrt(np.trace(U_H@U)*np.trace(T_H@T)))**2
    return fidel

def main():
    timer = TicToc()

    #Mesh size
    N = 4
    #Create component layers
    layers = []
    for i in range(N-1):
        if i % 2 == 0:
            #Create even component layer
            layers.append(MZIDelayLayer.from_waveguide_indices(MZIDelayLayer,i,N,list(range(N)),np.full(int(N/2),np.pi),np.full(int(N/2),np.pi)))
        else:
            layers.append(MZIDelayLayer.from_waveguide_indices(MZIDelayLayer,i,N,list(range(1,N-1)),np.full(int(N/2)-1,np.pi),np.full(int(N/2)-1,np.pi)))
    mesh = OpticalMeshNew(N,layers)
    
    mesh.layers[0].mzis[0].theta = np.pi/2 #Set MZI under test to 50:50 BS
    mesh.layers[0].mzis[0].phi = 0

    for layer_cur in mesh.layers:
        for mzi_cur in layer_cur:
            mzi_cur.loss_dB_cur = -10*np.log10(0.5)

    print(fidelity(mesh))
    
if __name__ == '__main__':
    main()
