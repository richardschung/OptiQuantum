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

def main():
    timer = TicToc()

    #Mesh size
    N = 4
    #Create component layers
    layers = []
    for i in range(N-1):
        if i % 2 == 0:
            #Create even component layer
            layers.append(MZIDelayLayer.from_waveguide_indices(MZIDelayLayer,i,N,list(range(N)),np.full(int(N/2),np.pi),np.full(int(N/2),np.pi),loss_dB=0.2))
        else:
            layers.append(MZIDelayLayer.from_waveguide_indices(MZIDelayLayer,i,N,list(range(1,N-1)),np.full(int(N/2)-1,np.pi),np.full(int(N/2)-1,np.pi),loss_dB=0.2))

    layers[0].mzis[0].theta = np.pi/2
    layers[0].mzis[0].phi = 0
    
    mesh = OpticalMeshNew(N,layers)
    
    timer.tic()

    res = 100
    taus = np.linspace(-5,5,res)
    tot_prob = np.zeros(res)
    
    for i in range(res):

        print(i)
        
        transfer = mesh.get_transfer_matrix()
        c11 = transfer[0,0]
        c12 = transfer[0,1]
        c21 = transfer[1,0]
        c22 = transfer[1,1]
        
        coinc = (np.abs(c12)**2)*(np.abs(c21)**2) + (np.abs(c11)**2)*(np.abs(c22)**2) + \
                (np.conjugate(c11)*np.conjugate(c22)*c12*c21 + np.conjugate(c12)*np.conjugate(c21)*c11*c22)*np.exp(-(taus[i])**2)
        prob_a = (np.abs(c12)**2)*(np.abs(c22)**2) + (np.abs(c11)**2)*(np.abs(c21)**2)*np.exp(-(taus[i])**2)
        prob_b = (np.abs(c11)**2)*(np.abs(c21)**2) + (np.abs(c12)**2)*(np.abs(c22)**2)*np.exp(-(taus[i])**2)
        
        tot_prob[i] = coinc + prob_a + prob_b

    fig, ax = plt.subplots()
    ax.plot(taus, tot_prob)

    timer.toc()

    plt.show()

    
if __name__ == '__main__':
    main()
