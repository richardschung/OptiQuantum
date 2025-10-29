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
    N = 2
    #Create component layers
    layers = []
    for i in range(N-1):
        if i % 2 == 0:
            #Create even component layer
            layers.append(MZIDelayLayer.from_waveguide_indices(MZIDelayLayer,i,N,list(range(N)),np.full(int(N/2),np.pi),np.full(int(N/2),np.pi)))
        else:
            layers.append(MZIDelayLayer.from_waveguide_indices(MZIDelayLayer,i,N,list(range(1,N-1)),np.full(int(N/2)-1,np.pi),np.full(int(N/2)-1,np.pi)))
    mesh = OpticalMeshNew(N,layers)

    print(mesh.layers[0].mzis[0].loss)
    print(mesh.get_transfer_matrix())
    print(mesh.get_transfer_matrix(False,False))
    print(mesh.layers[0].mzis[0].loss_dB_cur)

if __name__ == '__main__':
    main()
