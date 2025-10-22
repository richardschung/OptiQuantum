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

sys.path.append('../../')
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
    
    timer.tic()

    max_dB = 0.1
    step_size_dB = 0.01
    n_dB = int(max_dB/step_size_dB) + 1
    losses_dB = np.linspace(0,max_dB,n_dB)

    fidels = np.zeros([n_dB,5])

    n = 0 #MZI number

#    for layer in mesh.layers:
#            for mzi in layer:
#                print(np.round(mzi.get_transfer_matrix(),2))    

    n_samples = 1000

    fidel_samples = np.zeros(n_samples)
    
    for layer in mesh.layers:
        for mzi in layer:

            n += 1

            mzi.theta = np.pi/2 #Set MZI under test to 50:50 BS
            mzi.phi = 0
            
            for i in range(n_dB):
                for sample in range(n_samples):
                    for layer_cur in mesh.layers:
                        for mzi_cur in layer_cur:
                            mzi_cur.loss_diff = losses_dB[i]
                            mzi_cur.randomize_errors()

                    fidel_samples[sample] = fidelity(mesh)
    
                fidels[i,n-1] = np.mean(fidel_samples)
                #print(fidels[i,j])

            mzi.theta = np.pi #Restore bar state to MZI under test
            mzi.phi = np.pi

    #Plotting code from ONN_Simulation_Class.py
    labels_size = 30
    legend_size = 30
    tick_size = 28
    contour_color = (0.36, 0.54, 0.66)
    contour_color2 = 'black'
    contour_linewidth = 3.5
    tick_fmt = '%.2f'
    # plt.rcParams['font.family'] = 'STIXGeneral'
    # rc('font', weight='bold',**{'family':'serif','serif':['Times New Roman']})
    # rc('text', usetex=True)
    # the above settings has no effect... has to use preamble to change fonts
    rc('text.latex', preamble=r'\usepackage{mathptmx}')

    # Plot Loss + Phase uncert accuracies
    # plt.pcolor(self.loss_dB, self.phase_uncert_theta, self.accuracy_LPU, vmin=100/(self.N+1)*0, vmax=100, cmap=cmap, rasterized=True)
    plt.figure(figsize=(6.95, 5.03)) # compress the graph (around) quarter in size, by cutting top half and compress horizontally
    plt.plot(losses_dB,fidels.T[0],".-",label="MZI 1")
    plt.plot(losses_dB,fidels.T[1],"s-",label="MZI 2")
    plt.plot(losses_dB,fidels.T[2],"o-",label="MZI 3")
    plt.plot(losses_dB,fidels.T[3],"^-",label="MZI 4")
    plt.plot(losses_dB,fidels.T[4],"v-",label="MZI 5")
    
    ax = plt.gca()
    ax.set_ylim([0,1])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='minor', labelsize=tick_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    plt.xlabel('Loss Uncertainty (dB)', fontsize=labels_size)
    plt.ylabel(r'Fidelity', fontsize=labels_size)
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=tick_size)
    #cbar.set_label('Fidelity', fontsize=labels_size)
    # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
    plt.tight_layout()
    #plt.show()

    plt.savefig(f'figures_set1/fidelity_vs_random_loss_1000samples_big.png')

    timer.toc()
    
if __name__ == '__main__':
    main()
