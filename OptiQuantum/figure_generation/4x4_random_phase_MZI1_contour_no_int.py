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
from neuroptica.component_layers_new import MZIDelayLayer, OpticalMesh

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
    mesh = OpticalMesh(N,layers)
    
    timer.tic()

    max_phase = 0.8
    step_size_phase = 0.1
    n_phase = int(max_phase/step_size_phase) + 1
    phases = np.linspace(0,max_phase,n_phase)

    visibs = np.zeros([n_phase,n_phase])

    n = 0 #MZI number

    n_samples = 1000

    visib_samples = np.zeros(n_samples)

    mesh.layers[0].mzis[0].theta = np.pi/2 #Set MZI under test to 50:50 BS
    mesh.layers[0].mzis[0].phi = 0
    test_m = mesh.layers[0].mzis[0].m
    test_n = mesh.layers[0].mzis[0].n
    
    for i in range(n_phase):
        for j in range(n_phase):
            for sample in range(n_samples):
                for layer_cur in mesh.layers:
                    for mzi_cur in layer_cur:
                        mzi_cur.phase_uncert_theta = phases[i]
                        mzi_cur.phase_uncert_phi = phases[j]
                        mzi_cur.randomize_errors()

                visib_samples[sample] = visibility(mesh, test_m, test_n, test_m, test_n)

            visibs[i,j] = np.mean(visib_samples)
            #print(visibs[i,j])

    cmap='gist_heat'

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
    plt.pcolor(phases, phases, visibs.T, cmap=cmap, rasterized=True, vmin=0, vmax=1)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='minor', labelsize=tick_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    plt.xlabel(r'$\theta$ Uncertainty (rad)', fontsize=labels_size)
    plt.ylabel(r'$\phi$ Uncertainty (rad)', fontsize=labels_size)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.set_label('Visibility', fontsize=labels_size)
    # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
    plt.tight_layout()

    plt.savefig(f'figures_set1/4x4_random_phase_Clements_no_int_MZI_1_contour_1000_samples.png')
    #plt.show()

    timer.toc()

    
if __name__ == '__main__':
    main()
