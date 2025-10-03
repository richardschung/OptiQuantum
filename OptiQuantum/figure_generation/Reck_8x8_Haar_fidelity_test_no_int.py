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
from strawberryfields.decompositions_new import triangular_symmetric

sys.path.append('../')
from convert_decomposition import strawberryfields_to_neuroptica_reck

sys.path.append('../../')
from neuroptica.layers_new import flipped_ReckLayer
from neuroptica.component_layers_new import MZIDelayLayer, PhaseShifterLayer

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

def fidelity(mesh, ideal = None):
    #Get ideal matrix U and lossy/uncertain matrix T
    if ideal is None:
        U = mesh.get_transfer_matrix(add_uncertainties=False, add_loss=False)
    else:
        U = ideal
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
    N = 8
    n_Haars = 1

    max_phase = 0.8
    step_size_phase = 0.1
    n_phase = int(max_phase/step_size_phase) + 1
    phases = np.linspace(0,max_phase,n_phase)

    fidels = np.zeros([n_phase,n_phase,n_Haars])

    timer.tic()

    for haar_num in range(n_Haars):

        #Make Haar-random matrix
        #matrix = scipy.stats.ortho_group.rvs(N)

        #Load Haar random matrix
        matrix = np.loadtxt(f'figures_set1/test_Haar_8x8.txt')

        (tlist, localv, discard) = triangular_symmetric(matrix)
        tlist = np.array(strawberryfields_to_neuroptica_reck(tlist, N)) 
        localv = np.log(localv)/1j

        sz = np.shape(tlist)
        phase_data = [(tlist[i,2],tlist[i,3]) for i in range(sz[0])]

        network = flipped_ReckLayer(N, phases=phase_data, include_phase_shifter_layer=True,shifter_phases=localv,phase_uncert=np.pi/32)
        mesh = network.mesh

        n_samples = 1000

        fidel_samples = np.zeros(n_samples)

        print(matrix)
        print(mesh.get_transfer_matrix())

        for layer_cur in mesh.layers:
            if isinstance(layer_cur,MZIDelayLayer):
                for mzi_cur in layer_cur:
                    mzi_cur.theta0 = mzi_cur.theta
                    mzi_cur.phi0 = mzi_cur.phi
                
        for i in range(n_phase):
            for j in range(n_phase):
                for sample in range(n_samples):
                    for layer_cur in mesh.layers:
                        for mzi_cur in layer_cur:
                            try:
                                mzi_cur.theta = mzi_cur.theta0 + phases[i]
                                mzi_cur.phi = mzi_cur.phi0 + phases[j]
                                mzi_cur.randomize_errors()
                            except AttributeError:
                                pass
                            
                    fidel_samples[sample] = fidelity(mesh, matrix)
                fidels[i,j,haar_num] = np.mean(fidel_samples)
                #print(fidels[i,j])

    fidels = np.mean(fidels,2)

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
    plt.pcolor(phases, phases, fidels.T, cmap=cmap, rasterized=True, vmin=0, vmax=1)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='minor', labelsize=tick_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    plt.xlabel(r'Mean Error in $\theta$ (rad)', fontsize=labels_size)
    plt.ylabel(r'Mean Error in $\phi$(rad)', fontsize=labels_size)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.set_label('Fidelity', fontsize=labels_size)
    # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
    plt.tight_layout()

    #plt.show()

    plt.savefig(f'figures_set1/fidelity_vs_mean_phases_Reck_8x8_1000samples_1Haars_unc_pi_32.png')

    #Save Data
    np.savetxt(f'figures_set1/fidelity_vs_mean_phases_Reck_8x8_1000samples_1Haars_unc_pi_32.txt',fidels)

    timer.toc()

    
if __name__ == '__main__':
    main()
