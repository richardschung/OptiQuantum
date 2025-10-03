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

def transmittance(mesh, idx_in1, idx_in2, idx_out1, idx_out2):
        transfer = mesh.get_transfer_matrix()
        c11 = transfer[idx_in1,idx_out1]
        c12 = transfer[idx_in1,idx_out2]
        c21 = transfer[idx_in2,idx_out1]
        c22 = transfer[idx_in2,idx_out2]
        
        coinc = (np.abs(c12)**2)*(np.abs(c21)**2) + (np.abs(c11)**2)*(np.abs(c22)**2) + \
                (np.conjugate(c11)*np.conjugate(c22)*c12*c21 + np.conjugate(c12)*np.conjugate(c21)*c11*c22)
        prob_a = (np.abs(c12)**2)*(np.abs(c22)**2) + (np.abs(c11)**2)*(np.abs(c21)**2)
        prob_b = (np.abs(c11)**2)*(np.abs(c21)**2) + (np.abs(c12)**2)*(np.abs(c22)**2)
        
        tot_prob = coinc + prob_a + prob_b
        return tot_prob

def main():
    timer = TicToc()

    #Mesh size
    N = 4

    n_Haars = 5

    max_dB = 0.2
    step_size_dB = 0.01
    n_dB = int(max_dB/step_size_dB) + 1
    losses_dB = np.linspace(0,max_dB,n_dB)
    
    transmittances = np.zeros([n_dB,int(N*(N-1)/2),n_Haars])
    
    timer.tic()

    for haar_num in range(n_Haars):

        #Make Haar-random matrix
        matrix = scipy.stats.ortho_group.rvs(N)

        (tlist, localv, discard) = rectangular_symmetric(matrix)
        tlist = np.array(tlist)
        localv = np.log(localv)/1j

        sz = np.shape(tlist)
        phase_data = [(tlist[i,2],tlist[i,3]) for i in range(sz[0])]

        network = ClementsLayer(N, phases=phase_data, include_phase_shifter_layer=True,shifter_phases=localv,loss_dB=0.5)
        mesh = network.mesh

        n_samples = 1000

        transmittance_samples = np.zeros(n_samples)
                
        for i in range(n_dB):
            for j in range(n_dB):
                for sample in range(n_samples):
                    for layer_cur in mesh.layers:
                        for mzi_cur in layer_cur:
                            if isinstance(mzi_cur,MZI_delay):
                                mzi_cur.loss_diff = losses_dB[i]
                                mzi_cur.randomize_errors()
                                if mzi_cur.loss_dB_cur < 0:
                                    mzi_cur.loss_dB_cur = 0
                            
                    transmittance_samples[sample] = transmittance(mesh)
                transmittances[i,j,haar_num] = np.mean(transmittance_samples)
                #print(fidels[i,j])

    transmittances = np.mean(transmittances,2)

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
    plt.plot(losses_dB,transmittances.T[0],".-",label="MZI 1")
    plt.plot(losses_dB,transmittances.T[1],"s-",label="MZI 2")
    plt.plot(losses_dB,transmittances.T[2],"o-",label="MZI 3")
    plt.plot(losses_dB,transmittances.T[3],"^-",label="MZI 4")
    plt.plot(losses_dB,transmittances.T[4],"v-",label="MZI 5")
    plt.plot(losses_dB,transmittances.T[5],"<-",label="MZI 6")
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='minor', labelsize=tick_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    plt.xlabel('Standard Deviation of Loss(dB)', fontsize=labels_size)
    plt.ylabel(r'Transmittance', fontsize=labels_size)
    ax.legend()
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=tick_size)
    #cbar.set_label('Fidelity', fontsize=labels_size)
    # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
    plt.tight_layout()
    #plt.show()


    plt.savefig(f'figures_set1/transmittance_vs_random_loss_Reck_1000_samples_5Haars_mean_loss_05.png')

    #Save Data
    np.savetxt(f'figures_set1/transmittance_vs_random_phases_Reck_1000samples_5Haars_mean_loss_05.txt',transmittances)
    timer.toc()

    
if __name__ == '__main__':
    main()
