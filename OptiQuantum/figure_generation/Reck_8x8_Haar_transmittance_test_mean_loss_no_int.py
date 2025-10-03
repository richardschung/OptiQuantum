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

sys.path.append('../../')
from neuroptica.layers_new import flipped_ReckLayer
from neuroptica.component_layers_new import MZIDelayLayer, PhaseShifterLayer
from neuroptica.components_new import MZI_delay

sys.path.append('../')
from convert_decomposition import strawberryfields_to_neuroptica_reck
from metrics import fidelity, visibility, transmittance

def main():
    timer = TicToc()

    #Mesh size
    N = 8

    n_Haars = 1

    max_dB = 0.5
    step_size_dB = 0.05
    n_dB = int(max_dB/step_size_dB) + 1
    losses_dB = np.linspace(0,max_dB,n_dB)
    
    transmittances = np.zeros([n_dB,N-1,n_Haars])
##    out1 = 0
##    out2 = 1
    out1 = N-2
    out2 = N-1
    
    timer.tic()

    for haar_num in range(n_Haars):

        #Make Haar-random matrix
        #matrix = scipy.stats.ortho_group.rvs(N)

        #Load Haar random matrix
        matrix = np.loadtxt(f'figures_set1/test_Haar_8x8.txt')

        (tlist, localv, discard) = triangular_symmetric(matrix)
        tlist = strawberryfields_to_neuroptica_reck(tlist, N)
        tlist = np.array(tlist)
        localv = np.log(localv)/1j

        sz = np.shape(tlist)
        phase_data = [(tlist[i,2],tlist[i,3]) for i in range(sz[0])]

        network = flipped_ReckLayer(N, phases=phase_data, include_phase_shifter_layer=True,shifter_phases=localv,loss_diff=0.02)
        mesh = network.mesh

        n_samples = 1000

        transmittance_samples = np.zeros([n_samples,N-1])
                
        for i in range(n_dB):
            for sample in range(n_samples):
                for layer_cur in mesh.layers:
                    for mzi_cur in layer_cur:
                        if isinstance(mzi_cur,MZI_delay):
                            mzi_cur.loss_dB = losses_dB[i]
                            mzi_cur.randomize_errors()
                            if mzi_cur.loss_dB_cur < 0:
                                mzi_cur.loss_dB_cur = 0
                                print("gain detected")
                for n in reversed(range(N-1)):
                    transmittance_samples[sample,N-2-n] = transmittance(mesh, n, n+1, out1, out2)
            transmittances[i,:,haar_num] = np.mean(transmittance_samples,0)
            #print(fidels[i,j])

    transmittances = np.mean(transmittances,-1)

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
    plt.plot(losses_dB,transmittances.T[0],".-",label="Inputs 1 and 2")
    plt.plot(losses_dB,transmittances.T[1],"s-",label="Inputs 2 and 3")
    plt.plot(losses_dB,transmittances.T[2],"o-",label="Inputs 3 and 4")
    plt.plot(losses_dB,transmittances.T[3],"^-",label="Inputs 4 and 5")
    plt.plot(losses_dB,transmittances.T[4],"v-",label="Inputs 5 and 6")
    plt.plot(losses_dB,transmittances.T[5],"<-",label="Inputs 6 and 7")
    plt.plot(losses_dB,transmittances.T[6],"^-",label="Inputs 7 and 8")
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='minor', labelsize=tick_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    plt.xlabel('Mean Loss(dB)', fontsize=labels_size)
    plt.ylabel(r'Transmittance to ' + '\n' + r'Outputs 1 and 2', fontsize=labels_size)
    ax.legend()
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=tick_size)
    #cbar.set_label('Fidelity', fontsize=labels_size)
    # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
    plt.tight_layout()
    #plt.show()


    plt.savefig(f'figures_set1/transmittance_1_2_vs_mean_loss_Reck_1000_samples_1Haar_loss_diff_002_new_loss.png')

    #Save Data
    np.savetxt(f'figures_set1/transmittance_1_2_vs_mean_loss_Reck_1000samples_1Haar_loss_diff_002_new_loss.txt',transmittances)
    timer.toc()

    
if __name__ == '__main__':
    main()
