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

def main():
    timer = TicToc()
    timer.tic()

    max_dB = 0.5
    step_size_dB = 0.05
    n_dB = int(max_dB/step_size_dB) + 1
    losses_dB = np.linspace(0,max_dB,n_dB)
    
    #Get original data
    unnormed_transmittances = np.loadtxt(f'figures_set1/transmittance_1_2_vs_mean_loss_Reck_1000samples_1Haar_loss_diff_002_new_loss.txt')
    norms = np.loadtxt(f'figures_set1/transmittance_1_2_no_err_Reck_1Haar_new_loss.txt')
    transmittances = unnormed_transmittances/norms
    print(transmittances)
    
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
    plt.plot(losses_dB,transmittances.T[0],"s-",label="Inputs 1 and 2")
    plt.plot(losses_dB,transmittances.T[1],".-",label="Inputs 2 and 3")
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

    timer.toc()
    
    plt.savefig(f'figures_set1/norm_transmittance_1_2_vs_mean_loss_Reck_1000_samples_1Haar_loss_diff_002_new_loss.png')

if __name__ == '__main__':
    main()
