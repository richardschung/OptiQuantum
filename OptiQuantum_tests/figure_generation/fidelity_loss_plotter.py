import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from pytictoc import TicToc
from strawberryfields.decompositions import rectangular_symmetric

sys.path.append('../../')
from neuroptica.layers_new import ClementsLayer
from neuroptica.component_layers_new import MZIDelayLayer, PhaseShifterLayer
from neuroptica.components_new import MZI_delay

sys.path.append('../')
from convert_decomposition import strawberryfields_to_neuroptica_clements

from metrics import fidelity

def main():
    timer = TicToc()

    #Mesh size
    N = 8

    n_Haars = 1

    max_dB = 0.5
    step_size_dB = 0.05
    n_dB = int(max_dB/step_size_dB) + 1
    losses_dB = np.linspace(0,max_dB,n_dB)

    fidels = np.loadtxt(f'figures_set1/Update October 26/Clements_8x8_fidelity_vs_random_loss_1000_samples_loss_diff_002.txt')

    #Plotting code from ONN_Simulation_Class.py
    labels_size = 14
    legend_size = 14
    tick_size = 12
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
    plt.plot(losses_dB,fidels)

    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.tick_params(axis='both', which='minor', labelsize=tick_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    plt.xlabel('Mean Loss(dB)', fontsize=labels_size)
    plt.ylabel(r'Fidelity', fontsize=labels_size)
    #ax.legend()
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=tick_size)
    #cbar.set_label('Fidelity', fontsize=labels_size)
    # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
    plt.tight_layout()
    #plt.show()


    plt.savefig(f'figures_set1/Update October 26/Clements_8x8_fidelity_vs_random_loss_1000_samples_loss_diff_002_V2.png')

if __name__ == "__main__":
    main()