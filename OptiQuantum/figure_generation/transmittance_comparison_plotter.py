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
    unnormed_transmittances_Reck = np.loadtxt(f'figures_set1/transmittance_7_8_vs_mean_loss_Reck_1000samples_1Haar_loss_diff_002_new_loss.txt')
    norms_Reck = np.loadtxt(f'figures_set1/transmittance_7_8_no_err_Reck_1Haar_new_loss.txt')
    transmittances_Reck = unnormed_transmittances_Reck/norms_Reck
    unnormed_transmittances_Clem = np.loadtxt(f'figures_set1/transmittance_7_8_vs_mean_loss_Clements_1000samples_1Haar_loss_diff_002_new_loss.txt')
    norms_Clem = np.loadtxt(f'figures_set1/transmittance_7_8_no_err_Clements_1Haar_new_loss.txt')
    transmittances_Clem = unnormed_transmittances_Clem/norms_Clem
    print(np.shape(transmittances_Clem))
    
    #Plotting code from ONN_Simulation_Class.py
    labels_size = 14
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
    plt.plot(losses_dB,transmittances_Reck.T[0],"bs-",label="Inputs 1 and 2, Reck")
    plt.plot(losses_dB,transmittances_Reck.T[6],"b^-",label="Inputs 7 and 8, Reck")
    plt.plot(losses_dB,np.mean(transmittances_Clem,-1).T,"rv-",label="Average, Clements")
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.xlabel('Mean Loss(dB)', fontsize=labels_size)
    plt.ylabel(r'Transmittance to ' + '\n' + r'Outputs 7 and 8', fontsize=labels_size)
    ax.legend(loc='lower left')
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=tick_size)
    #cbar.set_label('Fidelity', fontsize=labels_size)
    # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
    plt.tight_layout()
    #plt.show()

    timer.toc()
    
    plt.savefig(f'figures_set1/transmittance_comparison_out78_V2.png')

if __name__ == '__main__':
    main()
