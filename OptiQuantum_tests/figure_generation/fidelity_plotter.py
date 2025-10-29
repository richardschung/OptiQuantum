import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter

def main():

    max_phase = 0.8
    step_size_phase = 0.1
    n_phase = int(max_phase/step_size_phase) + 1
    phases = np.linspace(0,max_phase,n_phase)

    fidels = np.loadtxt(f'figures_set1/fidelity_vs_mean_phases_Clements_8x8_1000samples_1Haar_unc_pi_32.txt')

    cmap='gist_heat'

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
    #plt.figure(figsize=(6.95, 5.03)) # compress the graph (around) quarter in size, by cutting top half and compress horizontally
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

    plt.savefig(f'figures_set1/Update October 26/fidelity_vs_mean_phases_Clements_8x8_1000samples_1Haar_unc_pi_32.png')

if __name__ == '__main__':
    main()