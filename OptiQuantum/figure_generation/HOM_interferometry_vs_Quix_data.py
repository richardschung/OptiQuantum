import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from pytictoc import TicToc
from strawberryfields.decompositions_new import triangular_symmetric

sys.path.append('../../')
from neuroptica.layers_new import ClementsLayer
from neuroptica.component_layers_new import MZIDelayLayer, PhaseShifterLayer
from neuroptica.components_new import MZI_delay

sys.path.append('../')
from metrics import HOM_simple

def main():
    timer = TicToc()
    timer.tic()
    
    #Get data
    #Data from Quix paper '20-Mode Universal Quantum Photonic Processor'
    quix = np.loadtxt('quix_data.txt')
    
    #Mesh size
    N = 8
    #Create Mesh
    network = ClementsLayer(N, phases=[(np.pi, np.pi) for _ in range(int(N*(N-1)/2))])
    mesh = network.mesh

    mesh.layers[0].mzis[0].theta = np.pi/2 #Set upper left MZI to 50:50 state
    mesh.layers[0].mzis[0].phi = 0

    m = mesh.layers[0].mzis[0].m
    n = mesh.layers[0].mzis[0].n

    lengths = np.linspace(-0.3, 0.3, 501) #Delay lengths (mm) for iteration
    taus = lengths/(3*10**11) #Conversion to delay time (l(m)/c(m/s))
    wavelength = 1550*(10**-9) #Wavelength from quix paper
    dlambda = 12*(10**-9)/(2*np.sqrt(2*np.log(2))) #FWHM from quix paper converted to standard deviation
    sigma = (2*np.pi)*(3*(10**8))/(wavelength**2)*dlambda #Conversion to frequency uncertainty

    coincs = 2*HOM_simple(mesh, m, n, m, n, taus, sigma)
    
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
    plt.plot(lengths,coincs,label='Simulated HOM curve')
    plt.plot(quix[0],quix[1],'ro',label=r'Experimental data$^\dagger$')
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.xlabel('Delay Length(mm)', fontsize=labels_size)
    plt.ylabel('Normalized Coincicdence\nProbability', fontsize=labels_size)
    ax.legend()
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=tick_size)
    #cbar.set_label('Fidelity', fontsize=labels_size)
    # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
    plt.tight_layout()
    #plt.show()

    timer.toc()
    
    plt.savefig(f'figures_set1/HOM_vs_Quix_v2.png')

if __name__ == '__main__':
    main()
