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

sys.path.append('../')
from neuroptica.component_layers_new import MZIDelayLayer, OpticalMesh

def HOM_integrand_mesh(mesh, omega1, omega2, tau, sigma, idx_in1, idx_in2, idx_out1, idx_out2):
    '''Function for the integrand used in the HOM interferometry simulation.
    :param mesh: The Neuroptica mesh under simulation
    :param omega1: Frequency of the first photon, integration variable
    :param omega2: Frequency of the second photon, integration variable
    :param tau: Input time delay
    :param omega0: Photon mean frequency
    :param sigma: Standard deviation of the photon mean frequency
    :param idx_in1: Waveguide index of the first input path
    :param idx_in2: Waveguide index of the second input path
    :param idx_out1: Waveguide index of the first output path
    :param idx_out2: Waveguide index of the second output path
    '''
    #Find transfer matrices

    #Set frequency to omega1
    for layer in mesh.layers:
        layer.adjust_omegas(omega1)

    #Get transfer matrix
    transfer1 = mesh.get_transfer_matrix()

    #Set frequency to omega2
    for layer in mesh.layers:
        layer.adjust_omegas(omega2)

    #Get transfer matrix
    transfer2 = mesh.get_transfer_matrix()

    #Evaluate integrand
    integ1 = transfer1[idx_in1,idx_out2]*(transfer1[idx_in1,idx_out2].conj())*transfer2[idx_in2,idx_out1]*(transfer2[idx_in2,idx_out1].conj())
    integ2 = transfer1[idx_in1,idx_out1]*(transfer1[idx_in1,idx_out1].conj())*transfer2[idx_in2,idx_out2]*(transfer2[idx_in2,idx_out2].conj())
    integ3 = (transfer1[idx_in1,idx_out1].conj())*(transfer2[idx_in2,idx_out2].conj())*transfer2[idx_in1,idx_out2]*transfer1[idx_in2,idx_out1]
    integ4 = (transfer1[idx_in1,idx_out2].conj())*(transfer2[idx_in2,idx_out1].conj())*transfer2[idx_in1,idx_out1]*transfer1[idx_in2,idx_out2]
    gauss1 = np.exp(-(omega1**2)/(2*(sigma**2)))/(np.sqrt(sigma*np.sqrt(np.pi)))
    gauss2 = np.exp(-(omega2**2)/(2*(sigma**2)))/(np.sqrt(sigma*np.sqrt(np.pi)))

    integrand = (integ1 + integ2 + (integ3 + integ4)*np.exp(1j*(omega2-omega1)*tau))*(abs(gauss1)**2)*(abs(gauss2)**2)

    return integrand

def main():

    timer = TicToc()

    #Mesh size
    N = 4
    #Create component layers
    layers = []
    for i in range(N-1):
        if i % 2 == 0:
            #Create even component layer
            layers.append(MZIDelayLayer.from_waveguide_indices(MZIDelayLayer,i,N,list(range(N)),np.full(int(N/2),np.pi),np.zeros(int(N/2))))
        else:
            layers.append(MZIDelayLayer.from_waveguide_indices(MZIDelayLayer,i,N,list(range(1,N-1)),np.full(int(N/2)-1,np.pi),np.zeros(int(N/2)-1)))
    layers[0].mzis[0].theta = np.pi/2
    layers[0].mzis[0].phi = 0
    mesh = OpticalMesh(N,layers)
    
    timer.tic()

    max_dB = 0.8
    step_size_dB = 0.1
    n_dB = int(max_dB/step_size_dB) + 1
    losses_dB = np.linspace(0,max_dB,n_dB)
    max_phase = 0.8
    step_size_phase = 0.1
    n_phase = int(max_phase/step_size_phase) + 1
    phases = np.linspace(0,0.8,n_phase)

    visibs = np.zeros([n_phase,n_dB])
    
    for i in range(n_dB):
        for j in range(n_phase):
            for layer in mesh.layers:
                for mzi in layer:
                    mzi.loss_dB_cur = losses_dB[i]
                    mzi.theta_err = phases[j]
                    mzi.phi_err = phases[j]
                        
            intfunc_min = lambda omega1, omega2:HOM_integrand_mesh(mesh,omega1,omega2,0, 1, 0, 1, 0, 1)
            coinc_min = scipy.integrate.dblquad(intfunc_min,-np.inf,np.inf,-np.inf,np.inf)

            intfunc_max = lambda omega1, omega2:HOM_integrand_mesh(mesh,omega1,omega2,5, 1, 0, 1, 0, 1)
            coinc_max = scipy.integrate.dblquad(intfunc_max,-np.inf,np.inf,-np.inf,np.inf)

            visibs[i,j] = (coinc_max[0]-coinc_min[0])*2

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
    plt.pcolor(losses_dB, phases, visibs.T, cmap=cmap, rasterized=True)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='minor', labelsize=tick_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    plt.xlabel('Loss/MZI (dB)', fontsize=labels_size)
    plt.ylabel(r'Phase error on $\theta$ and $\phi$ (radians)', fontsize=labels_size)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.set_label('Visibility', fontsize=labels_size)
    # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
    plt.tight_layout()
    
    timer.toc()

    plt.show()
    
if __name__ == '__main__':
    main()
