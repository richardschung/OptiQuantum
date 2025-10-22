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
            layers.append(MZIDelayLayer.from_waveguide_indices(MZIDelayLayer,i,N,list(range(N)),np.full(int(N/2),np.pi),np.full(int(N/2),0)))
        else:
            layers.append(MZIDelayLayer.from_waveguide_indices(MZIDelayLayer,i,N,list(range(1,N-1)),np.full(int(N/2)-1,np.pi),np.full(int(N/2)-1,0)))
    layers[0].mzis[0].theta = np.pi/2
    layers[0].mzis[0].phi = 0
    mesh = OpticalMesh(N,layers)

    for layer in mesh.layers:
            for mzi in layer:
                print(np.round(mzi.get_transfer_matrix(),2))
    
    timer.tic()

    n_phase = 9
    phase_errs = np.linspace(0,0.8,n_phase)

    visibs = np.zeros(n_phase)
    
    for i in range(n_phase):
        for layer in mesh.layers:
            for mzi in layer:
                mzi.theta_err = phase_errs[i]
                mzi.phi_err = phase_errs[i]

        print(np.round(mesh.get_transfer_matrix(),2))
            
        intfunc_min = lambda omega1, omega2:HOM_integrand_mesh(mesh,omega1,omega2,0, 1, 0, 1, 0, 1)
        intfunc_min_real = lambda omega1, omega2: np.real(intfunc_min(omega1, omega2))
        intfunc_min_imag = lambda omega1, omega2: np.imag(intfunc_min(omega1, omega2))
        coinc_min_real = scipy.integrate.dblquad(intfunc_min_real,-np.inf,np.inf,-np.inf,np.inf)
        coinc_min_imag = scipy.integrate.dblquad(intfunc_min_imag,-np.inf,np.inf,-np.inf,np.inf)

        intfunc_max = lambda omega1, omega2:HOM_integrand_mesh(mesh,omega1,omega2,5, 1, 0, 1, 0, 1)
        intfunc_max_real = lambda omega1, omega2: np.real(intfunc_max(omega1, omega2))
        intfunc_max_imag = lambda omega1, omega2: np.imag(intfunc_max(omega1, omega2))
        coinc_max_real = scipy.integrate.dblquad(intfunc_max_real,-np.inf,np.inf,-np.inf,np.inf)
        coinc_max_imag = scipy.integrate.dblquad(intfunc_max_imag,-np.inf,np.inf,-np.inf,np.inf)

        print((coinc_max_real[0] - coinc_min_real[0])*2)
        print(coinc_max_real[0])
        print(coinc_min_real[0])
        print((coinc_max_imag[0] - coinc_min_imag[0])*2)
        visibs[i] = (coinc_max_real[0] - coinc_min_real[0])*2

    fig, ax = plt.subplots()
    ax.plot(phase_errs, visibs)
    plt.xlabel('Absolute Phase Error (Radians)')
    plt.ylabel('Normalized Visibility')
    
    timer.toc()

    plt.show()
    
if __name__ == '__main__':
    main()
