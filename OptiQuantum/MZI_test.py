'''Program to test HOM interferometry simulation using Neuroptica layers.
Current layer to test is a single modified MZI.
'''
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
from pytictoc import TicToc

sys.path.append('../')
from neuroptica.components_new import MZI_delay

def HOM_integrand_MZI(layer, omega1, omega2, tau, sigma):
    '''Function for the integrand used in the HOM interferometry simulation.
    :param layer: The Neuroptica layer or component under simulation
    :param omega1: Frequency of the first photon, integration variable
    :param omega2: Frequency of the second photon, integration variable
    :param tau: Input time delay
    :param omega0: Photon mean frequency
    :param sigma: Standard deviation of the photon mean frequency
    '''
    #Find transfer matrices

    #Store original frequency
    omega_temp = layer.omega

    #Set frequency to omega1
    layer.omega = omega1

    #Get transfer matrix
    transfer1 = layer.get_transfer_matrix()

    #Set frequency to omega2
    layer.omega = omega2

    #Get transfer matrix
    transfer2 = layer.get_transfer_matrix()

    #Restore original frequency
    layer.omega = omega_temp

    #Evaluate integrand
    integ1 = transfer1[0,1]*(transfer1[0,1].conj())*transfer2[1,0]*(transfer2[1,0].conj())
    integ2 = transfer1[0,0]*(transfer1[0,0].conj())*transfer2[1,1]*(transfer2[1,1].conj())
    integ3 = (transfer1[0,0].conj())*(transfer2[1,1].conj())*transfer2[0,1]*transfer1[1,0]
    integ4 = (transfer1[0,1].conj())*(transfer2[1,0].conj())*transfer2[0,0]*transfer1[1,1]
    gauss1 = np.exp(-(omega1**2)/(2*(sigma**2)))/(np.sqrt(sigma*np.sqrt(np.pi)))
    gauss2 = np.exp(-(omega2**2)/(2*(sigma**2)))/(np.sqrt(sigma*np.sqrt(np.pi)))

    integrand = (integ1 + integ2 + (integ3 + integ4)*np.exp(1j*(omega2-omega1)*tau))*(abs(gauss1)**2)*(abs(gauss2)**2)

    return integrand

def main():

    timer = TicToc()
    timer.tic()
    print(np.exp(1j*np.pi))
    
    test_layer0 = MZI_delay(0, 1, 0, 0, 0, 0, 0, 0, 0, 0.0)
    print(test_layer0.get_transfer_matrix())
    test_layer1 = MZI_delay(0, 1, np.pi, 0, 0, 0, 0, 0, 0, 0.0)
    print(test_layer1.get_transfer_matrix())

##    res = 100
##    taus = np.linspace(-5,5,res)
##    coinc = np.zeros(res)
##    
##    for i in range(res):
##        intfunc = lambda omega1, omega2:HOM_integrand_MZI(test_layer0,omega1,omega2,taus[i], 1)
##        result = scipy.integrate.dblquad(intfunc,-np.inf,np.inf,-np.inf,np.inf)
##        coinc[i] = result[0]
##
##    fig, ax = plt.subplots()
##    ax.plot(taus, coinc)
    
    timer.toc()

##    plt.show()
    
if __name__ == '__main__':
    main()
