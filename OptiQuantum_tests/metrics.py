import numpy as np

def HOM_simple(mesh, idx_in1, idx_in2, idx_out1, idx_out2, tau, sigma):
    '''Function for HOM coincidence probability without integration.
    Fails when transfer matrices have strong frequency dependence
    :param mesh: The Neuroptica mesh under simulation
    :param idx_in1: Waveguide index of the first input path
    :param idx_in2: Waveguide index of the second input path
    :param idx_out1: Waveguide index of the first output path
    :param idx_out2: Waveguide index of the second output path
    :param tau: Time delay between photon entry
    :param sigma: Standard deviation of the photon mean frequency
    '''
    #Get transfer matrix
    transfer = mesh.get_transfer_matrix()

    #Evaluate visiblity
    term1 = transfer[idx_out1,idx_in2]*(transfer[idx_out1,idx_in2].conj())*transfer[idx_out2,idx_in1]*(transfer[idx_out2,idx_in1].conj())
    term2 = transfer[idx_out1,idx_in1]*(transfer[idx_out1,idx_in1].conj())*transfer[idx_out2,idx_in2]*(transfer[idx_out2,idx_in2].conj())
    term3 = (transfer[idx_out1,idx_in1].conj())*(transfer[idx_out2,idx_in2].conj())*transfer[idx_out1,idx_in2]*transfer[idx_out2,idx_in1]
    term4 = (transfer[idx_out1,idx_in2].conj())*(transfer[idx_out2,idx_in1].conj())*transfer[idx_out1,idx_in1]*transfer[idx_out2,idx_in2]
    gauss = np.exp(-(sigma**2)*(tau**2)/2)
    
    coinc = term1 + term2 + (term3 + term4)*gauss
    
    return coinc

def visibility(mesh, idx_in1, idx_in2, idx_out1, idx_out2):
    '''Function for calculating HOM visibility.
    :param mesh: The Neuroptica mesh under simulation
    :param idx_in1: Waveguide index of the first input path
    :param idx_in2: Waveguide index of the second input path
    :param idx_out1: Waveguide index of the first output path
    :param idx_out2: Waveguide index of the second output path
    '''
    #Get transfer matrix
    transfer = mesh.get_transfer_matrix()

    #Evaluate visiblity
    term1 = transfer[idx_out1,idx_in2]*(transfer[idx_out1,idx_in2].conj())*transfer[idx_out2,idx_in1]*(transfer[idx_out2,idx_in1].conj())
    term2 = transfer[idx_out1,idx_in1]*(transfer[idx_out1,idx_in1].conj())*transfer[idx_out2,idx_in2]*(transfer[idx_out2,idx_in2].conj())
    term3 = (transfer[idx_out1,idx_in1].conj())*(transfer[idx_out2,idx_in2].conj())*transfer[idx_out1,idx_in2]*transfer[idx_out2,idx_in1]
    term4 = (transfer[idx_out1,idx_in2].conj())*(transfer[idx_out2,idx_in1].conj())*transfer[idx_out1,idx_in1]*transfer[idx_out2,idx_in2]

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
    '''Function for calculating biphoton transmittance.
    :param mesh: The Neuroptica mesh under simulation
    :param idx_in1: Waveguide index of the first input path
    :param idx_in2: Waveguide index of the second input path
    :param idx_out1: Waveguide index of the first output path
    :param idx_out2: Waveguide index of the second output path
    '''

    transfer = mesh.get_transfer_matrix()
    c11 = transfer[idx_out1,idx_in1]
    c12 = transfer[idx_out1,idx_in2]
    c21 = transfer[idx_out2,idx_in1]
    c22 = transfer[idx_out2,idx_in2]
        
    coinc = (np.abs(c12)**2)*(np.abs(c21)**2) + (np.abs(c11)**2)*(np.abs(c22)**2) + \
            (np.conjugate(c11)*np.conjugate(c22)*c12*c21 + np.conjugate(c12)*np.conjugate(c21)*c11*c22)
    prob_a = (np.abs(c12)**2)*(np.abs(c22)**2) + (np.abs(c11)**2)*(np.abs(c21)**2)
    prob_b = (np.abs(c11)**2)*(np.abs(c21)**2) + (np.abs(c12)**2)*(np.abs(c22)**2)
        
    tot_prob = coinc + prob_a + prob_b
    return tot_prob
