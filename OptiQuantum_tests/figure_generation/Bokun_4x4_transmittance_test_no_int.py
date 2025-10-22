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

sys.path.append('../../')
from neuroptica.layers_new import DiamondLayer

def transmittance(mesh, idx_in1, idx_in2, idx_out1, idx_out2):
        transfer = mesh.get_transfer_matrix()
        c11 = transfer[idx_in1,idx_out1]
        c12 = transfer[idx_in1,idx_out2]
        c21 = transfer[idx_in2,idx_out1]
        c22 = transfer[idx_in2,idx_out2]
        
        coinc = (np.abs(c12)**2)*(np.abs(c21)**2) + (np.abs(c11)**2)*(np.abs(c22)**2) + \
                (np.conjugate(c11)*np.conjugate(c22)*c12*c21 + np.conjugate(c12)*np.conjugate(c21)*c11*c22)
        prob_a = (np.abs(c12)**2)*(np.abs(c22)**2) + (np.abs(c11)**2)*(np.abs(c21)**2)
        prob_b = (np.abs(c11)**2)*(np.abs(c21)**2) + (np.abs(c12)**2)*(np.abs(c22)**2)
        
        tot_prob = coinc + prob_a + prob_b
        return tot_prob

def main():
    timer = TicToc()

    #Mesh size
    N = 4
    #Create Mesh
    network = DiamondLayer(N, phases=[(np.pi, np.pi) for _ in range((N-1)**2)])
    mesh = network.mesh
    
    timer.tic()

    max_dB = 0.1
    step_size_dB = 0.01
    n_dB = int(max_dB/step_size_dB) + 1
    losses_dB = np.linspace(0,max_dB,n_dB)
    
    transmittances = np.zeros([n_dB,(N-1)**2])

    n = 0 #MZI number

    n_samples = 1000

    transmittance_samples = np.zeros(n_samples)

    #for layer in mesh.layers:
        #for mzi in layer:
            #print(np.round(mzi.get_transfer_matrix(),2))    
    
    for layer in mesh.layers:
        for mzi in layer:

            n += 1

            mzi.theta = np.pi/2 #Set MZI under test to 50:50 BS
            mzi.phi = 0
            
            for i in range(n_dB):
                for sample in range(n_samples):
                        
                    for layer_cur in mesh.layers:
                        for mzi_cur in layer_cur:
                            mzi_cur.loss_diff = losses_dB[i]
                            mzi_cur.randomize_errors()
                            if mzi_cur.loss_dB_cur < 0:
                                mzi_cur.loss_dB_cur = 0

                    transmittance_samples[sample] = transmittance(mesh, mzi.m, mzi.n, mzi.m, mzi.n)

                transmittances[i,n-1] = np.mean(transmittance_samples)

            mzi.theta = np.pi #Restore bar state to MZI under test
            mzi.phi = np.pi


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
    plt.plot(losses_dB,transmittances.T[0],".-",label="MZI 1")
    plt.plot(losses_dB,transmittances.T[1],"s-",label="MZI 2")
    plt.plot(losses_dB,transmittances.T[2],"o-",label="MZI 3")
    plt.plot(losses_dB,transmittances.T[3],"^-",label="MZI 4")
    plt.plot(losses_dB,transmittances.T[4],"v-",label="MZI 5")
    plt.plot(losses_dB,transmittances.T[5],"<-",label="MZI 6")
    plt.plot(losses_dB,transmittances.T[6],">-",label="MZI 7")
    plt.plot(losses_dB,transmittances.T[7],"+-",label="MZI 8")
    plt.plot(losses_dB,transmittances.T[8],"D-",label="MZI 9")
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='minor', labelsize=tick_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    plt.xlabel('Loss Uncertainty(dB)', fontsize=labels_size)
    plt.ylabel(r'Transmittance', fontsize=labels_size)
    ax.legend()
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=tick_size)
    #cbar.set_label('Fidelity', fontsize=labels_size)
    # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
    plt.tight_layout()
    #plt.show()

    plt.savefig(f'figures_set1/transmittance_vs_random_loss_Bokun_1000_samples.png')

    #Save Data
    np.savetxt(f'figures_set1/transmittance_vs_random_phases_Bokun_1000samples.txt',transmittances)


    timer.toc()

    
if __name__ == '__main__':
    main()
