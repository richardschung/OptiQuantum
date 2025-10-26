import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from pytictoc import TicToc

sys.path.append('../../')
from neuroptica.layers_new import ReckLayer
from metrics import visibility

def main():
    timer = TicToc()

    #Mesh size
    N = 4
    #Create Mesh
    network = ReckLayer(N, phases=[(np.pi, np.pi) for _ in range(int(N*(N-1)/2))], phase_uncert=np.pi/32)
    mesh = network.mesh
    
    timer.tic()

    max_phase = 0.8
    step_size_phase = 0.1
    n_phase = int(max_phase/step_size_phase) + 1
    phases = np.linspace(0,max_phase,n_phase)

    visibs = np.zeros([n_phase,n_phase])

    n = 0 #MZI number

    n_samples = 1000

    visib_samples = np.zeros(n_samples)

    for layer in mesh.layers:
        for mzi in layer:
            mzi.theta0 = mzi.theta
            mzi.phi0 = mzi.phi

    for layer in mesh.layers:
        for mzi in layer:

            n += 1

            mzi.theta0 = np.pi/2 #Set MZI under test to 50:50 BS
            mzi.phi0 = 0
            
            for i in range(n_phase):
                for j in range(n_phase):
                    for sample in range(n_samples):
                        for layer_cur in mesh.layers:
                            for mzi_cur in layer_cur:
                                mzi_cur.theta = mzi.theta0 + phases[i]
                                mzi_cur.phi = mzi.phi0 + phases[j]
                                mzi_cur.randomize_errors()

                        visib_samples[sample] = visibility(mesh, mzi.m, mzi.n, mzi.m, mzi.n)

                    visibs[i,j] = np.mean(visib_samples)
                    #print(visibs[i,j])

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
            plt.figure(figsize=(6.95, 5.03)) # compress the graph (around) quarter in size, by cutting top half and compress horizontally
            plt.pcolor(phases, phases, visibs.T, cmap=cmap, rasterized=True, vmin=0, vmax=1)

            ax = plt.gca()
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='both', which='minor', labelsize=tick_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_size)

            plt.xlabel(r'Mean Error in $\theta$ (rad)', fontsize=labels_size)
            plt.ylabel(r'Mean Error in $\phi$(rad)', fontsize=labels_size)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=tick_size)
            cbar.set_label('Visibility', fontsize=labels_size)
            # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
            plt.tight_layout()

            plt.savefig(f'figures_set1/Update October 26/visibility_vs_mean_phases_Reck_1000samples_MZI_{n}.png')

            #Save Data
            np.savetxt(f'figures_set1/Update October 26/visibiity_vs_mean_phases_Reck_1000samples_MZI_{n}.txt',visibs)
            
            mzi.theta0 = np.pi #Restore bar state to MZI under test
            mzi.phi0 = np.pi

    timer.toc()

    
if __name__ == '__main__':
    main()
