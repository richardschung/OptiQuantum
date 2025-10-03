import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter

x = np.linspace(0,20,21)
y = np.linspace(0,10,11)

test = np.zeros([21,11])
for num in x:
    for num2 in y:
        print(num2)
        test[int(num),int(num2)] = num + num2

cmap='gist_heat'

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
plt.pcolor(x, y, test.T, cmap=cmap, rasterized=True, vmin=0, vmax = 30)

ax = plt.gca()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.tick_params(axis='both', which='minor', labelsize=tick_size)
ax.tick_params(axis='both', which='major', labelsize=tick_size)

plt.xlabel(r'$\theta$ error (radians)', fontsize=labels_size)
plt.ylabel(r'$\phi$ error (radians)', fontsize=labels_size)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=tick_size)
cbar.set_label('Visibility', fontsize=labels_size)
# plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
plt.tight_layout()
plt.show()
