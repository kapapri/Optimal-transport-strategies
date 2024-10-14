import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
colors = sns.color_palette("CMRmap")
colors2 = sns.color_palette("Set2")
colors3 = sns.color_palette("icefire")
colors4 = sns.color_palette(palette='Paired')
colors5 = sns.color_palette("Set3")

fontsize = 12
plt.rc('font', family='serif', size=fontsize) # fonts same than latex
plt.rc('text', usetex=True)
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
fig_xsize = 6 
fig_ysize = 3.5

runs = 20
Lx = 20
Ly = 10
Nt = 175
start_recording = 75
recording_time = Nt-start_recording

# random_current = np.loadtxt(f'./TASEP_current_{Lx}x{Ly}_runs{runs}.txt') 
# forward_current = np.loadtxt(f'./Smart_TASEP_current_{Lx}x{Ly}_runs{runs}.txt')
# random_empty_sites = np.loadtxt(f'./TASEP_empty_sites_{Lx}x{Ly}_runs{runs}.txt') 
# forward_empty_sites = np.loadtxt(f'./Smart_TASEP_empty_sites_{Lx}x{Ly}_runs{runs}.txt')    
# episode_duration = np.arange(Nt-start_recording)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))
# ax1.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)
# ax2.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)

# # Current plot    -----------------------------
# # If your data has multiple columns and you want to plot each column against the first column:
# for i in range(1, random_current.shape[1]):
#     ax1.plot(episode_duration[:], forward_current[:, i], label=f'Trained', color = 'crimson')
#     ax1.plot(episode_duration, random_current[:, i], label=f'Untrained', color = colors2[0])

# # Add labels and title
# ax1.set(xlabel = 'Episode duration', ylabel = f'Current')
# ax1.set_ylim(0,0.35)
# ax1.legend(loc = 0, prop={'size': fontsize}, ncol = 1)

# # Empty  plot    -----------------------------
# for i in range(1, random_empty_sites.shape[1]):
#     ax2.plot(episode_duration, forward_empty_sites[:, i], label=f'Trained', color = 'indigo')
#     ax2.plot(episode_duration, random_empty_sites[:, i], label=f'Untrained', color = colors2[4])

# # Add labels and title
# ax2.set(xlabel = 'Episode duration', ylabel = f'Prob. of empty sites')
# ax2.set_ylim(-0.05,0.6)
# ax2.legend(loc='center right', prop={'size': 12}, ncol =1)
# plt.savefig(f"./TASEP_Comparison_{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')

#-----------------------------------------------------------------------------------------------
random_fast_chosen = np.loadtxt(f'./TASEP_fast_chosen{Lx}x{Ly}_runs{runs}.txt') 
trained_fast_chosen = np.loadtxt(f'./Smart_TASEP_fast_chosen{Lx}x{Ly}_runs{runs}.txt')
random_slow_chosen = np.loadtxt(f'./TASEP_slow_chosen{Lx}x{Ly}_runs{runs}.txt') 
trained_slow_chosen = np.loadtxt(f'./Smart_TASEP_slow_chosen{Lx}x{Ly}_runs{runs}.txt')    
episode_duration = np.arange(Nt-start_recording)

fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize))
ax.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)

# Prob. types of particle chosen    -----------------------------
# If your data has multiple columns and you want to plot each column against the first column:
for i in range(1, random_fast_chosen.shape[1]):
    ax.plot(episode_duration, random_fast_chosen[:, i], label=f'Untrained fast', color = colors5[5])
    ax.plot(episode_duration, random_slow_chosen[:, i], label=f'Untrained slow', color = colors5[4])
    ax.plot(episode_duration, trained_fast_chosen[:, i], label=f'Trained fast', color = colors3[5])
    ax.plot(episode_duration, trained_slow_chosen[:, i], label=f'Trained slow', color = colors3[1])    
# Add labels and title
ax.set(xlabel = 'Episode duration', ylabel = f'Probability of particle selected \n Averaged over {runs}runs')
ax.legend(loc = 0, prop={'size': fontsize}, ncol = 2)

plt.savefig(f"./TASEP_Comparison_types_chosen_{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')