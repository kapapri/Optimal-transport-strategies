import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
colors = sns.color_palette("CMRmap")
colors2 = sns.color_palette(palette='Paired')
colors3 = sns.color_palette("icefire")
colors4 = sns.color_palette("Set2")
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

runs = 100
Lx = 20
Ly = 10
Nt = 200
start_recording = 100
recording_time = Nt-start_recording

random_current = np.loadtxt(f'./output_Attractive_TASEP/AttractiveTASEP_current_{Lx}x{Ly}_runs{runs}.txt') 
forward_current = np.loadtxt(f'./output_smart_Attractive_TASEP/Smart_AttractiveTASEP_current_{Lx}x{Ly}_runs{runs}.txt')
random_empty_sites = np.loadtxt(f'./output_Attractive_TASEP/AttractiveTASEP_empty_sites_{Lx}x{Ly}_runs{runs}.txt') 
forward_empty_sites = np.loadtxt(f'./output_smart_Attractive_TASEP/Smart_AttractiveTASEP_empty_sites_{Lx}x{Ly}_runs{runs}.txt')    
episode_duration = np.arange(Nt-start_recording)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5), constrained_layout=True)
ax1.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)
ax2.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)

# Current plot    -----------------------------
# If your data has multiple columns and you want to plot each column against the first column:
for i in range(1, random_current.shape[1]):
    ax1.plot(episode_duration[:], forward_current[:, i], label=f'Trained', color = 'crimson')
    ax1.plot(episode_duration, random_current[:, i], label=f'Untrained', color = colors4[0])

# Add labels and title
ax1.set(xlabel = 'Episode duration', ylabel = f'Current \n Averaged over {runs} runs')
ax1.set_ylim(0,0.35)
ax1.legend(loc = 0, prop={'size': fontsize}, ncol = 1)

# Empty  plot    -----------------------------
for i in range(1, random_empty_sites.shape[1]):
    ax2.plot(episode_duration, forward_empty_sites[:, i], label=f'Trained', color = 'indigo')
    ax2.plot(episode_duration, random_empty_sites[:, i], label=f'Untrained', color = colors4[4])

# Add labels and title
ax2.set(xlabel = 'Episode duration', ylabel = f'Probability of empty sites \n Averaged over {runs} runs')
ax2.set_ylim(-0.05,0.6)
ax2.legend(loc='center right', prop={'size': 12}, ncol =1)
plt.savefig(f"./AttractiveTASEP_Comparison_{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')


#-------------------------------------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5), constrained_layout=True)
ax1.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))    
ax2.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))   

rand_YcurrentII_fast = np.loadtxt(f'output_Attractive_TASEP/AttractiveTASEP_YcurrentII_fast_{Lx}x{Ly}_runs{runs}.txt') 
rand_YcurrentII_slow = np.loadtxt(f'output_Attractive_TASEP/AttractiveTASEP_YcurrentII_slow_{Lx}x{Ly}_runs{runs}.txt')
train_YcurrentII_fast = np.loadtxt(f'output_smart_Attractive_TASEP/Smart_AttractiveTASEP_YcurrentII_fast_{Lx}x{Ly}_runs{runs}.txt')  
train_YcurrentII_slow = np.loadtxt(f'output_smart_Attractive_TASEP/Smart_AttractiveTASEP_YcurrentII_slow_{Lx}x{Ly}_runs{runs}.txt')
boundary_lane = Ly/2

ax1.set(xlabel = 'Vertical index', ylabel = f'Parallel current.\n Averaged over {runs} runs')
ax1.axvspan(-10, (boundary_lane - 0.5), facecolor = colors3[5], alpha=0.1)
ax1.set_xlim([-0.3, Ly-1+0.3])
ax1.axvspan((boundary_lane - 0.5), (Ly-1+10), facecolor = colors3[1], alpha=0.1)      

ax1.plot(rand_YcurrentII_fast, '-o' , label = 'Untrained fast particles', color = colors5[5])
ax1.plot(rand_YcurrentII_slow, '-o', label = 'Untrained slow particles', color = colors5[4])
ax1.plot(train_YcurrentII_fast, '-o' , label = 'Trained fast particles', color = colors3[5])
ax1.plot(train_YcurrentII_slow, '-o', label = 'Trained slow particles', color = colors3[1])

# ax1.legend(loc='upper right', prop={'size': fontsize}, ncol = 1)    

# plt.savefig(f"./AttractiveTASEP_Comparison_YcurrentII{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')

#-------------------------------------------------------------------------------------------------------------
rand_YcurrentT_fast = np.loadtxt(f'output_Attractive_TASEP/AttractiveTASEP_YcurrentT_fast_{Lx}x{Ly}_runs{runs}.txt') 
rand_YcurrentT_slow = np.loadtxt(f'output_Attractive_TASEP/AttractiveTASEP_YcurrentT_slow_{Lx}x{Ly}_runs{runs}.txt')
train_YcurrentT_fast = np.loadtxt(f'output_smart_Attractive_TASEP/Smart_AttractiveTASEP_YcurrentT_fast_{Lx}x{Ly}_runs{runs}.txt')  
train_YcurrentT_slow = np.loadtxt(f'output_smart_Attractive_TASEP/Smart_AttractiveTASEP_YcurrentT_slow_{Lx}x{Ly}_runs{runs}.txt')

boundary_lane = Ly/2
ax2.set(xlabel = 'Vertical index', ylabel = f'Perpendicullar current.\n Averaged over {runs} runs')
ax2.axvspan(-10, (boundary_lane - 0.5), facecolor = colors3[5], alpha=0.1)
ax2.set_xlim([-0.3, Ly-1+0.3])
ax2.set_ylim(-0.008, 0.0055)
ax2.axvspan((boundary_lane - 0.5), (Ly-1+10), facecolor = colors3[1], alpha=0.1)      

ax2.plot(rand_YcurrentT_fast, '-o' , color = colors5[5])
ax2.plot(rand_YcurrentT_slow, '-o',  color = colors5[4])
ax2.plot(train_YcurrentT_fast, '-o' ,  color = colors3[5])
ax2.plot(train_YcurrentT_slow, '-o',  color = colors3[1])

# ax2.legend(loc='lower right', prop={'size': fontsize}, ncol =1)    
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2)
plt.savefig(f"./AttractiveTASEP_Comparison_Ycurrents{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')

# #------------------------------------------------------------------------------------

random_fast_sites = np.loadtxt(f'output_Attractive_TASEP/AttractiveTASEP_fast_right_{Lx}x{Ly}_runs{runs}.txt') 
smart_fast_sites = np.loadtxt(f'output_smart_Attractive_TASEP/Smart_AttractiveTASEP_fast_right_{Lx}x{Ly}_runs{runs}.txt')  
random_slow_sites = np.loadtxt(f'output_Attractive_TASEP/AttractiveTASEP_slow_right_{Lx}x{Ly}_runs{runs}.txt') 
smart_slow_sites = np.loadtxt(f'output_smart_Attractive_TASEP/Smart_AttractiveTASEP_slow_right_{Lx}x{Ly}_runs{runs}.txt')  

episode_duration = np.arange(Nt)
fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize)) 
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))    

# If your data has multiple columns and you want to plot each column against the first column (for example):
for i in range(1, random_fast_sites.shape[1]):
    ax.plot(episode_duration, random_fast_sites[:, i], label=f'Untrained fast', color = colors5[5])
    ax.plot(episode_duration, random_slow_sites[:, i], label=f'Untrained slow', color = colors5[4])    
    ax.plot(episode_duration, smart_fast_sites[:, i], label=f'Trained fast', color = colors3[5])
    ax.plot(episode_duration, smart_slow_sites[:, i], label=f'Trained slow', color = colors3[1])

 
# Add labels and title
ax.set(xlabel = 'Time step', ylabel = f'Probability of being in their area. \n Averaged over {runs} runs')
ax.legend(loc=0, prop={'size': fontsize}, ncol =1) 
plt.savefig(f"./AttractiveTASEP_Comparison_particle_occupation{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')


# #------------------------------------------------------------------------------------
random_fast_chosen = np.loadtxt(f'./output_Attractive_TASEP/AttractiveTASEP_fast_chosen{Lx}x{Ly}_runs{runs}.txt') 
trained_fast_chosen = np.loadtxt(f'./output_smart_Attractive_TASEP/Smart_AttractiveTASEP_fast_chosen{Lx}x{Ly}_runs{runs}.txt')
random_slow_chosen = np.loadtxt(f'./output_Attractive_TASEP/AttractiveTASEP_slow_chosen{Lx}x{Ly}_runs{runs}.txt') 
trained_slow_chosen = np.loadtxt(f'./output_smart_Attractive_TASEP/Smart_AttractiveTASEP_slow_chosen{Lx}x{Ly}_runs{runs}.txt')    
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

plt.savefig(f"./AttractiveTASEP_Comparison_types_chosen_{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')


# #------------------------------------------------------------------------------------


# lane_density_fast = np.loadtxt(f'./AttractiveTASEP_lane_density_fast_{Lx}x{Ly}_runs{runs}.txt') 
# lane_density_slow = np.loadtxt(f'./AttractiveTASEP_lane_density_slow_{Lx}x{Ly}_runs{runs}.txt')
# train_lane_density_fast = np.loadtxt(f'./Smart_AttractiveTASEP_lane_density_fast_{Lx}x{Ly}_runs{runs}.txt')  
# train_lane_density_slow = np.loadtxt(f'./Smart_AttractiveTASEP_lane_density_slow_{Lx}x{Ly}_runs{runs}.txt')
# boundary_lane = Ly/2
# fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize)) 
# ax.set(xlabel = 'Vertical index', ylabel = 'Parallel current. Averaged over {runs} runs')
# ax.axvspan(-10, (boundary_lane - 0.5), facecolor = colors3[5], alpha=0.2, label='Fast region')
# ax.set_xlim([-0.3, Ly-1+0.3])
# ax.axvspan((boundary_lane - 0.5), (Ly-1+10), facecolor = colors3[1], alpha=0.2, label='Slow region')      

# ax.plot(lane_density_fast, '-' , label = 'Untrained fast particles', color = colors3[5])
# ax.plot(lane_density_slow, '-', label = 'Untrained slow particles', color = colors3[1])
# ax.plot(train_lane_density_fast, '-' , label = 'Trained fast particles', color = colors3[4])
# ax.plot(train_lane_density_slow, '-', label = 'Trained slow particles', color = colors3[2])
# ax.legend(loc=0, prop={'size': fontsize}, ncol =1)    

# plt.savefig(f"./AttractiveTASEP_Comparison_lane_density{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')