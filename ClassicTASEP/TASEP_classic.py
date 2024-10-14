import numpy as np
import random
from numpy import random as Random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
fontsize = 12
plt.rc('font', family='serif', size=fontsize) # fonts same than latex
plt.rc('text', usetex=True)
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
fig_xsize = 6 
fig_ysize = 3.5
import seaborn as sns
colors = sns.color_palette("icefire")
colors2 = sns.color_palette(palette='Dark2')
from tqdm import tqdm

def trunc_gaussian_sample(mu, sigma):
    #it returns a sample of a gaussian, with mean mu and standard deviation sigma, only if it is in the [0, 1] interval
    sample = -1
    while sample < 0 or sample > 1:
        sample = np.random.normal(mu, sigma)
    return sample    

def checkerboard_gaussian(Lx, Ly, mu, sigma): 
    system = np.zeros(shape=(Lx,Ly))
    system[::2, ::2] = 1  # Set even rows and even columns to 1
    system[1::2, 1::2] = 1  # Set odd rows and odd columns to 1
    for i in range(Lx):
        for j in range(Ly):
            if system[i][j] == 1:
                system[i][j] = trunc_gaussian_sample(mu, sigma)
                    
    return system

def checkerboard_types(Lx, Ly, N, v_slow, v_fast, p_fast): 
    system = np.zeros(shape=(Lx,Ly))
    system[::2, ::2] = 1  # Set even rows and even columns to 1
    system[1::2, 1::2] = 1  # Set odd rows and odd columns to 1
    for i in range(Lx):
        for j in range(Ly):
            if system[i][j] == 1:
                system[i][j] = Random.choice([v_slow, v_fast], p =[1-p_fast, p_fast])
                    
    return system

def random_system_gaussian(Lx, Ly, N, mu, sigma): 
    system = np.zeros(shape=(Lx,Ly))
    n = 0
    while n < N:
        X = random.randint(0, Lx-1)
        Y = random.randint(0, Ly-1)
        if system[X][Y] == 0:
            system[X][Y] = trunc_gaussian_sample(mu, sigma)
            n += 1

    return system

def random_system_types(Lx, Ly, N, v_slow, v_fast, p_fast):
    system = np.zeros(shape=(Lx,Ly))
    n = 0
    while n < N:
        X = random.randint(0, Lx-1)
        Y = random.randint(0, Ly-1)
        if system[X][Y] == 0:
            system[X][Y] = Random.choice([v_slow, v_fast], p =[1-p_fast, p_fast])
            n += 1

    return system

# steady state plots
def currents_sigmas(sigmas): 
    Currents = np.zeros((Nt, len(sigmas)), dtype= np.float32)
    i = 0
    for sigma in sigmas:
        current,_,_,_,_ = simulate(runs, Nt, start_recording, Lx, Ly, init, mu, sigma, v_slow, v_fast, p_fast)
        Currents[:,i] = current
        i += 1
    return Currents
    
def currents_sigmas_plot():
    fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize))
    ax.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)
    ax.set(xlabel = 'Time step', ylabel = f'Current. Averaged over {runs} runs')
    ax.grid(visible='both', which='major', axis='both')
    
    sigmas = [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 5.0]
    # sigmas = [0.0001, 0.1, 0.6, 5.0]
    Currents = currents_sigmas(sigmas)
    times = np.array(range(Nt))
    for i in range(len(sigmas)):        
        ax.plot(times, Currents[:, i], label=f"${sigmas[i]}$", color = colors2[i])

    ax.legend(loc=0, title = r'$\sigma=$', prop={'size': fontsize}, ncol =2)            
        
def currents_ratios(ratios): 
    Currents = np.zeros((Nt, len(ratios)), dtype= np.float32)
    i = 0
    for ratio in ratios:
        current,_,_,_,_ = simulate(runs, Nt, start_recording, Lx, Ly, init, mu, sigma, v_slow, v_fast, ratio)
        Currents[:,i] = current
        i += 1
    return Currents
    
def currents_ratios_plot():
    fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize)) 
    ax.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)
    ax.set(xlabel = 'Time step', ylabel = f'Current. Averaged over {runs} runs')
    ax.grid(visible='both', which='major', axis='both')
    
    ratios = [0.3, 0.5, 0.7]
    Currents = currents_ratios(ratios)
    times = np.array(range(Nt))
    for i in range(len(ratios)):
        percentage = ratios[i]*100
        ax.plot(times, Currents[:, i], label=f"{int(percentage)} %", color = colors2[i])

    ax.legend(loc=0, title = f'Percentage fast =', prop={'size': fontsize}, ncol =1)

def currents_types(types): 
    Currents = np.zeros((Nt, len(types)), dtype= np.float32)
    i = 0
    for type_ in types:
        current,_,_,_,_ = simulate(runs, Nt, start_recording, Lx, Ly, init, mu, sigma, type_, v_fast, p_fast)
        Currents[:,i] = current
        i += 1
    return Currents
    
def currents_types_plot():
    fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize)) 
    ax.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)
    ax.set(xlabel = 'Time step', ylabel = f'Current. Averaged over {runs} runs')
    ax.grid(visible='both', which='major', axis='both')
    
    types = [0.1, 0.5, 0.9]
    Currents = currents_types(types)
    times = np.array(range(Nt))
    for i in range(len(types)):        
        ax.plot(times, Currents[:, i], label=f"{types[i]}", color = colors2[i])
    
    ax.legend(loc=0, title = r'$v_{slow} =$', prop={'size': fontsize}, ncol =2)

# plots
def plot_current(current):
    fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize)) 
    ax.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)
    ax.set(xlabel = 'Time step', ylabel = f'Current. Averaged over {runs} runs')

   #The steady current of particle J, through a bond i, i+1 is given by the rate r multiplied by the probability that there is a particle at site i, and site i+1 is vacant
    r = 0.5 #jumping rate
    p = 0.5 #probability forward
    # prob_occ= 1 # we make a transition only when the site chosen is occupied
    prob_vac= 1/2 #density of particles in the system
    prob_occ= 1/2 # we make a transition only when the site chosen is occupied    
    # f = np.array([r*p*prob_occ*prob_vac]*(Nt-start_recording)) #theoretical parallel current ith one specie
    # ax.set_ylim(0, np.max(current)*1.05) 
    ax.plot(current,'-', color ='darkmagenta')
    # f = np.array([0.11875]*Nt)
    # ax.plot(f,'-', color ='black')

def plot_empty_sites(empty_sites):
    fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize)) 
    ax.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)
    ax.set(xlabel = 'Time step', ylabel = f'Selected empty sites. Averaged over {runs} runs')
    ax.set_ylim(0, np.max(empty_sites)*1.05)    
    ax.plot(empty_sites, '-', color ='darkmagenta')

def plot_type_particles(fast_chosen, slow_chosen):
    fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize))
    ax.set(xlabel = 'Time step', ylabel = f'Probability of particle selected \n Averaged over {runs} runs')
    # ax.set_ylim(0, 0.3)    
    ax.plot(fast_chosen, label = 'Fast particles', color = colors[1])
    ax.plot(slow_chosen, label = 'Slow particles', color = colors[2])
    ax.legend(loc=0, prop={'size': fontsize}, ncol = 1)   

# Animation
def create_animation(Frames_movie):
    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    
    cv0 = Frames_movie[0]
    im = ax.imshow(cv0, cmap="gnuplot")
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title('Frame 0', y=1)
        
    ax.axis('off')
    plt.close()  # To not have the plot of frame 0

    def animate(frame):
        arr = Frames_movie[frame]
        vmax = 1
        vmin = np.min(arr)
        im.set_data(arr)
        im.set_clim(vmin, vmax)
        cb.ax.set_ylabel('Jumping Rate')
        tx.set_text('Frame {0}'.format(frame))

    ani = FuncAnimation(fig, animate, frames=len(Frames_movie), repeat=False)
    return ani

def simulate(runs, Nt, start_recording, Lx, Ly, init, mu, sigma, v_slow, v_fast, p_fast):   
    current = np.zeros(Nt-start_recording)
    empty_sites = np.zeros(Nt-start_recording)
    fast_chosen = np.zeros(recording_time)    
    slow_chosen = np.zeros(recording_time)
    # movie
    one_short_movie = np.zeros((Nt + 1, Lx, Ly))  # One run: Frames of set of movements (Lx*Ly) after one episode
    divisions = True
    movie_divisions = 100
    if divisions == True:
        one_movie = np.zeros((((Lx*Ly)*Nt)//movie_divisions+1, Lx, Ly))  # All runs: All frames consecutively
    else:
        one_movie = np.zeros(((Lx*Ly)*Nt + 1, Lx, Ly))  # All runs: All frames consecutively
    
    for run in tqdm(range(runs)):
        N = int(Lx*Ly*density) 
        if init =='chess_gauss':
            lattice = checkerboard_gaussian(Lx, Ly, mu, sigma)
        elif init == 'random_gauss':
            lattice = random_system_gaussian(Lx, Ly, N, mu, sigma)
        elif init == 'chess_types':
            lattice = checkerboard_types(Lx, Ly, N, v_slow, v_fast, p_fast)            

        else: # start with random initial conditions
            lattice = np.zeros(shape=(Lx,Ly))
            n = 0
            while n < N:
                X = random.randint(0, Lx-1)
                Y = random.randint(0, Ly-1)
                if lattice[X][Y] == 0:
                    lattice[X][Y] = Random.choice([v_slow, v_fast], p =[1-p_fast, p_fast])
                    n += 1
        
        one_short_movie[0] = lattice
        one_movie[0] = lattice
        k = 1
        m = 0
        for t in range(Nt):
            total_current = 0
            selected_empty_site = 0
            selected_fast = 0
            selected_slow = 0             
            for move_attempt in range(Lx*Ly):
                # Random sampling of the new lattice to apply the training
                selectedX = random.randint(0, Lx-1)
                selectedY = random.randint(0, Ly-1)

                if lattice[selectedX][selectedY] != 0:
                    newX = -1
                    newY = -1
                    nextX = selectedX + 1 if selectedX < Lx - 1 else 0
                    nextY = selectedY + 1 if selectedY < Ly - 1 else 0
                    prevY = selectedY - 1 if selectedY > 0 else Ly - 1
                    # update position
                    direction = random.randint(0,3)
                    if direction == 0 or direction == 1: # jump to the right
                        newX = nextX
                        newY = selectedY
                    elif direction == 2: # jump to the top
                        newY = nextY
                        newX = selectedX
                    else: # jump to the bottom
                        newY = prevY
                        newX = selectedX

                    jump_dice = random.random()
                    speed = lattice[selectedX][selectedY]     

                    # counting of selected fast and slow particles   
                    if speed > 0.9:
                        selected_fast += 1/(Lx*Ly*runs)
                    elif speed <= 0.9:
                        selected_slow += 1/(Lx*Ly*runs)    

                    if speed >= jump_dice:
                        if lattice[newX][newY] == 0:
                            lattice[selectedX][selectedY] = 0
                            lattice[newX][newY] = speed
                            if newX != selectedX: # we have jump forward
                                if t >= start_recording:
                                    total_current += 1/(Lx*Ly*runs)

                            elif newY == nextY:
                                if log == True:
                                    print("  moved up")

                            elif newY == prevY:
                                if log == True:
                                    print("  moved down")
                                                     
                else:
                    if log == True:
                        print("empty site chosen")
                    selected_empty_site += 1/(Lx*Ly*runs)
                      
                if divisions == True:
                    if k % movie_divisions == 0:  # Check if k is a multiple of movie_divisions
                        one_movie[k//movie_divisions] = lattice  # Store lattice in one_movie at every movie_divisions-th index                    
                else:
                    one_movie[k] = lattice
                k += 1

            one_short_movie[t + 1] = lattice

            if t >= start_recording:
                current[m] += total_current #sum of the currents of all runs
                empty_sites[m] += selected_empty_site
                fast_chosen[m] += selected_fast
                slow_chosen[m] += selected_slow                    
                m += 1
                
    return current, empty_sites, fast_chosen, slow_chosen, one_movie

# Main
if __name__ == '__main__': 
    log = False
    runs = 20
    Lx = 20
    Ly = 10
    Nt = 175
    start_recording = 75
    recording_time = Nt-start_recording

    init = "random_types"
    # two types particles
    v_slow = 0.5
    v_fast = 1
    p_fast = 0.5 # equal to probability of p_slow=1-p_fast=0.5
    density = 0.5          

    # gaussian particles # work with half-density
    mu = 0.5
    sigma = 0.01

    current, empty_sites, fast_chosen, slow_chosen, one_movie = simulate(runs, Nt, start_recording, Lx, Ly, init, mu, sigma, v_slow, v_fast, p_fast)
    # plot_current(current)
    # plt.savefig(f"./Current_{runs}runs_{Nt}steps_{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')
    # plot_empty_sites(empty_sites)
    # plt.savefig(f"./Empty_Sites_Chosen_{runs}runs_{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')

    filename = "TASEP_current_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    with open(filename, 'w') as f:
        for t in range(recording_time):
            output_string = str(t) + "\t" + str(current[t]) + "\n"
            f.write(output_string)

    filename = "TASEP_empty_sites_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    with open(filename, 'w') as f:
        for t in range(recording_time):
            output_string = str(t) + "\t" + str(empty_sites[t]) + "\n"
            f.write(output_string)

    filename = "TASEP_fast_chosen"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    with open(filename, 'w') as f:
        for t in range(recording_time):
            output_string = str(t) + "\t" + str(fast_chosen[t]) + "\n"
            f.write(output_string)

    filename = "TASEP_slow_chosen"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    with open(filename, 'w') as f:
        for t in range(recording_time):
            output_string = str(t) + "\t" + str(slow_chosen[t]) + "\n"
            f.write(output_string)                            
    # Movie
    # print('Action! (recording movie)')
    # straight_movie = np.transpose(one_movie, axes=(0, 2, 1))

    # ani = create_animation(straight_movie[:Nt]) #last run
    # HTML(ani.to_jshtml()) # interactive python
    # ani.save("./Movie"+ str(Lx) + "x" + str(Ly)  + "_runs" + str(runs) + ".gif", fps = 8)
    # print('Cut! (movie ready)')

    #6x6, 50 fps
    #12x12 movie_divisions = 500, 8 fps
    #20x10 movie_divisions = 1000, 8 fps
    #50x20 movie_divisions = 1200, 8 fps

    # #saving the movie array in a file
    # filename = "movie_storage" + ".pkl"
    # with open(filename, 'wb') as file:
    #     if divisions == True: 
    #         movie_storage = np.zeros(((Lx*Ly)*Nt//movie_divisions + 1, Lx, Ly))
    #     else:
    #         movie_storage = np.zeros(((Lx*Ly)*Nt + 1, Lx, Ly))

    #     movie_storage = straight_movie            
    #     pickle.dump(movie_storage, file)        


    # Steady State current plot -------------------------------------------------------------------------
    
    # start_recording = 0
    # # init = 'chess_gauss'
    # # currents_sigmas_plot()
    # # plt.savefig(f"./Current_Different_Sigmas_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')

    # init = 'chess_types'
    # start_recording = 0
    # recording_time = Nt-start_recording
    # currents_types_plot()
    # plt.savefig(f'./Current_Different_Types_{runs}_{Lx}x{Ly}.pdf', format="pdf", dpi=600, bbox_inches = 'tight') 
    # # currents_ratios_plot()
    # # plt.savefig(f"./Current_Different_Ratios_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')