
import numpy as np
import random
from numpy import random as Random
import matplotlib
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
import os

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def trunc_gaussian_sample(mu, sigma):
    sample = -1
    while sample < 0 or sample > 1:
        sample = np.random.normal(mu, sigma)
    return sample    

def truncated_gaussian(mu, sigma, size):
    samples = np.zeros(size, dtype=np.float32)
    for i in range(size):
        samples[i] = trunc_gaussian_sample(mu, sigma)   
    return samples

def checkboard_gaussian(Lx, Ly, mu, sigma): 
    system = np.zeros(shape=(Lx,Ly))
    system[::2, ::2] = 1  # Set even rows and even columns to 1
    system[1::2, 1::2] = 1  # Set odd rows and odd columns to 1
    for i in range(Lx):
        for j in range(Ly):
            if system[i][j] == 1:
                system[i][j] = trunc_gaussian_sample(mu, sigma)
                    
    return system

def checkboard_types(Lx, Ly, v_slow, v_fast, p_fast): 
    system = np.zeros(shape=(Lx,Ly))
    system[::2, ::2] = 1  
    system[1::2, 1::2] = 1  
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
    p_slow = 1 - p_fast
    n = 0
    while n < N:
        X = random.randint(0, Lx-1)
        Y = random.randint(0, Ly-1)
        if system[X][Y] == 0:
            system[X][Y] = Random.choice([v_slow, v_fast], p =[1-p_fast, p_fast])
            n += 1

    return system

def compute_prob(k, y, Ly, speed):
    # p is a probability to jump from Y to Y+1 here
    # we select force to be proportional to a distance from the center of the spring potential
    fast_center = int(Ly/4)
    slow_center = int(3*Ly/4)
    if speed > 0.9:
        distance = 0
        if y < slow_center:
            distance = y - fast_center
        else:
            distance = y - Ly - 1 - fast_center
    elif speed <= 0.9:
        if y > fast_center:
            distance = y - slow_center
        else:
            distance = y + Ly - 1 - slow_center

    force = -k*distance
    p = 0.5 + force # no force, no bias

    return p 

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

def plot_current(current):
    plt.clf() 
    plt.figure(2)
    plt.xlabel('Episode duration')
    plt.ylabel('Average current over runs {runs} runs')

    plt.ylim(0, np.max(current)*1.05)    
    plt.plot(current, color = 'darkmagenta')

def plot_empty_sites(empty_sites):
    plt.clf() 
    plt.figure(5)
    plt.xlabel('Episode duration')
    plt.ylabel(f'Probability of selected empty sites over {runs} runs')

    plt.ylim(0, np.max(empty_sites)*1.05)    
    plt.plot(empty_sites, color = 'darkmagenta')

def plot_YcurrentII(YcurrentII_fast, YcurrentII_slow):
    plt.clf() 
    plt.figure(3)
    plt.title(f"Attractive TASEP. Parallel current")
    plt.xlabel('Vertical index')
    plt.ylabel(f'Average parallel current over {runs} runs')
    plt.axvspan(-10, (boundary_lane - 0.5), facecolor = colors[5], alpha=0.2, label='Fast region')
    plt.xlim([-0.3, Ly-1+0.3])
    plt.axvspan((boundary_lane - 0.5), (Ly-1+10), facecolor = colors[1], alpha=0.2, label='Slow region')      
    plt.axhline(y=0, color='black', linestyle='--')
    plt.plot(YcurrentII_fast, '-o' ,label = 'fast particles', color = colors[5])
    plt.plot(YcurrentII_slow, '-o', label = 'slow particles', color = colors[1])
    plt.legend()

def plot_YcurrentT(YcurrentT_fast, YcurrentT_slow):
    plt.clf() 
    plt.figure(4)
    plt.title(f"Attractive TASEP. Perpendicular current")
    plt.xlabel('Vertical index')
    plt.ylabel('Average current over {runs} runs') 
    plt.axvspan(-10, (boundary_lane - 0.5), facecolor = colors[5], alpha=0.2, label='Fast region')
    plt.xlim([-0.3, Ly-1+0.3])
    plt.axvspan((boundary_lane - 0.5), (Ly-1+10), facecolor = colors[1], alpha=0.2, label='Slow region')      
    plt.axhline(y=0, color='black', linestyle='--')
    plt.plot(YcurrentT_fast, '-o' ,label = 'fast particles', color = colors[5])
    plt.plot(YcurrentT_slow, '-o', label = 'slow particles', color = colors[1])
    plt.legend()


def plot_type_particles(fast_chosen, slow_chosen):
    plt.clf() 
    plt.figure(6)
    plt.title(f"Attractive TASEP. Probability of choosing a particle")
    plt.xlabel('Episode duration')
    plt.ylabel(f'Probability over {runs} runs')
    # plt.ylim(0, np.max(slow_chosen)*1.10)    
    plt.plot(fast_chosen, label = 'fast particles', color = colors[5])
    plt.plot(slow_chosen, label = 'slow particles', color = colors[1])
    plt.legend()

def plot_particle_occupation(fast_sites, slow_sites):
    plt.clf() 
    plt.figure(7)
    plt.title(f"Attractive TASEP. Probability of a particle to be in its region")
    plt.xlabel('Episode duration')
    plt.ylabel(f'Average probability over {runs} runs')

    # plt.ylim(0, np.max(fast_sites)*1.10)    
    plt.plot(fast_sites, label = 'fast particles', color = colors[5])
    plt.plot(slow_sites, label = 'slow particles', color = colors[1])
    plt.legend(loc='lower right')

def plot_lane_density(density_fast, density_slow):
    plt.clf() 
    plt.xlabel('vertical lattice site')
    plt.ylabel(f'Average probability over {runs} runs')

    # plt.ylim(0, np.max(density_fast)*1.10)    
    plt.plot(density_fast, label = 'fast particles', color = colors[5])
    plt.plot(density_slow, label = 'slow particles', color = colors[1])
    plt.legend(loc='lower right')    
    
def currents_ratios(ratios): 
    Currents = np.zeros((Nt, len(ratios)), dtype= np.float32)
    i = 0
    for ratio in ratios:
        current = simulate(runs, Nt, start_recording, Lx, Ly, init, mu, sigma, v_slow, v_fast, ratio)
        Currents[:,i] = current
        i += 1
    return Currents
    
def currents_ratios_plot():
    plt.cla()
    plt.title(f"Attractive TASEP. Two types of particles over {runs} runs")
    plt.xlabel('Episode duration')
    plt.ylabel(f'Current. Averaged over {runs}')
    plt.grid(True)
    
    ratios = [0.3, 0.5, 0.7]
    Currents = currents_ratios(ratios)
    times = np.array(range(Nt))
    for i in range(len(ratios)):
        percentage = ratios[i]*100
        plt.plot(times, Currents[:, i], label=f"{int(percentage)} %", color = colors2[i])
        
    plt.legend(title = f'Percentage fast =', loc = 0)

def currents_types(types): 
    Currents = np.zeros((Nt, len(types)), dtype= np.float32)
    i = 0
    for type_ in types:
        current = simulate(runs, Nt, start_recording, Lx, Ly, init, mu, sigma, type_, v_fast, p_fast)
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
    
    ax.legend(loc=1, title = r"$v_{slow} =$", prop={'size': fontsize}, ncol =2)            

def simulate(runs, Nt, start_recording, Lx, Ly, init, mu, sigma, v_slow, v_fast, p_fast):
    recording_time = Nt-start_recording
    current = np.zeros(recording_time)
    empty_sites = np.zeros(recording_time)
    fast_chosen = np.zeros(recording_time)    
    slow_chosen = np.zeros(recording_time)
    fast_sites = np.zeros(Nt)      # fast particles in the right region
    slow_sites = np.zeros(Nt)
    density_fast = np.zeros(Ly)
    density_slow = np.zeros(Ly)
    YcurrentII_fast = np.zeros(Ly) # parallel current for fast particles
    YcurrentII_slow = np.zeros(Ly) 
    YcurrentT_fast = np.zeros(Ly)  # perpendicular current for fast particles
    YcurrentT_slow = np.zeros(Ly)

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
            lattice = checkboard_gaussian(Lx, Ly, mu, sigma)

        elif init == 'random_gauss':
            lattice = random_system_gaussian(Lx, Ly, N, mu, sigma)
        
        elif init == 'chess_types':
            lattice = checkboard_types(Lx, Ly, v_slow, v_fast, p_fast)            

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
                total_fast, fast_up  = 0, 0
                total_slow, slow_down = 0, 0

                selectedX = random.randint(0, Lx-1)
                selectedY = random.randint(0, Ly-1)

                if lattice[selectedX][selectedY] != 0:  
                    speed = lattice[selectedX][selectedY]
                    # counting of selected fast and slow particles   
                    if speed > 0.9:
                        selected_fast += 1/(Lx*Ly*runs)
                    elif speed <= 0.9:
                        selected_slow += 1/(Lx*Ly*runs)                     
                    # update position: we will throw three dices -- one for direction (|| or perp), one for perpendicular direction, and one for speed
                    newX = -1
                    newY = -1
                    nextX = selectedX + 1 if selectedX < Lx - 1 else 0
                    nextY = selectedY + 1 if selectedY < Ly - 1 else 0
                    prevY = selectedY - 1 if selectedY > 0 else Ly - 1
                    
                    direction = random.randint(0,1)
                    if direction == 0: # jump to the right
                        newX = nextX
                        newY = selectedY
                    else: # jump top or bottom
                        p = compute_prob(k, selectedY, Ly, speed) # transitional probabilities depend on the particle's position, this is where potential plays a role
                        dice = random.random()
                        if dice < p:
                            newY = nextY
                            newX = selectedX
                        else:
                            newY = prevY
                            newX = selectedX

                    jump_dice = random.random()  
                #                     
                    if jump_dice <= speed:
                        if lattice[newX][newY] == 0:
                            lattice[selectedX][selectedY] = 0
                            lattice[newX][newY] = speed
                            # record current
                            if newX != selectedX: # we have jump forward
                                if t >= start_recording:
                                    total_current += 1/(Lx*Ly*runs)
                                    if speed > 0.9: #fast particles
                                        YcurrentII_fast[selectedY] += 1/(Lx*Ly*recording_time*runs)
                                    elif speed <= 0.9: #slow particles
                                        YcurrentII_slow[selectedY] += 1/(Lx*Ly*recording_time*runs)
                            elif newY == nextY: #moved up                        
                                if t >= start_recording:
                                    if speed > 0.9:
                                        YcurrentT_fast[selectedY] += 1/(Lx*Ly*recording_time*runs)
                                    elif speed <= 0.9:
                                        YcurrentT_slow[selectedY] += 1/(Lx*Ly*recording_time*runs)
                            elif newY == prevY: #moved down
                                if t > start_recording:
                                    if speed > 0.9:
                                        YcurrentT_fast[selectedY] -= 1/(Lx*Ly*recording_time*runs)
                                    elif speed <= 0.9:
                                        YcurrentT_slow[selectedY] -= 1/(Lx*Ly*recording_time*runs)
                                    
                else:
                #    print("empty site chosen")
                    selected_empty_site += 1/(Lx*Ly*runs)
                # movie
                if divisions == True:
                    if k % movie_divisions == 0:  # Check if k is a multiple of movie_divisions
                        one_movie[k//movie_divisions] = lattice  # Store lattice in one_movie at every movie_divisions-th index                    
                else:
                    one_movie[k] = lattice
                k += 1                    

            # counting particles in their respective areas
            right_fast, right_slow = 0, 0
            fast_up, slow_down  = 0, 0                        
            total_fast, total_slow  = 0, 0                        
            for i in range(Lx):
                for j in range(Ly):
                    if lattice[i][j] > 0.9:
                        total_fast += 1
                        if j <= boundary_lane and j != 0:
                            fast_up += 1
                    elif lattice[i][j] <= 0.9 and lattice[i][j] != 0:
                        total_slow += 1
                        if j >= boundary_lane or j == 0:
                            slow_down += 1
            right_fast += fast_up /(total_fast*runs) if total_fast != 0 else 0
            right_slow += slow_down / (total_slow*runs) if total_slow != 0 else 0 
            # # density across the lanes
            # for j in range(Ly):
            #     if lattice[0][j] > 0.9:
            #         density_fast[j] += 1/(Lx*Ly*Ly*recording_time*runs)
            #     elif lattice[0][j] <= 0.9 and lattice[0][j] != 0:
            #         density_slow[j] += 1/(Lx*Ly*Ly*recording_time*runs)


            one_short_movie[t + 1] = lattice

            if t >= start_recording:
                current[m] += total_current #sum of the currents of all runs
                empty_sites[m] += selected_empty_site
                fast_chosen[m] += selected_fast
                slow_chosen[m] += selected_slow
                m += 1

            fast_sites[t] += right_fast
            slow_sites[t] += right_slow
    return current, empty_sites, fast_chosen, slow_chosen, fast_sites, slow_sites, YcurrentII_fast, YcurrentII_slow, YcurrentT_fast, YcurrentT_slow, density_fast, density_slow, one_movie


# Main
if __name__ == '__main__': 
    ############# Lattice simulation parameters #############
    density = 0.5              # work with half-density
    Lx = 20
    Ly = 10
    N = int(Lx*Ly*density)
    k = 1/2*(Ly)               # spring constant
    boundary_lane = int(Ly/2)  # the regions are: fast [0, (boundary_lane-1)] and slow [boundary lane, (Ly-1)]
    runs = 100
    Nt = 200                   # episode duration
    start_recording = 100      # for steady state measurements
    recording_time = Nt-start_recording 
    init = "random_types"
    # two types particles
    v_slow = 0.5
    v_fast = 1
    p_fast = 0.5

    # gaussian particles # work with half-density
    mu = 0.5
    sigma = 0.01

    current, empty_sites, fast_chosen, slow_chosen, fast_sites, slow_sites, YcurrentII_fast, YcurrentII_slow, YcurrentT_fast, YcurrentT_slow, density_fast, density_slow, one_movie = simulate(runs, Nt, start_recording, Lx, Ly, init, mu, sigma, v_slow, v_fast, p_fast)

    plot_current(current)
    plt.savefig(f"./Current_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)
    # plot_empty_sites(empty_sites)
    # plt.savefig(f"./Empty_Sites_Chosen_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)
    # plot_type_particles(fast_chosen, slow_chosen)
    # plt.savefig(f"./Type_Particle_Chosen_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)                    
    # plot_particle_occupation(fast_sites, slow_sites)
    # plt.savefig(f"./Particle_Occupation{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)
    plot_YcurrentII(YcurrentII_fast, YcurrentII_slow)
    plt.savefig(f"./Y_CurrentII_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600) 
    plot_YcurrentT(YcurrentT_fast, YcurrentT_slow)
    plt.savefig(f"./Y_CurrentT_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)  
    # plot_lane_density(density_fast,density_slow)
    # plt.savefig(f"./lane_density_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)  

    folder_path = "output_Attractive_TASEP"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = "AttractiveTASEP_current_" + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    file_path = os.path.join(folder_path, filename)                    
    with open(file_path, 'w') as f:
        for t in range(recording_time):
            output_string = str(t) + "\t" + str(current[t]) + "\n"
            f.write(output_string)

    filename = "AttractiveTASEP_empty_sites_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    file_path = os.path.join(folder_path, filename)                    
    with open(file_path, 'w') as f:
        for t in range(recording_time):
            output_string = str(t) + "\t" + str(empty_sites[t]) + "\n"
            f.write(output_string)

    filename = "AttractiveTASEP_fast_right_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    file_path = os.path.join(folder_path, filename)                    
    with open(file_path, 'w') as f:
        for t in range(Nt):
            output_string = str(t) + "\t" + str(fast_sites[t]) + "\n"
            f.write(output_string)

    filename = "AttractiveTASEP_slow_right_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    file_path = os.path.join(folder_path, filename)                    
    with open(file_path, 'w') as f:
        for t in range(Nt):
            output_string = str(t) + "\t" + str(slow_sites[t]) + "\n"
            f.write(output_string)


    filename = "AttractiveTASEP_YcurrentII_fast_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    file_path = os.path.join(folder_path, filename)                    
    with open(file_path, 'w') as f:
            for y in range(Ly):
                output_string = str(YcurrentII_fast[y]) + "\n"
                f.write(output_string) 

    filename = "AttractiveTASEP_YcurrentII_slow_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    file_path = os.path.join(folder_path, filename)                    
    with open(file_path, 'w') as f:
            for y in range(Ly):
                output_string = str(YcurrentII_slow[y]) + "\n"
                f.write(output_string) 

    filename = "AttractiveTASEP_YcurrentT_fast_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    file_path = os.path.join(folder_path, filename)                    
    with open(file_path, 'w') as f:
        for y in range(Ly):
            output_string =  str(YcurrentT_fast[y]) + "\n"
            f.write(output_string) 

    filename = "AttractiveTASEP_YcurrentT_slow_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    file_path = os.path.join(folder_path, filename)                    
    with open(file_path, 'w') as f:
        for y in range(Ly):
            output_string =  str(YcurrentT_slow[y]) + "\n"
            f.write(output_string) 

    filename = "AttractiveTASEP_fast_chosen"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    file_path = os.path.join(folder_path, filename)                    
    with open(file_path, 'w') as f:
        for t in range(recording_time):
            output_string = str(t) + "\t" + str(fast_chosen[t]) + "\n"
            f.write(output_string)

    filename = "AttractiveTASEP_slow_chosen"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    file_path = os.path.join(folder_path, filename)                    
    with open(file_path, 'w') as f:
        for t in range(recording_time):
            output_string = str(t) + "\t" + str(slow_chosen[t]) + "\n"
            f.write(output_string)   

    # filename = "Smart_AttractiveTASEP_lane_density_fast_" + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    # with open(filename, 'w') as f:
    #     for y in range(Ly):
    #         output_string = str(density_fast[y]) + "\n"
    #         f.write(output_string)

    # filename = "Smart_AttractiveTASEP_lane_density_slow_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
    # with open(filename, 'w') as f:
    #     for y in range(Ly):
    #         output_string = str(density_slow[y]) + "\n"
    #         f.write(output_string) 


    # Movie
    # print('Action! (recording movie)')
    # straight_movie = np.transpose(one_movie, axes=(0, 2, 1))

    # ani = create_animation(straight_movie[:Nt]) #last run
    # HTML(ani.to_jshtml()) # interactive python
    # ani.save("./Movie"+ str(Lx) + "x" + str(Ly)  + "_runs" + str(runs) + ".gif", fps = 8)
    # print('Cut! (movie ready)')

    # 6x6, 50 fps
    # 12x12 movie_divisions = 500, 8 fps
    # 20x10 movie_divisions = 1000, 8 fps
    # 50x20 movie_divisions = 1200, 8 fps

    #saving the movie array in a file
    # filename = "movie_storage" + ".pkl"
    # with open(filename, 'wb') as file:
    #     if divisions == True: 
    #         movie_storage = np.zeros(((Lx*Ly)*Nt//movie_divisions + 1, Lx, Ly))
    #     else:
    #         movie_storage = np.zeros(((Lx*Ly)*Nt + 1, Lx, Ly))

    #     movie_storage = straight_movie            
    #     pickle.dump(movie_storage, file)
#-----------------------------------------------------------------------------------------------
    # init = 'chess_types'
    # start_recording = 0
    # currents_types_plot()
    # plt.savefig(f'./Attract_Current_Different_Types_{runs}_{Lx}x{Ly}.pdf', format="pdf", dpi=600, bbox_inches = 'tight')
    # currents_ratios_plot()
    # plt.savefig(f"./Current_Different_Ratios_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)
