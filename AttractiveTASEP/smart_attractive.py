import math
import numpy as np
import random
from numpy import random as Random
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple, deque
import matplotlib.pyplot as plt
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
colors = sns.color_palette("Set2")
colors2 = sns.color_palette("Pastel1")
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def checkboard_types(Lx, Ly, N, v_slow, v_fast, p_fast): 
    system = np.zeros(shape=(Lx,Ly))
    system[::2, ::2] = 1  # Set even rows and even columns to 1
    system[1::2, 1::2] = 1  # Set odd rows and odd columns to 1
    for i in range(Lx):
        for j in range(Ly):
            if system[i][j] == 1:
                system[i][j] = Random.choice([v_slow, v_fast], p =[1-p_fast, p_fast])
                    
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

# structure of the Q table
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# class that defines the Q table
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # deque is a more efficient form of a list

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, hidden_size, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

def get_state_system(system, Lx, Ly, Xcenter, Ycenter, L):
 # creates the input for the NN from the system (Lx x Ly)
 # the state consists on two channels: two patches (fast and slow particles) taken from the system with dimensions (L x L) and centered at (Xcenter, Ycenter),
 # and a number that indicates the distance of the patch to the center in the y-axis.
    
    # defining the training patch
    fast_channel = np.zeros(shape=(L,L)) 
    slow_channel = np.zeros(shape=(L,L)) 

    # calculate the index offsets of the patch from the center of the system
    x_offset = Xcenter - int(L / 2)
    y_offset = Ycenter - int(L / 2)
    for i in range(L):
        x_index = x_offset + i
        x_index = x_index % Lx # periodic boundaries
        
        for j in range(L):
            y_index = y_offset + j
            y_index = y_index % Ly # periodic boundaries

            value = system[x_index][y_index]
            if value == 1:
                fast_channel[i][j] = 1
            elif value != 0:
                slow_channel[i][j] = 1
                
    patch = np.concatenate((fast_channel, slow_channel), axis= None)
    fast_center = int(Ly/4)
    slow_center = int(3*Ly/4)
    distance_fast_center = min(abs(Ycenter - fast_center), abs(Ycenter - Ly-1) + fast_center) # to account for PBC
    distance_slow_center = min(abs(Ycenter - slow_center), abs(Ycenter + Ly-1) - slow_center)
    distances = [distance_fast_center, distance_slow_center]
    state = np.append(patch, distances) # added distance to edges

    return state

# i am assuming here that fasl/slow regions have the same size and that fast particles have the speed 1, and slow < 1
def count_wrong_particles(lattice, Lx,Ly):
    count_slow = 0
    count_fast = 0
    border_line = Ly/2
    for x in range(Lx):
        for y in range(Ly):
            if lattice[x][y] != 0:
                if lattice[x][y] == 1 and y > border_line - 1:
                    count_fast +=1
                elif lattice[x][y] != 1 and y < border_line:
                    count_slow +=1 

    return count_slow, count_fast

# i will choose quadratic attractive potential -r^2 around centers of each zones
def compute_potential_energy(lattice,Lx,Ly):
    energy = 0
    fast_center = int(Ly/4)
    slow_center = int(3*Ly/4)
    for x in range(Lx):
        for y in range(Ly):
            if lattice[x][y] != 0:
                if lattice[x][y] == 1:
                    measure_distance = min(abs(y - fast_center), abs(y - Ly-1) + fast_center) # to account for PBC
                    distance = 1 if measure_distance == 0 else measure_distance
                    energy += -1*(distance*distance)
                else:
                    measure_distance = min(abs(y - slow_center), abs(y + Ly-1) - slow_center)
                    distance = 1 if measure_distance == 0 else measure_distance
                    energy += -1*(distance*distance)

    return energy

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

def get_coordinates_from_patch(x, y, Xcenter, Ycenter, L, Lx, Ly):
 # translates the lattice site (x, y) from the patch to the system reference (x_sys, y_sys)
 # the training patch is centered around (Xcenter, Ycenter) with (L x L) dimensions, 
 # and the system has (Lx x Ly) dimensions
    
    # translate the coordinates to be relative to the center of the patch
    relative_x = x - int(L/ 2)
    relative_y = y - int(L/ 2)
    
    # translate to system reference
    x_sys = Xcenter + relative_x
    y_sys = Ycenter + relative_y
    
    # periodic boundary conditions
    x_sys = x_sys % Lx
    y_sys = y_sys % Ly

    return x_sys, y_sys    

def select_action_training(state): 
    global steps_done # count total number of steps to go from almost random exploration to more efficient actions
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold: # exploitation
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1) # view(1,1) changes shape to [[action], dtype]
    else:
        # select a random action
        rand_action = random.randint(0,L*L-1) # random lattice site in the observation patch
        return torch.tensor([[rand_action]], device=device, dtype=torch.long)

def select_action_post_training(state): 
    # interpret Q values as probabilities when simulating dynamics of the system 
    # in principle this could be easily extended to make this more general, but i am a lazy boi
    with torch.no_grad():
        # print("state ", state)
        Q_values = trained_net(state)
        # print("Q-values ", Q_values)
        probs = torch.softmax(Q_values, dim=1) # converts logits to probabilities (torch object)
        # print("probs ", probs)
        dist = Categorical(probs) # feeds torch object to generate a list of probs (numpy object ?)
        # print("dist ", dist)
        action = dist.sample().numpy()[0] # sample list of probs and return the action

        return action

# move
def step(lattice, X, Y, L, Xcenter, Ycenter, k, log = False):
    newX = -1
    newY = -1
    speed = lattice[X][Y]
    # periodic boundaries
    Lx, Ly = lattice.shape
    nextX = X + 1 if X < Lx - 1 else 0
    prevX = X - 1 if X > 0 else Lx - 1
    nextY = Y + 1 if Y < Ly - 1 else 0
    prevY = Y - 1 if Y > 0 else Ly - 1

    # update position
    direction = random.randint(0,1)
    if direction == 0: # jump to the right
        newX = nextX
        newY = Y
    else: # jump top or bottom
        p = compute_prob(k, Y, Ly, speed) # transitional probabilities depend on the particle's position, this is where potential plays a role
        dice = random.random()
        if dice < p:
            newY = nextY
            newX = X
        else:
            newY = prevY
            newX = X

    current_along = 0
    reward = 1 # simply for choosing a particle and not an empty space

    # idea: move particles to reduce their energy
    energy_before = compute_potential_energy(lattice,Lx,Ly)


    up = 1 if lattice[X][prevY] != 0 else 0
    down = 1 if lattice[X][nextY] != 0 else 0
    forward = 1 if lattice[nextX][Y] != 0 else 0
     
    reward += int(-1*(up + down) - 1*(2*forward - 1)) # punish for all blocked neighbors, reward if the forward lattice site is empty
    # reward += int(-1*(lattice[X][prevY] + lattice[X][nextY]) - 1*(2*lattice[nextX][Y] - 1)) # punish for all blocked neighbors, reward if the forward lattice site is empty

    # do the position update
    jump_dice = random.random()
    if jump_dice <= speed:
        if lattice[newX][newY] == 0: # next site free
            lattice[X][Y] = 0
            lattice[newX][newY] = speed     
            if newX != X: # we have jump forward
                current_along = 1
                reward += 1
        else:          
            if log == True:
                print("  obstacle: it couldn't jump :(")                         
    else:
        if log == True:
            print("  no speed: it couldn't jump :(")

    # try potential 
    energy_after = compute_potential_energy(lattice,Lx,Ly)

    # reward += -5*int(energy_before - energy_after)
    reward += -10*int(energy_before - energy_after) 
    #reward += int(0.05*energy_after + 10)
    
    next_state = get_state_system(lattice, Lx, Ly, Xcenter, Ycenter, L)

    return reward, next_state, current_along

def optimize_model():
    if len(memory) < BATCH_SIZE: # execute 'optimize_model' only if #BATCH_SIZE number of updates have happened 
        return
    transitions = memory.sample(BATCH_SIZE) # draws a random set of transitions; the next_state for terminal transition will be NONE
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions)) # turn [transition, (args)] array into [[transitions], [states], [actions], ... ]

    # Compute a mask of non-final states and concatenate the batch elements (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s != None, batch.next_state)), device=device, dtype=torch.bool) # returns a set of booleans
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # creates a list of non-empty next states
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Policy_net produces [[Q1,...,QN], ...,[]] (BATCH x N)-sized matrix, where N is the size of action space, 
    # and action_batch is BATCH-sized vector whose values are the actions that have been taken. 
    # Gather tells which Q from [Q1,...,QN] row to take, using action_batch vector, and returns BATCH-sized vector of Q(s_t, a) values
    state_action_values = policy_net(state_batch).gather(1, action_batch) # input = policy_net, dim = 1, index = action_batch

    # Compute Q^\pi(s_t,a) values of actions for non_final_next_states by using target_net (old policy_net), from which max_a{Q(s_t, a)} are selected with max(1)[0].
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0] # target_net produces a vector of Q^pi(s_t+1,a)'s and max(1)[0] takes maxQ
    # Compute the expected Q^pi(s_t,a) values for all BATCH_SIZE (default=128) transitions
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def do_training(num_episodes, L, density, Nt, Lx, Ly, init, k, log = False):
    for i_episode in tqdm(range(num_episodes)):
        N = int(Lx*Ly*density) 
        if init == 'chess_types':
            lattice = checkboard_types(Lx, Ly, N, v_slow, v_fast, p_fast)            

        else: # start with random initial conditions
            lattice = np.zeros(shape=(Lx,Ly))
            n = 0
            while n < N:
                X = random.randint(0, Lx-1)
                Y = random.randint(0, Ly-1)
                if lattice[X][Y] == 0:
                    lattice[X][Y] = Random.choice([v_slow, v_fast], p =[1-p_fast, p_fast])
                    n += 1

        # main update loop; I use Monte Carlo random sequential updates here
        score = 0
        total_current = 0
        selected_empty_site = 0
        selected_fast = 0
        selected_slow = 0 
        for t in range(Nt):
            for i in range(Lx*Ly):
                # random sampling in the lattice to apply the training
                Xcenter = random.randint(0, Lx-1)
                Ycenter = random.randint(0, Ly-1)           

                state = get_state_system(lattice, Lx, Ly, Xcenter, Ycenter, L)
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                action = select_action_training(state) # get the index of the particle
                lattice_site = action.item() # a number, and we encode it as x*L + y
                patchX = int(lattice_site / L)
                patchY = int(lattice_site % L)
                selectedX, selectedY = get_coordinates_from_patch(patchX, patchY, Xcenter, Ycenter, L, Lx, Ly)
                                                                                                                              
                if lattice[selectedX][selectedY] != 0:
                    # counting of selected fast and slow particles
                    # if lattice[selectedX][selectedY] > 0.5:
                    #     selected_fast += 1
                    # elif lattice[selectedX][selectedY] <= 0.5:
                    #     selected_slow += 1                    
                    # update particle's position and do stochastic part                                         
                    reward, next_state, current_along = step(lattice, selectedX, selectedY, L, Xcenter, Ycenter, k, log) 
                    total_current += current_along / (Lx*Ly*Nt)
                    reward = torch.tensor([reward], device=device)
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0) 
                    memory.push(state, action, next_state, reward)                  
                else: # empty site chosen
                    reward = -10
                    selected_empty_site += 1
                    reward = torch.tensor([reward], device=device)  
                    memory.push(state, action, state, reward)

                # add for cummulative score
                score += reward

            # counting particles in their respective areas
            right_fast, right_slow = 0, 0
            fast_up, slow_down  = 0, 0                        
            total_fast, total_slow  = 0, 0   
            for i in range(Lx):
                for j in range(Ly):
                    if lattice[i][j] > 0.9:
                        total_fast += 1
                        if j <= boundary_lane and j != 0: # for 10 lanes
                            fast_up += 1
                    elif lattice[i][j] <= 0.9 and lattice[i][j] != 0: 
                        total_slow += 1
                        if j > boundary_lane or j == 0: # for 10 lanes
                            slow_down += 1
            right_fast += fast_up / total_fast if total_fast != 0 else 0
            right_slow += slow_down / total_slow if total_slow != 0 else 0            

            optimize_model()
            # Soft update of the target network's weights: θ′ ← τ θ + (1 −τ)θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
  

        # slow_after, fast_after = count_wrong_particles(lattice,Lx,Ly) # number of particles in the wrong region
        # total_wrong = slow_after + fast_after
        # wrong_count.append(total_wrong)

        print("Training episode ", i_episode, " is over. Current = ", total_current, "; Selected empty sites / L*L = ", selected_empty_site / (Lx*Ly*Nt))             
        # print("Fast particles chosen ", selected_fast/ (Lx*Ly*Nt), ". Slow particles chosen = ", selected_slow / (Lx*Ly*Nt))
        print("Fast particles in the upper side ", right_fast, ". Slow particles in the lower side = ", right_slow) # already normalized by total_fast & slow
        # print("Number of wrong particles ", total_wrong)

        rewards.append(score.numpy()) 
        current.append(total_current)
        empty_sites.append(selected_empty_site / (Lx*Ly*Nt))
        # fast_chosen.append(selected_fast / (Lx*Ly*Nt))
        # slow_chosen.append(selected_slow / (Lx*Ly*Nt))
        # fast_sites.append(right_fast / (Lx*Ly*Nt))
        # slow_sites.append(right_slow / (Lx*Ly*Nt))


    torch.save(target_net.state_dict(), PATH)
    plot_score() 
    plt.savefig(f"./AttractiveTASEP_Training_Reward_{Nt}steps_{Lx}x{Ly}_{k}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')
    plot_training(current, empty_sites)
    plt.savefig(f"./AttractiveTASEP_Training_Plot_{Nt}steps_{Lx}x{Ly}_{k}.pdf", format="pdf", dpi=600, bbox_inches = 'tight')        
    # plt.savefig(f"./AttractiveTASEP_Training_Current_{Lx}x{Ly}.pdf", format="pdf", dpi=600, dpi=600, bbox_inches = 'tight')
    # plot_empty_sites(empty_sites)
    # plt.savefig(f"./AttractiveTASEP_Training_Empty_Sites{Lx}x{Ly}.pdf", format="pdf", dpi=600, dpi=600, bbox_inches = 'tight')

# plots
def plot_score():
    fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize))
    ax.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))    
    ax.set(xlabel = 'Episodes', ylabel = 'Cumulative reward')
    ax.plot(rewards, color = colors[4])

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
def plot_training(current, empty_sites):
    fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize)) 
    ax.tick_params(axis='both', which='major', direction='in', length = 3, labelsize=fontsize)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))    
    ax.set_ylim(0, np.max(current)*1.05)
    ax.set(xlabel = 'Episodes')

    # moving average    
    window_size = 20     
    moving_avg_current = moving_average(current, window_size)
    moving_avg_empty = moving_average(empty_sites, window_size)
    ax.plot(current, color = colors[1], label = 'Current')
    ax.plot(moving_avg_current, color = 'crimson', label = 'Current (moving average)')
    ax.plot(empty_sites, color = colors2[3], label = 'Prob. of selecting empty sites')
    ax.plot(moving_avg_empty, color = 'indigo', label = 'Prob. of selecting empty sites \n (moving average)')        
    ax.legend(loc=0, prop={'size': fontsize}, ncol = 1)    

def plot_current(current, post = False):
    fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize))
    if post == True:
        ax.set(xlabel = 'Time step', ylabel = f'Average current over {runs} runs')
    else:
        ax.set(xlabel = 'Time step', ylabel = 'Current')

    ax.set_ylim(0, np.max(current)*1.05)
    ax.plot(current, color = 'darkmagenta')

def plot_empty_sites(empty_sites, post = False):
    fig, ax = plt.subplots(figsize=(fig_xsize, fig_ysize))
    if post == True:
        ax.set(xlabel = 'Time step', ylabel = f'Selected empty sites. Averaged over {runs} runs')
    else:
        ax.set(xlabel = 'Time step', ylabel = 'Selected empty sites')

    ax.set_ylim(0, np.max(empty_sites)*1.05)    
    ax.plot(empty_sites, color = 'darkmagenta')

def plot_type_particles(fast_chosen, slow_chosen):
    fig, ax = plt.subplots(figsize=(fig_xsize-1, fig_ysize-0.5))
    ax.set(xlabel = 'Time step', ylabel = f'Probability of particle selected. \n Averaged over {runs} runs')
    # ax.set_ylim(0, 0.3)    
    ax.plot(fast_chosen, label = 'Fast particles', color = colors[1])
    ax.plot(slow_chosen, label = 'Slow particles', color = colors[2])
    ax.legend(loc=0, prop={'size': fontsize}, ncol = 1)

def plot_lane_density(density_fast, density_slow):
    plt.clf() 
    plt.xlabel('Vertical lattice site')
    plt.ylabel(f'Average probability over {runs} runs')

    # plt.ylim(0, np.max(density_fast)*1.10)    
    plt.plot(density_fast, label = 'Fast particles', color = colors[5])
    plt.plot(density_slow, label = 'Slow particles', color = colors[1])
    plt.legend(loc='lower right')  

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


def plot_particle_occupation(fast_sites, slow_sites, post = False):
    plt.clf() 
    plt.figure(7)
    plt.title(f"Attractive TASEP. Probability of a particle to be in its region")
    if post == True:
        plt.xlabel('Episode duration')
        plt.ylabel(f'Probability over {runs} runs')    
    else:
        plt.xlabel('Episodes')
        plt.ylabel(f'Particles in their regions over {Lx} x {Ly} lattice')     

    plt.ylim(0, np.max(fast_sites)*1.10)    
    plt.plot(fast_sites, label = 'fast particles', color = colors[5])
    plt.plot(slow_sites, label = 'slow particles', color = colors[1])
    plt.legend(loc='lower right')
    

# Main
if __name__ == '__main__': 
    Jessie_we_need_to_train_NN = True
    Post_training = False
    log = False
    log_post = False
    ############# Model parameters for Machine Learning #############
    num_episodes = 500      # number of training episodes
    BATCH_SIZE = 200        # the number of transitions sampled from the replay buffer
    GAMMA = 0.99            # the discounting factor
    EPS_START = 0.9         # EPS_START is the starting value of epsilon; determines how random our action choises are at the beginning
    EPS_END = 0.001         # EPS_END is the final value of epsilon
    EPS_DECAY = 200         # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = 0.005             # TAU is the update rate of the target network
    LR = 1e-3               # LR is the learning rate of the AdamW optimizer
    ############# Lattice simulation parameters #############
    init = 'random_types'
    # two types particles
    v_slow = 0.5
    v_fast = 1
    p_fast = 0.5
    L = 5                   # squared patches for the training
    density = 0.5           # work with half-density
    Lx = 20
    Ly = 10
    N = int(Lx*Ly*density)
    k = 1/2*(Ly)               # spring constant
    boundary_lane = int(Ly/2)  # the regions are: fast [0, (boundary_lane-1)] and slow [boundary lane, (Ly-1)]
    Nt = 200                   # episode duration
    n_observations = 2*L*L + 2 # three channels. Two of the patch size: fast and slow particles and one with the distance to the center; added two more numbers
    n_actions = L*L            # patch size, in principle, the empty spots can also be selected
    hidden_size = 128          # hidden size of the network
    PATH = f"./2d_attrakt_NN_params_{Nt}steps_{Lx}x{Ly}.txt"
    start_recording = 0

    ############# Do the training if needed ##############
    if Jessie_we_need_to_train_NN:
        policy_net = DQN(n_observations, hidden_size, n_actions).to(device)
        target_net = DQN(n_observations, hidden_size, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(100*Nt) # the overall memory batch size 
        rewards = []
        wrong_count = []
        current = []
        empty_sites = []
        fast_chosen = []
        slow_chosen = []
        fast_sites = []
        slow_sites = []
        steps_done = 0

        do_training(num_episodes, L, density, Nt, Lx, Ly, init, k, log) 

    ############# Post-training simulation ##############
    if Post_training:
        runs = 100
        Lx = 20
        Ly = 10
        boundary_lane = int(Ly/2)
        k = 1/2*(Ly) 
        Nt = 200
        start_recording = 100           # for steady state measurements
        recording_time = Nt-start_recording        
        trained_net = DQN(n_observations, hidden_size, n_actions).to(device)
        trained_net.load_state_dict(torch.load(PATH))

        current = np.zeros(recording_time)
        empty_sites = np.zeros(recording_time)
        fast_chosen = np.zeros(recording_time)    
        slow_chosen = np.zeros(recording_time)
        fast_sites = np.zeros(Nt) # fast particles in the right region
        slow_sites = np.zeros(Nt)
        density_fast = np.zeros(Ly)
        density_slow = np.zeros(Ly)        
        YcurrentII_fast = np.zeros(Ly) # parallel current for fast particles
        YcurrentII_slow = np.zeros(Ly) 
        YcurrentT_fast = np.zeros(Ly)  # perpendicular current for fast particles
        YcurrentT_slow = np.zeros(Ly)

        for run in tqdm(range(runs)):
            N = int(Lx*Ly*density) 
            if init == 'chess_types':
                lattice = checkboard_types(Lx, Ly, N, v_slow, v_fast, p_fast)            

            else: # start with random initial conditions
                lattice = np.zeros(shape=(Lx,Ly))
                n = 0
                while n < N:
                    X = random.randint(0, Lx-1)
                    Y = random.randint(0, Ly-1)
                    if lattice[X][Y] == 0:
                        lattice[X][Y] = Random.choice([v_slow, v_fast], p =[1-p_fast, p_fast])
                        n += 1
            m = 0
            for t in tqdm(range(Nt)):
                total_current = 0
                selected_empty_site = 0
                selected_fast = 0
                selected_slow = 0                

                for move_attempt in range(Lx*Ly):
                   # Random sampling of the new lattice to apply the training
                    Xcenter = random.randint(0, Lx-1)
                    Ycenter = random.randint(0, Ly-1)

                    state = get_state_system(lattice, Lx, Ly, Xcenter, Ycenter, L)
                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = select_action_post_training(state)
                    lattice_site = action.item() # a number, and we encode it as x*L + y
                    patchX = int(lattice_site / L)
                    patchY = int(lattice_site % L)
                    selectedX, selectedY = get_coordinates_from_patch(patchX, patchY, Xcenter, Ycenter, L, Lx, Ly)

                    # to check how does the simulation perform for the random "untrained" simulation, 
                    # uncomment two lines below and comment the line above
                    # selectedX = random.randint(0, Lx-1)
                    # selectedY = random.randint(0, Ly-1)

                    if lattice[selectedX][selectedY] != 0:
                        # counting of selected fast and slow particles   
                        speed = lattice[selectedX][selectedY]                     
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
                        if jump_dice <= speed:
                            if lattice[newX][newY] == 0:
                                lattice[selectedX][selectedY] = 0
                                lattice[newX][newY] = speed
                                # record current
                                if newX != selectedX: # we have jump forward
                                    if t >= start_recording:
                                        current[m] += 1/(Lx*Ly*runs)
                                        if speed > 0.9:
                                            YcurrentII_fast[selectedY] += 1/(Lx*Ly*recording_time*runs)
                                        elif speed <= 0.9:
                                            YcurrentII_slow[selectedY] += 1/(Lx*Ly*recording_time*runs)
                                elif newY == nextY:
                                    if log == True:
                                        print("  moved up")
                                    
                                    if t >= start_recording:
                                        if speed > 0.9:
                                            YcurrentT_fast[selectedY] += 1/(Lx*Ly*recording_time*runs)
                                        elif speed <= 0.9:
                                            YcurrentT_slow[selectedY] += 1/(Lx*Ly*recording_time*runs)
                                elif newY == prevY:
                                    if log == True:
                                        print("  moved down")
                                    
                                    if t >= start_recording:
                                        if speed > 0.9:
                                            YcurrentT_fast[selectedY] -= 1/(Lx*Ly*recording_time*runs)
                                        elif speed <= 0.9:
                                            YcurrentT_slow[selectedY] -= 1/(Lx*Ly*recording_time*runs)          
                                                                    
                    else:
                        if log == True:
                            print("ALARM! ALARM!")                            
                            print("empty site chosen")
                        selected_empty_site += 1/(Lx*Ly*runs)

                # counting particles in their respective areas    
                right_fast, right_slow = 0, 0
                fast_up, slow_down  = 0, 0                        
                total_fast, total_slow  = 0, 0                            
                for i in range(Lx):
                    for j in range(Ly):
                        if lattice[i][j] > 0.9:
                            total_fast += 1
                            if j <= boundary_lane and j != 0: # for 10 lanes
                                fast_up += 1
                        elif lattice[i][j] <= 0.9 and lattice[i][j] != 0: 
                            total_slow += 1
                            if j > boundary_lane or j == 0: # for 10 lanes
                                slow_down += 1
                right_fast += fast_up / (total_fast*runs) if total_fast != 0 else 0
                right_slow += slow_down / (total_slow*runs) if total_slow != 0 else 0                          

                # density across the lanes
                for j in range(Ly):
                    if lattice[0][j] > 0.9:
                        density_fast[j] += 1/(Lx*Ly*Ly*recording_time*runs)
                    elif lattice[0][j] <= 0.9 and lattice[0][j] != 0:
                        density_slow[j] += 1/(Lx*Ly*Ly*recording_time*runs)

                if t >= start_recording:
                    current[m] += total_current #sum of the currents of all runs
                    empty_sites[m] += selected_empty_site
                    fast_chosen[m] += selected_fast
                    slow_chosen[m] += selected_slow              
                    m += 1
                
                fast_sites[t] += right_fast
                slow_sites[t] += right_slow


        # plot_current(current, post = True)
        # plt.savefig(f"./Attractive_TASEP_Post_Current_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)
        # plot_empty_sites(empty_sites, post = True)
        # plt.savefig(f"./AttractiveTASEP_Post_Empty_Sites_Chosen_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)
        plot_type_particles(fast_chosen, slow_chosen)
        plt.savefig(f"./AttractiveTASEP_Post_Type_Particle_Chosen_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)                    
        # plot_particle_occupation(fast_sites, slow_sites, post = True)
        # plt.savefig(f"./AttractiveTASEP_Post_Particle_Occupation{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)
        # plot_YcurrentII(YcurrentII_fast, YcurrentII_slow)
        # plt.savefig(f"./AttractiveTASEP_Post_Y_CurrentII_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600) 
        # plot_YcurrentT(YcurrentT_fast, YcurrentT_slow)
        # plt.savefig(f"./AttractiveTASEP_Post_Y_CurrentT_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)  
        # plot_lane_density(density_fast,density_slow)
        # plt.savefig(f"./last_results/lane_density_{runs}_{Lx}x{Ly}.pdf", format="pdf", dpi=600)


        folder_path = "output_smart_Attractive_TASEP"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filename = "Smart_AttractiveTASEP_current_" + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
        file_path = os.path.join(folder_path, filename)                    
        with open(file_path, 'w') as f:
            for t in range(recording_time):
                output_string = str(t) + "\t" + str(current[t]) + "\n"
                f.write(output_string)

        filename = "Smart_AttractiveTASEP_empty_sites_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
        file_path = os.path.join(folder_path, filename)                    
        with open(file_path, 'w') as f:
            for t in range(recording_time):
                output_string = str(t) + "\t" + str(empty_sites[t]) + "\n"
                f.write(output_string)

        filename = "Smart_AttractiveTASEP_fast_right_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
        file_path = os.path.join(folder_path, filename)                    
        with open(file_path, 'w') as f:
            for t in range(Nt):
                output_string = str(t) + "\t" + str(fast_sites[t]) + "\n"
                f.write(output_string)

        filename = "Smart_AttractiveTASEP_slow_right_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
        file_path = os.path.join(folder_path, filename)                    
        with open(file_path, 'w') as f:
          for t in range(Nt):
                output_string = str(t) + "\t" + str(slow_sites[t]) + "\n"
                f.write(output_string)


        filename = "Smart_AttractiveTASEP_YcurrentII_fast_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
        file_path = os.path.join(folder_path, filename)                    
        with open(file_path, 'w') as f:
             for y in range(Ly):
                 output_string = str(YcurrentII_fast[y]) + "\n"
                 f.write(output_string) 

        filename = "Smart_AttractiveTASEP_YcurrentII_slow_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
        file_path = os.path.join(folder_path, filename)                    
        with open(file_path, 'w') as f:
             for y in range(Ly):
                 output_string = str(YcurrentII_slow[y]) + "\n"
                 f.write(output_string) 

        filename = "Smart_AttractiveTASEP_YcurrentT_fast_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
        file_path = os.path.join(folder_path, filename)                    
        with open(file_path, 'w') as f:
            for y in range(Ly):
                output_string =  str(YcurrentT_fast[y]) + "\n"
                f.write(output_string) 

        filename = "Smart_AttractiveTASEP_YcurrentT_slow_"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
        file_path = os.path.join(folder_path, filename)                    
        with open(file_path, 'w') as f:
            for y in range(Ly):
                output_string =  str(YcurrentT_slow[y]) + "\n"
                f.write(output_string) 

        filename = "Smart_AttractiveTASEP_fast_chosen"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
        file_path = os.path.join(folder_path, filename)                    
        with open(file_path, 'w') as f:
            for t in range(recording_time):
                output_string = str(t) + "\t" + str(fast_chosen[t]) + "\n"
                f.write(output_string)

        filename = "Smart_AttractiveTASEP_slow_chosen"  + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
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