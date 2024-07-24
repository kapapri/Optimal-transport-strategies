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
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML
from tqdm import tqdm
import pickle

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    distance_center = abs(Ycenter - int(Ly / 2))/int(Ly / 2) # normalized
    state = np.append(patch, distance_center)

    return state

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


def is_patch_crossing_boundary(Y_boundary, Ycenter, L, Ly):
    half_L = int(L / 2)
    distance = abs(Y_boundary - Ycenter)

    if min(distance, Ly - distance) <= half_L:
        return True
    else:
        return False

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
def step(lattice, X, Y, L, Xcenter, Ycenter, boundary_lane, log = False):
    newX = -1
    newY = -1
    # periodic boundaries
    Lx, Ly = lattice.shape
    nextX = X + 1 if X < Lx - 1 else 0
    prevX = X - 1 if X > 0 else Lx - 1
    nextY = Y + 1 if Y < Ly - 1 else 0
    prevY = Y - 1 if Y > 0 else Ly - 1

    # update position
    direction = random.randint(0,3)
    if direction == 0 or direction == 1: # jump right
        newX = nextX
        newY = Y
    elif direction == 2: # jump up
        newY = nextY
        newX = X
    else: # jump down
        newY = prevY
        newX = X

    current_along = 0
    reward = 1 # simply for choosing a particle and not an empty space
    speed = lattice[X][Y]
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

 # surroundings reward (before jump)
    if lattice[nextX][Y] == 0:
        forward_particle = 0
    else:
        forward_particle = 1

    reward += int(- 1*(2*forward_particle - 1))   

 # lanes rewards (before jump)
    # extracts a vector with the columns of the patch
    half_patch = int(L/2)
    start_idx = (Xcenter - half_patch) % Lx
    end_idx = (Xcenter + half_patch + 1) % Lx
    if start_idx < end_idx:
        patch_columns = lattice[start_idx:end_idx, :]
    else:
        patch_columns = np.concatenate((lattice[start_idx:Lx, :], lattice[0:end_idx, :]), axis=0)

    for boundary in [boundary_lane, (boundary_lane-1), 0, (Ly-1)]:
        if is_patch_crossing_boundary(boundary, Ycenter, L, Ly):
            particles_boundary = patch_columns[:, boundary]
            if speed == 1:
                if boundary == boundary_lane or boundary == (Ly-1):
                    if Y == boundary:
                        reward += 5
                    elif Y > boundary_lane and np.any(particles_boundary == speed): # slow region
                        reward += -1
                    elif Y < boundary_lane and np.any(particles_boundary == speed): # fast region
                        reward += -5

                elif boundary == 0 or boundary == (boundary_lane-1):
                    if Y == boundary:
                        reward += 0                                   
                    elif Y >= boundary_lane and np.any(particles_boundary == speed): # slow region
                        reward += -1
                    elif Y < boundary_lane and np.any(particles_boundary == speed): # fast region
                        reward += -5

            elif speed == 0.8:
                if boundary == (boundary_lane-1) or boundary == 0:
                    if Y == boundary:
                        reward += 5                                
                    elif Y < boundary_lane and np.any(particles_boundary == speed): # fast region
                        reward += -1
                    elif Y >= boundary_lane and np.any(particles_boundary == speed): # slow region
                        reward += -5

                elif boundary == boundary_lane or boundary == (Ly-1):
                    if Y == boundary:
                        reward += 0                                    
                    elif Y > boundary_lane and np.any(particles_boundary == speed): # slow region
                        reward += -1
                    elif Y < boundary_lane and np.any(particles_boundary == speed): # fast region
                        reward += -5                      

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

def do_training(num_episodes, L, density, Nt, Lx, Ly, boundary_lane, log = False):
    for i_episode in tqdm(range(num_episodes)):
        # start with random initial conditions
        N = int(Lx*Ly*density) 
        lattice = np.zeros(shape=(Lx,Ly))
        n = 0
        while n < N:
            X = random.randint(0, Lx-1)
            Y = random.randint(0, Ly-1)
            if lattice[X][Y] == 0:
                lattice[X][Y] = Random.choice([0.8, 1], p =[0,1])
                n += 1

        # main update loop; I use Monte Carlo random sequential updates here
        score = 0
        total_current = 0
        selected_empty_site = 0
        selected_fast = 0
        selected_slow = 0
        right_fast, right_slow = 0, 0         

        for t in range(Nt):
            for i in range(Lx*Ly):
                total_fast, fast_up  = 0, 0
                total_slow, slow_down = 0, 0
                before_fast, after_fast = 0, 0
                before_slow, after_slow = 0, 0
                # random sampling in the lattice to apply the training
                Xcenter = random.randint(0, Lx-1)
                Ycenter = random.randint(0, Ly-1)

                # counting particles in the patch before jumping
                for i in range(L):
                    for j in range(L):
                        X_sys, Y_sys = get_coordinates_from_patch(i, j, Xcenter, Ycenter, L, Lx, Ly)
                        if lattice[X_sys][Y_sys] == 1 and Y_sys < boundary_lane:
                            before_fast += 1
                        elif lattice[X_sys][Y_sys] != 0 and Y_sys >= boundary_lane:
                            before_slow += 1                

                state = get_state_system(lattice, Lx, Ly, Xcenter, Ycenter, L)
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                action = select_action_training(state) # get the index of the particle
                lattice_site = action.item() # a number, and we encode it as x*L + y
                patchX = int(lattice_site / L)
                patchY = int(lattice_site % L)
                selectedX, selectedY = get_coordinates_from_patch(patchX, patchY, Xcenter, Ycenter, L, Lx, Ly)
                                                                                                                              
                if lattice[selectedX][selectedY] != 0:
                    # counting of selected fast and slow particles
                    if lattice[selectedX][selectedY] == 1:
                        selected_fast += 1
                    else:
                        selected_slow += 1                    
                    # update particle's position and do stochastic part                                         
                    reward, next_state, current_along = step(lattice, selectedX, selectedY, L, Xcenter, Ycenter, boundary_lane, log) 
                    total_current += current_along / (Lx*Ly*Nt)
                    reward = torch.tensor([reward], device=device)
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0) 
                    memory.push(state, action, next_state, reward)
                    
                    
                else: # empty site chosen
                    reward = -10
                    selected_empty_site += 1
                    reward = torch.tensor([reward], device=device)  
                    memory.push(state, action, state, reward)

                # counting particles in the patch after jumping
                for i in range(L):
                    for j in range(L):
                        X_sys, Y_sys = get_coordinates_from_patch(i, j, Xcenter, Ycenter, L, Lx, Ly)
                        if lattice[X_sys][Y_sys] == 1 and Y_sys < boundary_lane:
                            after_fast += 1
                        elif lattice[X_sys][Y_sys] != 0 and Y_sys >= boundary_lane:
                            after_slow += 1   

                score += reward

                # counting particles in their respective areas
                for i in range(Lx):
                    for j in range(Ly):
                        if lattice[i][j] == 1:
                            total_fast += 1
                            if j < boundary_lane:
                                fast_up += 1
                        elif lattice[i][j] != 0: 
                            total_slow += 1
                            if j >= boundary_lane:
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

        print("Training episode ", i_episode, " is over. Current = ", total_current, "; Selected empty sites / L*L = ", selected_empty_site / (Lx*Ly*Nt))             
        print("Fast particles chosen ", selected_fast/ (Lx*Ly*Nt), ". Slow particles chosen = ", selected_slow / (Lx*Ly*Nt))
        print("Fast particles in the upper side ", right_fast/ (Lx*Ly*Nt), ". Slow particles in the lower side = ", right_slow / (Lx*Ly*Nt))

        rewards.append(score.numpy()) 
        current.append(total_current)

        empty_sites.append(selected_empty_site / (Lx*Ly*Nt))
        fast_chosen.append(selected_fast / (Lx*Ly*Nt))
        slow_chosen.append(selected_slow / (Lx*Ly*Nt))
        fast_sites.append(right_fast / (Lx*Ly*Nt))
        slow_sites.append(right_slow / (Lx*Ly*Nt))

        plot_score() # here if you want to see the training
                     # only with interactive python

    torch.save(target_net.state_dict(), PATH)
    plot_score(show_result=True) # here to see the result
    plt.savefig(f"./Training_Reward_{Lx}x{Ly}.png", format="png", dpi=600)
    plot_current(current)
    plt.savefig(f"./Training_Current_{Lx}x{Ly}.png", format="png", dpi=600)
    plot_empty_sites(empty_sites)
    plt.savefig(f"./Training_Empty_Sites{Lx}x{Ly}.png", format="png", dpi=600)
    plot_type_particles(fast_chosen, slow_chosen)
    plt.savefig(f"./Training_Types_Particles{Lx}x{Ly}.png", format="png", dpi=600)
    plot_particle_occupation(fast_sites, slow_sites)
    plt.savefig(f"./Training_Particle_Occupation{Lx}x{Ly}.png", format="png", dpi=600) 

# plots
def plot_score(show_result=False):
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf() # clf -- clear current figure
        plt.title('Training...')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.plot(rewards)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            output = "./training_score.png"
            plt.savefig(output, format = "png", dpi = 300)

def plot_current(current, post = False):
    plt.clf() 
    plt.figure(2)
    if post == True:
         plt.xlabel('Episode duration')
         plt.ylabel('Average current over runs')
    else:
        plt.xlabel('Episodes')
        plt.ylabel('Current')
    plt.ylim([0, 0.5])    
    plt.plot(current)

def plot_YcurrentII(YcurrentII_fast, YcurrentII_slow):
    plt.clf() 
    plt.figure(3)
    plt.xlabel('Y index')
    plt.ylabel('Average parallel current over runs')
    plt.axvspan(0, (boundary_lane - 0.5), facecolor='lightblue', alpha=0.5, label='Fast region')
    plt.axvspan((boundary_lane - 0.5), (Ly-1), facecolor='lightgreen', alpha=0.5, label='Slow region')      
    plt.plot(YcurrentII_fast,'-o', label = 'fast particles')
    plt.plot(YcurrentII_slow,'-o', label = 'slow particles')
    plt.legend()

def plot_YcurrentT(YcurrentT_fast, YcurrentT_slow):
    plt.clf() 
    plt.figure(4)
    plt.xlabel('Y index')
    plt.ylabel('Average perpendicular current over runs') 
    plt.axvspan(0, (boundary_lane - 0.5), facecolor='lightblue', alpha=0.5, label='Fast region')
    plt.axvspan((boundary_lane - 0.5), (Ly-1), facecolor='lightgreen', alpha=0.5, label='Slow region')      
    plt.axhline(y=0, color='black', linestyle='--')
    plt.plot(YcurrentT_fast, '-o' ,label = 'fast particles')
    plt.plot(YcurrentT_slow, '-o', label = 'slow particles')
    plt.legend()

def plot_empty_sites(empty_sites, post = False):
    plt.clf() 
    plt.figure(5)
    if post == True:
         plt.xlabel('Episode duration')
         plt.ylabel(f'Average selected empty sites over {Lx} x {Ly} lattice')
    else:
        plt.xlabel('Episodes')
        plt.ylabel(f'Number of empty sites chosen over {Lx} x {Ly} lattice')    
    plt.ylim([0, 0.5])    
    plt.plot(empty_sites)

def plot_type_particles(fast_chosen, slow_chosen, post = False):
    plt.clf() 
    plt.figure(6)
    if post == True:
         plt.xlabel('Episode duration')
         plt.ylabel(f'Average selected particles over {Lx} x {Ly} lattice')
    else:
        plt.xlabel('Episodes')
        plt.ylabel(f'Selected particles over {Lx} x {Ly} lattice')       

    plt.ylim([0,1])    
    plt.plot(fast_chosen, label = 'fast particles')
    plt.plot(slow_chosen, label = 'slow particles')
    plt.legend()

def plot_particle_occupation(fast_sites, slow_sites, post = False):
    plt.clf() 
    plt.figure(7)
    if post == True:
         plt.xlabel('Episode duration')
         plt.ylabel(f'Average particles in their regions over {Lx} x {Ly} lattice')
    else:
        plt.xlabel('Episodes')
        plt.ylabel(f'Particles in their regions over {Lx} x {Ly} lattice')     

    plt.ylim([0,1])
    plt.plot(fast_sites, label = 'fast particles')
    plt.plot(slow_sites, label = 'slow particles')
    plt.legend(loc='lower right')
    

# Main
if __name__ == '__main__': 
    Jessie_we_need_to_train_NN = True
    Post_training = True
    log = False
    log_post = False
    ############# Model parameters for Machine Learning #############
    num_episodes = 100       # number of training episodes
    BATCH_SIZE = 100        # the number of transitions sampled from the replay buffer
    GAMMA = 0.99            # the discounting factor
    EPS_START = 0.9         # EPS_START is the starting value of epsilon; determines how random our action choises are at the beginning
    EPS_END = 0.001         # EPS_END is the final value of epsilon
    EPS_DECAY = 200         # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = 0.005             # TAU is the update rate of the target network
    LR = 1e-3               # LR is the learning rate of the AdamW optimizer
    ############# Lattice simulation parameters #############
    L = 5                   # squared patches for the training
    density = 0.5           # work with half-density
    Lx = 10
    Ly = 10
    N = int(Lx*Ly*density)
    boundary_lane = int(Ly/2)  # the regions are: fast [0, (boundary_lane-1)] and slow [boundary lane, (Ly-1)]
    Nt = 100                   # episode duration
    n_observations = 2*L*L + 1 # three channels. Two of the patch size: fast and slow particles and one with the distance to the center
    n_actions = L*L            # patch size, in principle, the empty spots can also be selected
    hidden_size = 128          # hidden size of the network
    PATH = f"./2d_TASEP_NN_params_{Lx}x{Ly}.txt"

    ############# Do the training if needed ##############
    if Jessie_we_need_to_train_NN:
        policy_net = DQN(n_observations, hidden_size, n_actions).to(device)
        target_net = DQN(n_observations, hidden_size, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(100*Nt) # the overall memory batch size 
        rewards = []
        current = []
        empty_sites = []
        fast_chosen = []
        slow_chosen = []
        fast_sites = []
        slow_sites = []
        steps_done = 0

        do_training(num_episodes, L, density, Nt, Lx, Ly, boundary_lane, log) 

    ############# Post-training simulation ##############
    if Post_training:
        runs = 10
        Lx = 10
        Ly = 10
        Nt = 1000
        trained_net = DQN(n_observations, hidden_size, n_actions).to(device)
        trained_net.load_state_dict(torch.load(PATH))

        current = np.zeros(Nt)
        empty_sites = np.zeros(Nt)
        fast_chosen = np.zeros(Nt)    
        slow_chosen = np.zeros(Nt)
        fast_sites = np.zeros(Nt)      # fast particles in the right region
        slow_sites = np.zeros(Nt)
        YcurrentII_fast = np.zeros(Ly) # parallel current for fast particles
        YcurrentII_slow = np.zeros(Ly) 
        YcurrentT_fast = np.zeros(Ly)  # perpendicular current for fast particles
        YcurrentT_slow = np.zeros(Ly)

        for run in tqdm(range(runs)):
            # start with random initial conditions
            N = int(Lx*Ly*density) 
            lattice = np.zeros(shape=(Lx,Ly))
            n = 0
            while n < N:
                X = random.randint(0, Lx-1)
                Y = random.randint(0, Ly-1)
                if lattice[X][Y] == 0:
                    lattice[X][Y] = Random.choice([0.8, 1], p =[0,1])
                    n += 1

            for t in tqdm(range(Nt)):
                total_current = 0
                selected_empty_site = 0
                selected_fast = 0
                selected_slow = 0                
                right_fast = 0
                right_slow = 0

                for move_attempt in range(Lx*Ly):
                    total_fast, fast_up  = 0, 0
                    total_slow, slow_down = 0, 0

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

                    # to check how does the simulation perform for the random "stupid" simulation, 
                    # uncomment two lines below and comment the two lines above
                    # selectedX = random.randint(0, Lx-1)
                    # selectedY = random.randint(0, Ly-1)

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
                        if speed == 1:
                            selected_fast += 1/(Lx*Ly*runs)
                        else:
                            selected_slow += 1/(Lx*Ly*runs)     

                        if jump_dice <= speed:
                            if lattice[newX][newY] == 0:
                                lattice[selectedX][selectedY] = 0
                                lattice[newX][newY] = speed
                                if newX != selectedX: # we have jump forward
                                    total_current += 1/(Lx*Ly*runs)
                                    if speed == 1:
                                        YcurrentII_fast[selectedY] += 1/(Lx*Ly*Nt*runs)
                                    else:
                                        YcurrentII_slow[selectedY] += 1/(Lx*Ly*Nt*runs)

                                elif newY == nextY:
                                    if log == True:
                                        print("  moved up")
                                    if speed == 1:
                                        YcurrentT_fast[selectedY] += -1/(Lx*Ly*Nt*runs)
                                    else:
                                        YcurrentT_slow[selectedY] += -1/(Lx*Ly*Nt*runs)

                                elif newY == prevY:
                                    if log == True:
                                        print("  moved down")
                                    if speed == 1:
                                        YcurrentT_fast[selectedY] += 1/(Lx*Ly*Nt*runs)
                                    else:
                                        YcurrentT_slow[selectedY] += 1/(Lx*Ly*Nt*runs)
                                                                  
                    else:
                        if log == True:
                            print("ALARM! ALARM!")                            
                            print("empty site chosen")
                        selected_empty_site += 1/(Lx*Ly*runs)

                    # counting particles in their respective areas    
                    for i in range(Lx):
                        for j in range(Ly):
                            if lattice[i][j] == 1:
                                total_fast += 1
                                if j < boundary_lane:
                                    fast_up += 1
                            elif lattice[i][j] != 0:
                                total_slow += 1
                                if j >= boundary_lane:
                                    slow_down += 1
                    right_fast += fast_up /(total_fast*Lx*Ly*runs) if total_fast != 0 else 0
                    right_slow += slow_down / (total_slow*Lx*Ly*runs) if total_slow != 0 else 0                        

                current[t] += total_current #sum of the currents of all runs
                empty_sites[t] += selected_empty_site
                fast_chosen[t] += selected_fast
                slow_chosen[t] += selected_slow              
                fast_sites[t] += right_fast
                slow_sites[t] += right_slow

        plot_current(current, post = True)
        plt.savefig(f"./Post_Current_{runs}_{Lx}x{Ly}.png", format="png", dpi=600)
        plot_empty_sites(empty_sites, post = True)
        plt.savefig(f"./Post_Empty_Sites_Chosen_{runs}_{Lx}x{Ly}.png", format="png", dpi=600)
        plot_type_particles(fast_chosen, slow_chosen, post = True)
        plt.savefig(f"./Post_Type_Particle_Chosen_{runs}_{Lx}x{Ly}.png", format="png", dpi=600)                    
        plot_particle_occupation(fast_sites, slow_sites, post = True)
        plt.savefig(f"./Post_Particle_Occupation{runs}_{Lx}x{Ly}.png", format="png", dpi=600)
        plot_YcurrentII(YcurrentII_fast, YcurrentII_slow)
        plt.savefig(f"./Post_Y_CurrentII_{runs}_{Lx}x{Ly}.png", format="png", dpi=600) 
        plot_YcurrentT(YcurrentT_fast, YcurrentT_slow)
        plt.savefig(f"./Post_Y_CurrentT_{runs}_{Lx}x{Ly}.png", format="png", dpi=600)  


        filename = "2d_TASEP_current_" + str(Lx) + "x" + str(Ly) + "_runs" + str(runs) + ".txt"
        with open(filename, 'w') as f:
            for t in range(Nt):
                output_string = str(t) + "\t" + str(current[t]) + "\n"
                f.write(output_string)
