import numpy as np
import matplotlib.pyplot as plt

runs = 1
newLx = 12
newLy = 12
Nt = 1000
current = np.loadtxt('/home/a/A.Rivera/Code_Projects/1_Thesis_Code/Current_TASEP/results_new_input_NN 0.8-1/3. reward boundary lane/12x12/2d_TASEP_current_12x12_runs1.txt') 
random_current = np.loadtxt('/home/a/A.Rivera/Code_Projects/1_Thesis_Code/Current_TASEP/results_new_input_NN 0.8-1/3. reward boundary lane/12x12/Random2d_TASEP_current_12x12_runs1.txt')  
episode_duration = np.arange(Nt)
plt.figure(figsize=(10, 6))  # Set the figure size (optional)

# If your data has multiple columns and you want to plot each column against the first column (for example):
for i in range(1, random_current.shape[1]):
    plt.plot(episode_duration, current[:, i], label=f'Learning')
    plt.plot(episode_duration, random_current[:, i], label=f'Random')

# Add labels and title
plt.xlabel('Episode duration')
plt.ylabel(f'Average current over {runs} runs')
plt.title(f'Current comparation for {newLx}x{newLy} system')
plt.legend() 
plt.ylim([0, 0.7])    

# # Show the plot
# plt.show()

plt.savefig(f"./Comparation_Currents{newLx}x{newLy}.png", format="png", dpi=600)
