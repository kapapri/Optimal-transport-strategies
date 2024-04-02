import numpy as np
import matplotlib.pyplot as plt

runs = 10
newLx = 50
newLy = 20
Nt = 1000
current = np.loadtxt('results/1-0.5_particles/Posttraining/2d_TASEP_current_50x20_runs10.txt')  # This assumes data is separated by whitespace
random_current = np.loadtxt('results/1-0.5_particles/Posttraining/Random2d_TASEP_current_50x20_runs10.txt')  # This assumes data is separated by whitespace
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
plt.ylim([0, 0.5])    

# # Show the plot
# plt.show()

plt.savefig(f"./Comparation_Currents{newLx}x{newLy}.png", format="png", dpi=600)
