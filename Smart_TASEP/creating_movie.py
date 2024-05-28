

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from IPython.display import HTML
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\FFmpeg\\bin\\ffmpeg.exe'


def create_animation(Frames_movie):
    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    # cax = div.append_axes('right', '5%', '5%')
    
    cv0 = Frames_movie[0]
    cmap=colors.ListedColormap(['black', 'purple', 'yellow'])
    bounds = [0,0.25,0.85,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(cv0, cmap=cmap, norm=norm)

    # im = ax.imshow(cv0, cmap="gnuplot")
    # cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title('Frame 0', y=1)
        
    ax.axis('off')
    plt.close()  # To not have the plot of frame 0

    def animate(frame):
        arr = Frames_movie[frame]
        vmax = 1
        vmin = np.min(arr)
        im.set_data(arr)
        im.set_clim(vmin, vmax)
        # cb.ax.set_ylabel('Jumping Rate')
        tx.set_text('Frame {0}'.format(frame))

    ani = FuncAnimation(fig, animate, frames=len(Frames_movie), repeat=False)
    return ani

    
if __name__ == '__main__':

    newLx = 12
    newLy = 12
    Nt = 1000

    with open(f"./movie_storage.pkl", 'rb') as file:
        movie_storage = pickle.load(file)
    print('Action! (recording movie)')
    ani = create_animation(movie_storage[:Nt]) #last run
    HTML(ani.to_jshtml()) # interactive python
    ani.save("./Movie"+".gif", fps = 8)
    print('Cut! (movie ready)')    
