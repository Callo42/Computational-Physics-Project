import numpy as np
import matplotlib.pyplot as plt

def velocity_plot(positions,velocities):
    """
    plot the velocity vectors at each point on positions
    note here is 2D situation
    """

    position_x = positions[::2]
    position_y = positions[1::2]   
    velocity_x = velocities[::2]
    velocity_y = velocities[1::2]

    figure,ax = plt.subplots()
    ax.quiver(position_x,position_y,velocity_x,velocity_y)
    plt.show()