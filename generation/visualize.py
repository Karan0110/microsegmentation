from typing import Union
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

def visualize(points : np.ndarray,
              show_axes : bool = False,
              sample_size : Union[int, None] = 2 * 10**4,
              colors : Union[list, str, None] = None,
              background_color : Union[str, None] = None) -> None:
    num_points = points.shape[0]

    if sample_size is not None and sample_size <= num_points:
        indices = random.sample(range(num_points), sample_size)
        points = points[indices]


    # Separate the points into x, y, and z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if background_color is not None:
        fig.patch.set_facecolor(background_color) #type: ignore
        ax.set_facecolor(background_color)

    # Plot the points in black
    if colors is None:
        colors = 'black'
    ax.scatter(x, y, z, c=colors, s=0.1)

    # Set labels (optional)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Draw lines through the origin
    if show_axes:
        ax.plot(ax.get_xlim(), [0,0], [0,0], color='black', linewidth=3)
        ax.plot([0,0], ax.get_ylim(), [0,0], color='black', linewidth=3)
        ax.plot([0,0], [0,0], ax.get_zlim(), color='black', linewidth=3)

    ax.set_axis_off()
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    # Define the points
    points = np.random.normal(loc=0., scale=1., size=(1000,3))

    visualize(points)
