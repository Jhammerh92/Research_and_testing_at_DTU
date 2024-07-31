import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

def plot_plane_with_lighting(normal, point_on_plane, size=10, grid_resolution=100):
    

    leg = np.linspace(-size/2, size/2, grid_resolution)
    X, Y = np.meshgrid(leg, leg)
    # Z = np.random.uniform(-size*0.02,size*0.02, X.shape)
    Z = np.sin(X) * np.cos(Y)


    # Create a LightSource object
    ls = LightSource(azdeg=0, altdeg=60)
    
    # Apply the lighting effect to a constant color
    color = Z
    rgb = ls.shade(color, cmap=plt.cm.jet, vert_exag=0.5, blend_mode='soft')
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with lighting
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, antialiased=True)
    
    def set_equal_aspect(ax):
        x = np.asarray(ax.get_xlim())
        y = np.asarray(ax.get_ylim())
        z = np.asarray(ax.get_zlim())

        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        return ax

    # Set plot limits
    set_equal_aspect(ax)
    
    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.show()

# Example usage
normal = np.random.uniform(-1,1, 3)  # Normal vector of the plane (a, b, c)
normal = normal / np.linalg.norm(normal)
point_on_plane = (0, 0, 0)  # A point on the plane (px, py, pz)
plot_plane_with_lighting(normal, point_on_plane)