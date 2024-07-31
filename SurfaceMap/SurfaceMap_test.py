import numpy as np
from SurfaceMap import SurfaceMap
import matplotlib.pyplot as plt
from plane_analysis_functions import *

from normal_gradients_testing import make_error_gradient_space,extract_isosurface, plot_isosurface

def random_transformation_matrix(translation_limits, rotation_limits):
    """
    Generate a random transformation matrix within given translation and rotation limits.
    
    Parameters:
    - translation_limits: A dictionary with 'x', 'y', 'z' keys and (min, max) tuples as values.
                          Example: {'x': (-1, 1), 'y': (-1, 1), 'z': (-1, 1)}
    - rotation_limits: A dictionary with 'roll', 'pitch', 'yaw' keys and (min, max) tuples in radians as values.
                       Example: {'roll': (-np.pi, np.pi), 'pitch': (-np.pi/2, np.pi/2), 'yaw': (-np.pi, np.pi)}
                       
    Returns:
    - transformation_matrix: A 4x4 numpy array representing the transformation matrix.
    """
    
    # Generate random translation values within limits
    tx = np.random.uniform(*translation_limits['x'])
    ty = np.random.uniform(*translation_limits['y'])
    tz = np.random.uniform(*translation_limits['z'])
    
    # Generate random rotation angles within limits
    roll = np.random.uniform(*rotation_limits['roll'])
    pitch = np.random.uniform(*rotation_limits['pitch'])
    yaw = np.random.uniform(*rotation_limits['yaw'])
    
    # Create rotation matrix using ZYX order
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R = Rz @ Ry @ Rx
    
    # Create transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = [tx, ty, tz]
    
    return transformation_matrix



if __name__ == "__main__":

    # data_name = '../000200'
    # test_data_name = '../000201'
    # data_name = '../HAP_sweep_ds'
    data_name = 'sphere_cube_random'
    test_data_name = 'sphere_cube_random'
    # data_name = '../bavaria_plane_test_section'
    points = np.load(data_name + ".npy")
    normals = np.load(data_name + "_normals.npy")

    # Example usage
    translation_limits = {'x': (-0, 0), 'y': (-0.1, 0.1), 'z': (-0, 0)}
    # rotation_limits = {'roll': (-np.pi/10, np.pi/10), 'pitch': (-np.pi/2/10, np.pi/2/10), 'yaw': (-np.pi/10, np.pi/10)}
    rotation_limits = {'roll': (0,0), 'pitch': (0,0), 'yaw': (0,0)}

    T_test = random_transformation_matrix(translation_limits, rotation_limits)
    print(T_test)

    test_points = np.load(test_data_name + ".npy")
    test_normals = np.load(test_data_name + "_normals.npy")

    test_points = homogenous_transformation(test_points, T_test, pre=True)



    # _,sample_indices = farthest_point_sampling(points, 100)

    test_points = test_points[1::50,:]# + np.array([0.0,0.5,0.0])

    every_n = 10
    points = points[::every_n,:]
    normals = normals[::every_n,:]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points.T, c='g', s=1)
    ax.scatter(*test_points.T, c='r', s=1)
    set_equal_aspect(ax)
    plt.show()

    map = SurfaceMap()

    map.create_initial_map(points, normals)

    map.visualize_map()

    # for tp in test_points:
    #     map.query_point(tp)

    error_vectors, error_points = map.query_points(test_points)

    X,Y, Z, vals = make_error_gradient_space(error_vectors)
    
    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X[:,:,0],Y[:,:,0],vals[:,:,0], alpha=0.5)
    ax.scatter(0,0,0, s=20, c='r')
    # ax.scatter(*test_delta.T, test_grads, s=20, c='b')
    # ax.scatter(*normal_gradient_contour.T,np.ones(normal_gradient_contour.shape[0]), s=20, c='g')
    plt.show()

    isosurface = extract_isosurface(vals, 20)
    plot_isosurface(*isosurface)







    median_error = np.median(error_vectors, axis=0)
    mean_error = np.mean(error_vectors, axis=0)

    print(median_error, mean_error)
    

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points.T, c='g', s=2)
    ax.scatter(*error_points.T, c='r', s=10)
    ax.quiver(*error_points.T, *error_vectors.T)
    # ax.quiver(*[0.0,0.0,0.0], *median_error.T, length=2.0)

    set_equal_aspect(ax)
    plt.show()

