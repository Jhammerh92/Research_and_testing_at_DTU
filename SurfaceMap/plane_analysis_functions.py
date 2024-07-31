import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LightSource


def median_absolute_deviation(x, k= 1.4826):
    median = np.median(x)
    mad = k * np.median(np.abs(x - median))
    return mad

def median_position(x):
    return np.median(x, axis=0)

def median_normal_vector(n):
    median_normal = np.median(n, axis=0)
    norm = np.linalg.norm(median_normal)
    median_normal /= norm
    return median_normal

def plane_distance(query_points, surface_point, surface_normal):
    return  (query_points - surface_point) @ surface_normal.T 
    # return np.abs( (x - c) @ n.T )

def normal_distances(point, normals, normal_points):
    a = point @ normals.T
    b = np.einsum("ij,ij -> i", normal_points, normals)
    dists = a - b
    return dists


def radial_distance(x, c, n):
    normal_distance = (x - c) @ n.T
    projected_point = (np.atleast_2d(c).T + normal_distance * np.atleast_2d(n).T).T
    # Compute the distance between the projected point and the given point
    radial_distances = np.linalg.norm(x - projected_point, axis=1)
    return radial_distances

def vector_angular_distance(reference_vector, vectors):
    """
    Compute the angular distance of unit vectors from a single unit vector.

    Parameters:
    - reference_vector: The reference unit vector as a numpy array of shape (3,).
    - vectors: A list of unit vectors as numpy arrays of shape (3, N).

    Returns:
    - angular_distances: A numpy array of angular distances corresponding to each vector.
    """
    # Normalize reference vector
    # reference_vector /= np.linalg.norm(reference_vector)

    # Compute the dot product of each vector with the reference vector
    dot_products =  np.clip(vectors @ reference_vector.T, -1, 1)

    # Compute the angular distance for each vector
    angular_distances = np.arccos(dot_products)

    return angular_distances

def vector_angular_distance_and_direction(reference_vector, vectors):
    """
    Compute the angular distance and azimuthal direction of unit vectors from a single unit vector.

    Parameters:
    - reference_vector: The reference unit vector as a numpy array of shape (3,).
    - vectors: A list of unit vectors as a numpy array of shape (N, 3).

    Returns:
    - angular_distances: A numpy array of angular distances corresponding to each vector.
    - azimuthal_angles: A numpy array of azimuthal angles corresponding to each vector.
    """
    # Ensure the reference vector is normalized
    reference_vector /= np.linalg.norm(reference_vector)

    # Compute the dot product of each vector with the reference vector
    dot_products = np.clip(vectors @ reference_vector.T, -1, 1)

    # Compute the angular distance for each vector
    angular_distances = np.arccos(dot_products)

    # Find an orthonormal basis for the plane perpendicular to the reference vector
    # Create a vector that is not parallel to the reference vector
    if np.abs(reference_vector[0]) < np.abs(reference_vector[1]):
        perp_vector = np.array([1, 0, 0])
    else:
        perp_vector = np.array([0, 1, 0])
    
    # # Create two orthogonal vectors in the plane orthogonal to the reference vector
    # basis_vector1 = np.cross(reference_vector, perp_vector)
    # basis_vector1 /= np.linalg.norm(basis_vector1)
    # basis_vector2 = np.cross(reference_vector, basis_vector1)
    # basis_vector2 /= np.linalg.norm(basis_vector2)

    # # Project vectors onto the plane orthogonal to the reference vector
    # projected_vectors = vectors - np.outer(dot_products, reference_vector)

    basis = construct_orthonormal_basis(reference_vector)

    coords = vectors @ basis.T


    # Compute the coordinates in the new basis
    x_coords = coords[:,0]
    y_coords = coords[:,1]

    # Compute the azimuthal angles using arctan2 for better numerical stability
    azimuthal_angles = np.arctan2(x_coords, - y_coords)

    # Normalize azimuthal angles to be between 0 and 2*pi
    azimuthal_angles = np.mod(azimuthal_angles, 2 * np.pi)

    return angular_distances, azimuthal_angles

def is_plane(_points, _normals, plot=False): 
    points = np.array(_points)
    normals = np.array(_normals)
    count = points.shape[0]
    
    median_point = median_position(points)
    median_normal = median_normal_vector(normals)

    plane_distances = plane_distance(points, median_point, median_normal)
    angular_distances = vector_angular_distance(median_normal, normals)

    median_absolute_deviation_plane = median_absolute_deviation(np.abs(plane_distances))
    median_absolute_deviation_angle = median_absolute_deviation(angular_distances)

    is_plane_val = np.sqrt(median_absolute_deviation_angle**2 + median_absolute_deviation_plane**2)
    is_plane_bool = is_plane_val < 0.2 and count >= 3
    # is_plane_bool = is_plane_val < 0.2

    curvature = compute_curvature_from_points(points)

    origodistance = plane_distance(median_point, np.zeros(3), -median_normal)

    # print('is plane: ', is_plane_bool, 'mad: ', median_absolute_deviation_plane, 'mad angle: ', np.rad2deg(median_absolute_deviation_angle), 'count: ', count)


    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*points.T)
        # ax.scatter(*inlier_points.T, alpha=0.7, c='g', s=5, edgecolors=None)
        ax.quiver(*median_point, *median_normal)
        ax.scatter(*median_point, c='b', s=200)
        ax.quiver(*points.T, *normals.T, length=0.1, color='k')
        set_equal_aspect(ax)
        # plt.show()

    return is_plane_bool, median_point, median_normal, median_absolute_deviation_plane, median_absolute_deviation_angle, origodistance, curvature

def plot_plane(ax, params, size=1.0, color='g'):
    """
    Plot a plane in 3D given a center position and a normal vector.

    Parameters:
    - ax: The matplotlib 3D axis to plot the plane on.
    - params: dict containing 'center' (array-like, shape (3,)) and 'normal' (array-like, shape (3,)).
    - size: float, optional, default is 1.0, the size of the plane to be plotted.

    Returns:
    - ax: The matplotlib 3D axis with the plane plotted.
    """
    # Extract center and normal from the params dictionary
    center = np.array(params['center'])
    normal = np.array(params['normal'])
    mad = np.array(params['mad']).item()
    size = np.array(params['size']).item()
    origodist = np.array(params['origodist']).item()
    R_basis = np.array(params['basis'])
    corners = np.array(params['corners'])

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Generate a grid of points in the XY plane
    point_grid = np.linspace(-size, size, 2)
    X, Y = np.meshgrid(point_grid, point_grid)
    Z = np.zeros_like(X) 
    # X += center[0]
    # Y += center[1]
    # Z += origodist

    # Create the grid of points
    # grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    grid_points = corners

    # Calculate the rotation matrix to align Z-axis with the normal
    def rotation_matrix_from_vectors(vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2 """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    # R = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal)

    # local_normal_basis = construct_orthonormal_basis(normal @ R_basis[:3,:3])

    # elevation_from_normal(normal @ R_basis[:3,:3], grid_points)

    # Rotate the grid points to align with the normal vector
    # rotated_grid_points = grid_points @ R.T
    rotated_grid_points =  homogenous_transformation(grid_points, R_basis, pre=True)

    # Offset the grid to the center position
    X_rot = rotated_grid_points[:, 0]
    Y_rot = rotated_grid_points[:, 1]
    Z_rot = rotated_grid_points[:, 2]

    # Reshape the points back to the grid shape
    X_rot = X_rot.reshape(X.shape)
    Y_rot = Y_rot.reshape(Y.shape)
    Z_rot = Z_rot.reshape(Z.shape)


    # ls = LightSource(270, 45)
    # color = rotated_grid_points[:,2]
    # fc = ls.shade(color, cmap=plt.cm.gray, vert_exag=0.1, blend_mode='soft')

    # Plot the plane
    # fc = ls.shade_normals(normal)
    # fc = ls.shade(mad, cmap='jet')
    # ax.plot_surface(X_rot, Y_rot, Z_rot, alpha=0.5, color=color, lightsource=ls, linewidth=1, edgecolor='k')
    ax.plot_surface(X_rot, Y_rot, Z_rot, alpha=0.5, color=color, linewidth=0.2, edgecolor='k')
    # ax.plot_surface(X_rot, Y_rot, Z_rot, alpha=1.0, facecolor=fc, linewidth=1)
    # ax.quiver(*center, *normal, length=1.0, color='k')

    # Plot the center point
    # ax.scatter(center[0], center[1], center[2], color='red')

    # # Set labels for clarity
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')

    # # Set equal aspect ratio
    # ax.set_box_aspect([1, 1, 1])

    return ax

def construct_orthonormal_basis(normal):
    """
    Construct an orthonormal basis from a normal vector such that the normal vector is the z-axis.

    Parameters:
    - normal: numpy array of shape (3,) representing the normal vector.

    Returns:
    - basis: numpy array of shape (3, 3) representing the orthonormal basis.
    """
    # Normalize the normal vector to get the new z-axis
    z_axis = normal / np.linalg.norm(normal)

    # Create an arbitrary vector in the xy plane
    if np.abs(z_axis[0]) < 1e-10 and np.abs(z_axis[1]) < 1e-10:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = np.array([-z_axis[1], z_axis[0], 0.0])
    
    # Normalize the new x-axis
    x_axis /= np.linalg.norm(x_axis)

    # Compute the new y-axis using the cross product
    y_axis = np.cross(z_axis, x_axis)
    
    # Normalize the new y-axis
    y_axis /= np.linalg.norm(y_axis)

    # Ensure orthonormality
    basis = np.array([x_axis, y_axis, z_axis])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.quiver(*np.zeros_like(basis), *basis.T)
    # plt.show()

    return basis

def elevation_from_normal(normal, points, normal_point=None):
    a, b, c = normal
    # x0, y0, z0 = np.mean(points, axis=0)
    if normal_point is None:
        normal_point = np.mean(points, axis=0)
    px,py,pz = normal_point
    # d = a * x0 + b * y0 + c * z0
    d = a * px + b * py + c * pz
    x, y, _z = points.T
    z = (d - a * x - b * y) / c
    points[:,2] = z

def homogenous_transformation(points, transformation, pre=False):
    points = np.atleast_2d(points)
    if points.shape[1] != 3:
        points = points.T
    points = np.c_[points, np.ones((points.shape[0],1))]

    if pre:
        transformed_points = points @ transformation.T
    else:
        transformed_points = points @ transformation

    return transformed_points[:,:3]

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

def normals_to_rpy(normals):
    """
    Calculate roll, pitch, and yaw (ZYX order) from an array of normal vectors.

    Parameters:
    - normals (array-like): Array of normal vectors with shape (n, 3)

    Returns:
    - rpy (ndarray): Array of roll, pitch, and yaw angles with shape (n, 3)
    """
    normals = np.array(normals)
    
    # Ensure the normals array is 2D
    if normals.ndim == 1:
        normals = normals[np.newaxis, :]

    # Normalize the normal vectors
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms

    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]

    # Calculate yaw (ψ)
    # yaw = np.arctan2(ny, nx)
    yaw = np.arctan2(nx, - ny)
    # yaw = np.arctan(nx/ - ny)

    # Calculate pitch (θ)
    # pitch = np.arctan2(-nx * np.sin(yaw) + ny * np.cos(yaw), nz)
    pitch = np.arctan2(np.sqrt(nx**2 + ny**2), nz)

    # Calculate roll (φ)
    # roll = np.arctan2(nx * np.cos(yaw) + ny * np.sin(yaw), nz)
    roll = np.zeros_like(nx)

    return np.vstack((roll, pitch, yaw)).T

def compute_curvature_from_points(points):
    """
    Compute the curvature of a plane defined by a set of points.

    Parameters:
    - points (ndarray): An array of shape (n, 3), where each row represents a point (x, y, z).

    Returns:
    - k_x (float): Curvature along x.
    - k_y (float): Curvature along y.
    - k (float): Total curvature.
    """

    if points.shape[0] <= 2:
        return 0.0,0.0,0.0

    # Calculate the mean of the points
    mean = np.mean(points, axis=0)
    
    # Center the points
    centered_points = points - mean
    
    # Calculate the covariance matrix
    cov_matrix = np.cov(centered_points, rowvar=False)
    
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort the eigenvalues and eigenvectors by eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Principal curvatures are inversely proportional to the eigenvalues
    k_x = 1 / np.sqrt(eigenvalues[0])
    k_y = 1 / np.sqrt(eigenvalues[1])
    
    # Total curvature (mean curvature)
    k = (k_x + k_y) / 2

    return k, k_x, k_y


def points_within_distance(tree, points, center_point, distance):
    """
    Find all points within a specified distance from a given center point using a pre-built KD-tree.

    Parameters:
    - tree (KDTree): Pre-built KD-tree of the points.
    - points (array-like): Array of points in the form [[x1, y1], [x2, y2], ...] or [[x1, y1, z1], [x2, y2, z2], ...].
    - center_point (array-like): The center point from which distance is measured.
    - distance (float): The distance threshold.

    Returns:
    - within_distance (array): Array of points within the specified distance from the center point.
    """
    points = np.array(points)
    center_point = np.array(center_point)

    # Query the tree for points within the specified distance from the center point
    indices = tree.query_ball_point(center_point, distance)

    # Select the points within the specified distance
    # within_distance = points[indices]

    return indices
    # return within_distance