import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from pyquaternion import Quaternion
from scipy.spatial import KDTree

from QuadTree import Quadtree
from SurfaceQuadTree import SurfaceQuadtree
from plane_analysis_functions import *



# def L1_clustering(data):

# def homogenous_transformation(points, transformation):
#     points = np.array(points)
#     points = np.c_[points, np.zeros((points.shape[0],1))]

#     transformed_points = points @ transformation
#     return transformed_points[:,:3]


def cluster_with_unknown_k(data):
    """
    Perform clustering with an unknown number of clusters using KMeans and the elbow method.

    Parameters:
    - data: Input data with shape (num_samples, num_features).

    Returns:
    - cluster_labels: Cluster labels assigned to each data point.
    """
    # Initialize an empty list to store inertia values
    inertia_values = []

    # Maximum number of clusters to consider
    max_clusters = min(len(data), 10)

    # Try different numbers of clusters and compute inertia for each
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k )
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)

    # Plot the elbow curve

    # Choose the optimal number of clusters based on the elbow point
    # In this simple example, we'll choose the elbow point manually
    # optimal_k = int(input("Enter the optimal number of clusters: "))
    optimal_k = 3


    plt.plot(range(1, max_clusters + 1), inertia_values, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')

    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k)
    cluster_labels = kmeans.fit_predict(data)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=False)

    for ci in np.unique(cluster_labels):
        c_data = data[cluster_labels==ci,:]
        plt.scatter(c_data[:,0], np.rad2deg(c_data[:,1]), alpha=0.05, s=5)

    # plt.title('Normal Spherical Coordinates ')
    # plt.xlabel('Polar Angle')
    # plt.ylabel('Azimuth Angle')
    plt.show()

    return cluster_labels

def cluster_with_unknown_k_gaussian(data):
    """
    Perform clustering with an unknown number of clusters using KMeans and the elbow method.

    Parameters:
    - data: Input data with shape (num_samples, num_features).

    Returns:
    - cluster_labels: Cluster labels assigned to each data point.
    """
    # Initialize an empty list to store inertia values
    inertia_values = []

    # Maximum number of clusters to consider
    max_clusters = min(len(data), 5)

    # Try different numbers of clusters and compute inertia for each
    # k_clusters = np.arange(2, max_clusters)
    k_clusters = [2]
    for k in k_clusters:
        gmm = GaussianMixture(n_components=k )
        gmm.fit(data)
        inertia_values.append(gmm.lower_bound_)

    # Plot the elbow curve

    # Choose the optimal number of clusters based on the elbow point
    # In this simple example, we'll choose the elbow point manually
    # optimal_k = int(input("Enter the optimal number of clusters: "))
    # optimal_k = 3
    optimal_k = k_clusters[np.argmin(inertia_values)]
        
    # fig = plt.figure(figsize=(8, 6))
    # plt.plot(k_clusters, inertia_values, marker='o')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Log Likelihood')
    # plt.title('Elbow Method for Optimal k')

    # Perform KMeans clustering with the optimal number of clusters
    gmm = GaussianMixture(n_components=optimal_k)
    cluster_labels = gmm.fit_predict(data)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=False)

    for ci in np.unique(cluster_labels):
        c_data = data[cluster_labels==ci,:]
        ax.scatter(c_data[:,0], c_data[:,1], alpha=0.05, s=5)

    # plt.title('Normal Spherical Coordinates ')
    # plt.xlabel('Polar Angle')
    # plt.ylabel('Azimuth Angle')
    plt.show()

    return cluster_labels

def normal_to_quaternion(normal_vectors):
    # Define the reference direction
    reference_direction = np.array([0,0,1])  # z-axis

    # Compute the rotation axis using cross product
    rotation_axis = np.cross(normal_vectors, reference_direction)
    rotation_axis /= np.linalg.norm(rotation_axis, axis=1, keepdims=True)

    # Compute the angle between the normal vectors and the reference direction
    dot_products = np.sum(normal_vectors * reference_direction, axis=1)
    angle = np.arccos(dot_products)

    # Create quaternions
    quaternions = []
    for ax, ang in zip(rotation_axis, angle):
        quaternion = Quaternion(axis=ax, angle=ang)
        quaternions.append([quaternion.scalar, *quaternion.vector])

    return np.asarray(quaternions)

def cartesian_to_unit_spherical(p):
    x = p[:,0]
    y = p[:,1]
    z = p[:,2]
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    return theta, phi

def standardize_points(dataset):
    """
    Standardize the given dataset.

    Parameters:
    - dataset: A numpy array representing the dataset with shape (num_samples, num_features).

    Returns:
    - standardized_dataset: A numpy array representing the standardized dataset.
    """

    # Compute the mean and standard deviation along the first axis (rows)
    mean = np.mean(dataset, axis=0)
    std_dev = np.std(dataset, axis=0)

    # Avoid division by zero by adding a small value (epsilon)
    epsilon = 1e-8
    std_dev += epsilon

    # Standardize the dataset
    standardized_dataset = (dataset - mean) / std_dev

    return standardized_dataset

def normalize_points(dataset):
    # Compute the mean and standard deviation along the first axis (rows)
    # mean = np.mean(dataset, axis=0)
    max_value = np.max(np.abs(dataset), axis=0)

    # Avoid division by zero by adding a small value (epsilon)
    # epsilon = 1e-8
    # std_dev += epsilon

    # Standardize the dataset
    normalized_dataset = dataset / max_value

    return normalized_dataset

def custom_loss(x, y):
    """
    Define a custom loss function.
    For example, you might define a distance metric or similarity measure
    between points x and y.
    """
    return np.sum(np.abs(x - y))  # Example: Manhattan distance

def custom_clustering(data, num_clusters):
    """
    Perform clustering based on a custom loss function.

    Parameters:
    - data: Input data with shape (num_samples, num_features).
    - num_clusters: Number of clusters.

    Returns:
    - cluster_labels: Cluster labels assigned to each data point.
    """

    # Define the custom distance matrix based on the custom loss function
    # distance_matrix = np.zeros((len(data), len(data)))
    # for i, point1 in enumerate(data):
    #     for j, point2 in enumerate(data):
    #         distance_matrix[i, j] = custom_loss(point1, point2)

    # Use KMeans with precomputed distances
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
    kmeans.fit(data)

    return kmeans.labels_

def plane_clustering_loss(points, normals, query_point_normal):

    point = query_point_normal[:3]
    p_normal = np.atleast_2d(query_point_normal[3:])

    # offset_vectors = (points - point)
    # normal_projected_distance =  offset_vectors @ p_normal.T # dot products
    normal_projected_distance = plane_distance(points, point, p_normal)
    # p_normal_projected_distance =  point @ p_normal.T # dot products

    projected_point = point + normal_projected_distance * p_normal
    # Compute the distance between the projected point and the given point
    radial_distances = np.linalg.norm(points - projected_point, axis=1)
    # normal_projected_points = points @ np.array([[0.88,0.44,0.11]]).T

    angular_distances = vector_angular_distance(p_normal, normals)
    
    # normal_projected_distance = normal_projected_distance - normal_projected_distance[k]


    a, b = (0.5, 0.5)
    loss_area = np.pi * a * b 

    # ellipse loss
    loss = angular_distances**2/a**2 + normal_projected_distance**2/b**2

    # circle loss
    # loss = np.sqrt(angular_distances**2 + normal_projected_distance**2)
    mask = (loss < 1).flatten()
    count = np.sum(mask)
    density = count / loss_area
    print(count, density)

    # mask = np.logical_and( angular_distances < 0.1 , normal_projected_distance < 0.2).flatten()

    return mask, loss, angular_distances, normal_projected_distance, radial_distances

def cluster_points(points):
    """
    Cluster 3D points into 2 or 3 clusters using K-means algorithm.

    Parameters:
    - points: numpy array of shape (n, 3) containing 3D points

    Returns:
    - labels: cluster labels assigned to each point
    """
    best_score = -1
    best_labels = None
    best_num_clusters = None
    
    # Test between 2 to n clusters
    for num_clusters in range(2,10):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(points)
        labels = kmeans.labels_
        
        # Evaluate cluster quality using silhouette score
        try:
            score = silhouette_score(points, labels)
        except:
            score = 0.0
        
        # Update best clustering if score is higher
        if score > best_score:
            best_score = score
            best_labels = labels
            best_num_clusters = num_clusters
    
    print(f"Best number of clusters: {best_num_clusters}")
    return best_labels

def build_kd_tree(points):
    """
    Build a KD-tree from a given set of points.

    Parameters:
    - points (array-like): Array of points in the form [[x1, y1], [x2, y2], ...] or [[x1, y1, z1], [x2, y2, z2], ...].

    Returns:
    - tree (KDTree): KD-tree built from the given points.
    """
    points = np.array(points)
    tree = KDTree(points)
    return tree

# def points_within_distance(tree, points, center_point, distance):
#     """
#     Find all points within a specified distance from a given center point using a pre-built KD-tree.

#     Parameters:
#     - tree (KDTree): Pre-built KD-tree of the points.
#     - points (array-like): Array of points in the form [[x1, y1], [x2, y2], ...] or [[x1, y1, z1], [x2, y2, z2], ...].
#     - center_point (array-like): The center point from which distance is measured.
#     - distance (float): The distance threshold.

#     Returns:
#     - within_distance (array): Array of points within the specified distance from the center point.
#     """
#     points = np.array(points)
#     center_point = np.array(center_point)

#     # Query the tree for points within the specified distance from the center point
#     indices = tree.query_ball_point(center_point, distance)

#     # Select the points within the specified distance
#     # within_distance = points[indices]

#     return indices
#     # return within_distance

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def cylindrical_plot(ax, r, theta, z):
    """
    Creates a 3D cylindrical plot.

    Parameters:
    - r: array-like, radial coordinates
    - theta: array-like, angular coordinates (in radians)
    - z: array-like, height coordinates
    """
    # Convert cylindrical coordinates to Cartesian coordinates
    # x = r * np.cos(theta)
    # y = r * np.sin(theta)
    x, y = polar_to_cartesian(r, theta)

    ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

    # # Label axes
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # Set title
    ax.set_title('Cylindrical Plot')

def draw_from_prob_dist(draw_probability):
    draw_probability_cumsum = np.cumsum(draw_probability)
    draw_sum = draw_probability_cumsum[-1]
    draw = np.random.uniform(low=0, high=draw_sum, size=1)
    # k = np.where(draw > draw_probability_cumsum)[0][-1]
    k = np.argmin(np.abs(draw_probability_cumsum - draw))
    # test = draw_probability[k]
    # if test < 0.2:
    #     print(test)
    return k

def standardize_columns(arr):
    """
    Standardize the columns of a 2D array.

    Parameters:
    - arr: numpy array of shape (n, m) to be standardized

    Returns:
    - standardized_arr: numpy array of shape (n, m) with standardized columns
    """
    # Calculate the mean and standard deviation for each column
    col_means = np.mean(arr, axis=0)
    col_stds = np.std(arr, axis=0)

    # Avoid division by zero by replacing zero stds with 1
    col_stds[col_stds == 0] = 1

    # Standardize the columns
    standardized_arr = (arr - col_means) / col_stds

    return standardized_arr

def farthest_point_sampling(points, num_samples):
    """
    Perform farthest point sampling on a set of points.

    Parameters:
    - points (numpy.ndarray): Array of shape (N, D) representing N points in D dimensions.
    - num_samples (int): The number of points to sample.

    Returns:
    - numpy.ndarray: Array of shape (num_samples, D) representing the sampled points.
    - list: Indices of the sampled points in the original array.
    """
    N, D = points.shape
    sampled_points = np.zeros((num_samples, D))
    sampled_indices = []

    # Randomly select the first point
    first_index = np.random.randint(N)
    sampled_points[0] = points[first_index]
    sampled_indices.append(first_index)

    # Initialize the distances to the selected point
    distances = np.linalg.norm(points - points[first_index], axis=1)

    for i in range(1, num_samples):
        # Select the point that is farthest from the selected points
        farthest_index = np.argmax(distances)
        sampled_points[i] = points[farthest_index]
        sampled_indices.append(farthest_index)

        # Update the distances to the closest selected point
        distances = np.minimum(distances, np.linalg.norm(points - points[farthest_index], axis=1))

    return sampled_points, sampled_indices

def calculate_probability_density(values, num_bins=50):
    """
    Calculate the probability density of given values and assign it to each point.

    Parameters:
    - values (numpy.ndarray): Array of values to calculate the histogram for.
    - num_bins (int): Number of bins to use for the histogram.

    Returns:
    - numpy.ndarray: Array of probability densities corresponding to each input value.
    """
    # Calculate the histogram
    hist, bin_edges = np.histogram(values, bins=num_bins, density=True)

    # Calculate the bin widths
    bin_widths = np.diff(bin_edges)

    # Calculate the bin centers
    bin_centers = bin_edges[:-1] + bin_widths / 2

    # Create an array to hold the probability densities for each value
    # probability_densities = np.zeros_like(values, dtype=float)

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    X, _ = np.meshgrid(centers, centers, sparse=True)

    probability_densities = np.array([hist[np.argmin(np.abs(centers - x))] for x in values])

    # Assign the probability density to each value
    # for i, value in enumerate(values):
    #     # Find the bin index for the current value
    #     bin_index = np.digitize(value, bin_edges) - 1
    #     if bin_index >= num_bins:  # Handle edge case where value is exactly the maximum bin edge
    #         bin_index = num_bins - 1
    #     probability_densities[i] = hist[bin_index]

    return probability_densities,hist, X

def calculate_probability_density_2d(values, num_bins=(10, 10), smoothing_kernel_size=3, sigma=1):
    """
    Calculate the probability density of given 2D values and assign it to each point.

    Parameters:
    - values (numpy.ndarray): Array of shape (n, 2) representing the 2D values.
    - num_bins (tuple): Number of bins to use for the 2D histogram in x and y directions.
    - smoothing_kernel_size (int): Size of the Gaussian kernel for smoothing.
    - sigma (float): Standard deviation of the Gaussian distribution for smoothing.

    Returns:
    - numpy.ndarray: Array of probability densities corresponding to each input value.
    """
    def gaussian_kernel(size, sigma=1):
        """
        Create a 2D Gaussian kernel.
        
        Parameters:
        - size (int): Size of the kernel.
        - sigma (float): Standard deviation of the Gaussian distribution.
        
        Returns:
        - kernel (numpy.ndarray): 2D Gaussian kernel.
        """
        kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)), (size, size))
        return kernel / np.sum(kernel)

    def smooth_histogram(hist, kernel):
        """
        Smooth a 2D histogram with a given kernel.
        
        Parameters:
        - hist (numpy.ndarray): 2D histogram.
        - kernel (numpy.ndarray): 2D kernel for smoothing.
        
        Returns:
        - smoothed_hist (numpy.ndarray): Smoothed 2D histogram.
        """
        kernel_size = kernel.shape[0]
        pad_width = kernel_size // 2
        hist_padded = np.pad(hist, pad_width, mode='constant')
        smoothed_hist = np.zeros_like(hist)

        for i in range(hist.shape[0]):
            for j in range(hist.shape[1]):
                smoothed_hist[i, j] = np.sum(hist_padded[i:i+kernel_size, j:j+kernel_size] * kernel)

        return smoothed_hist
    # Calculate the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(values[:, 0], values[:, 1], bins=num_bins, density=True)

    # Create a Gaussian kernel
    kernel = gaussian_kernel(smoothing_kernel_size, sigma)

    # Smooth the histogram with the Gaussian kernel
    hist_smooth = smooth_histogram(hist, kernel)

    # Interpolate the smoothed histogram to get probability densities for the input values
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    prob_density = np.array([hist_smooth[np.argmin(np.abs(x_centers - x)), np.argmin(np.abs(y_centers - y))] for x, y in values])

    return prob_density, hist_smooth, X, Y

def vector_sum_to_matrix(a, b):
    """
    Create a matrix where each element is the sum of elements from two vectors.

    Parameters:
    - a (ndarray): A 1D numpy array.
    - b (ndarray): A 1D numpy array.

    Returns:
    - C (ndarray): A 2D numpy array where C[i, j] = a[i] + b[j].
    """
    # Convert to numpy arrays if not already
    a = np.asarray(a)
    b = np.asarray(b)

    # Use np.einsum to create the desired matrix
    C = np.einsum('i,j->ij', a, np.ones_like(b)) + np.einsum('i,j->ij', np.ones_like(a), b)
    
    return C

if __name__ == "__main__":

    data_name = '../000200'
    # data_name = '../HAP_sweep_ds'
    # data_name = '../sphere_cube_random'
    # data_name = '../bavaria_plane_test_section'
    points = np.load(data_name + ".npy")
    normals = np.load(data_name + "_normals.npy")


    _,sample_indices = farthest_point_sampling(points, 100)

    # selection_mask  = points[:,2] < 0
    # points = points[selection_mask,:]
    # normals = normals[selection_mask,:]
    every_n = 50
    points = points[::every_n,:]
    normals = normals[::every_n,:]


    ## add some noise to the normals in case of the test sphere cube
    # noise = np.random.normal(0.0,0.05, normals.shape)
    # normals += noise
    # normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # points += np.random.uniform(-5,5,3)
    # points += np.random.uniform(-0.01,0.01,points.shape)


    ## example plot 
    # ax = plt.figure().add_subplot(111, projection='3d')
    # # ax.scatter(*points.T, s=1, alpha=0.5)
    # ax.quiver(*points.T, *normals.T, length=0.5)
    # set_equal_aspect(ax)
    # plt.show()

    # make a kd tree of the cloud
    total_points = points.shape[0]
    kdtree = build_kd_tree(points)

    thetas, phis = cartesian_to_unit_spherical(normals)

    pdf_thetas, hist_thetas, x_thetas = calculate_probability_density(thetas)
    pdf_phis, hist_phis, y_phis = calculate_probability_density(phis)
    # pdf_joined, hist, X, Y  = calculate_probability_density_2d(np.c_[thetas, phis], (15,15))
    
    pdf_joined = pdf_thetas + pdf_phis
    hist_joined = vector_sum_to_matrix(hist_phis, hist_thetas)
    
    X,Y = np.meshgrid(x_thetas, y_phis)

    # TT,PP = np.meshgrid(np.sort(thetas), np.sort(phis))
    # PDF = pdf_joined
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)
    # ax.scatter(thetas, pdf_thetas)
    # ax.scatter(phis, pdf_phis)
    # ax.scatter(thetas,pdf_joined)
    ax.plot_surface(X,Y, hist_joined)
    plt.show()



    alpha = 3

    planes = []
    plane_bools = []
    mad_plane = []
    origodistances = []
    mad_angle = []

    # k = np.random.randint(0, len(points)+1) # pick first point at random from the draw probabilty
    
    # base_draw_probability = np.ones(len(points))
    base_draw_probability = pdf_joined +  np.ones(len(points)) * 0.5

    surface_trees = []

    swath_size = 2.0
    desired_swath_part = 0.002

    # plane_indices = set()
    iterations = 0

    fig_planes = plt.figure(figsize=(12,8), layout="constrained")
    ax_planes = fig_planes.add_subplot(111, projection='3d')

    # while len(surface_trees) < 4 and not np.all(base_draw_probability <= 0.5**2):
    while not np.all(base_draw_probability <= 0.5**2) and swath_size >= 0.05 and iterations < 10000:
    # for sample in sample_indices:
        iterations += 1
        # pick a random point
        mask_in_pool = base_draw_probability > 0.0
        count_in_pool = np.sum(mask_in_pool)

        next_point_index = draw_from_prob_dist(base_draw_probability)
        # next_point_index = sample
        # indicies_in_pool = np.where(mask_in_pool)[0]

        point = points[next_point_index,:]
        point_normal = normals[next_point_index, :]
        query_point_normal = np.r_[point, point_normal]

        # normal_projected_distance = plane_distance(points, point, p_normal)
        # angular_distances = vector_angular_distance(query_point_normal, normals) 

        # for i in range(1):
        # get nearby points
        print(swath_size)
        proximate_inliers = np.array(points_within_distance(kdtree, points, point, swath_size))
        

        # half the probability of these points being draw, maybe make it less likely the closer they are to the point

        group_count = len(proximate_inliers)
        if group_count > 0.5*total_points:
            print("Swath size too big and cover more than 50 percent of the cloud")
            swath_size = 0.9*swath_size
            
            continue

        usable_proximate_inliers = proximate_inliers[np.array(mask_in_pool[proximate_inliers])]

        usable_inlier_count = len(usable_proximate_inliers)
        if usable_inlier_count < 4:
            print("not enough points in proximity to make plane model")
            continue

        used_points_part = usable_inlier_count / count_in_pool
        swath_size *= 1 + (desired_swath_part - used_points_part) * 0.5


        proximate_points = points[usable_proximate_inliers]
        proximate_normals = normals[usable_proximate_inliers]



        is_plane_bool, mp, mnv, mad, mad_angle_dist, o_d , c = is_plane(proximate_points, proximate_normals,False)

        base_draw_probability[proximate_inliers] *= 0.5 # in case it os not a plane
        # origodist = plane_distance(mp, np.zeros(3), -mnv)
        # mad_plane.append(mad)
        # mad_angle.append(mad_angle_dist)
        # origodistances.append(origodist)

        # plane_bools.append(is_plane_bool)

        

        if is_plane_bool:
            

            print("Plane found...")


            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(proximate_points[:,0], proximate_points[:,1], proximate_points[:,2])
            # # ax.scatter(*inlier_points.T, alpha=0.7, c='g', s=5, edgecolors=None)
            # ax.quiver(*mp, *mnv)
            # ax.scatter(*mp, c='b', s=200)
            # ax.quiver(*proximate_points.T, *proximate_normals.T, length=0.1, color='k')
            # set_equal_aspect(ax)
            # # ax.plot_surface(plane_mesh_x, plane_mesh_y, plane_mesh_z, alpha=0.3)
            # plt.show()

            # find point that could be in the same plane the found plane by plane distane and normal angular distance

            normal_projected_distance = plane_distance(points, mp, mnv)
            angular_distances = vector_angular_distance(mnv, normals) 

            norm_dist_limit = alpha*mad *3 + 1e-5
            angle_dist_limit = np.deg2rad(30)# alpha*mad_angle_dist *2 + 1e-3

            print(f"search limits: {norm_dist_limit}, {np.rad2deg(angle_dist_limit)}")
            
            prospect_plane_indices = np.where(np.logical_and( np.abs(normal_projected_distance) <= norm_dist_limit,  angular_distances < angle_dist_limit ))[0]
            # prospect_plane_indices = np.where( np.abs(normal_projected_distance) < 0.3)[0]q
            prospect_plane_indices = prospect_plane_indices[mask_in_pool[prospect_plane_indices]]
            prospect_plane_points = points[prospect_plane_indices]
            prospect_plane_normals = normals[prospect_plane_indices]


            if len(prospect_plane_indices) < 20:
                print("Not enough points within prospect plane to make tree")
                continue

            # _is_plane_bool, _mp, prospect_mnv, _mad, _mad_angle_dist, o_d, c = is_plane(prospect_plane_points, prospect_plane_normals, False)

            # plane_basis = construct_orthonormal_basis(prospect_mnv).T # construct a orthonormal basis from the normal vector of the plane


            # prospect_quadtree = Quadtree(transformed_prospect_plane_points[:,:2], min_density=0.1, max_points=250, indices=prospect_plane_indices)
            prospect_quadtree = SurfaceQuadtree(points=prospect_plane_points, normals=prospect_plane_normals, indices=prospect_plane_indices, min_density=0.5, max_points=5)
            
            points_still_in_pool = points[mask_in_pool]
             
            # if set(prospect_plane_indices) in plane_indices:
            #     print("HOOOOLD UUP!") 
            # [plane_indices.add(i) for i in prospect_plane_indices]

            ax_planes.scatter(*prospect_plane_points.T,'.', s=1, alpha=0.8)
            # ax.scatter(*points_still_in_pool.T,'.', s=1, alpha=0.5, c=base_draw_probability[mask_in_pool], vmin=0.0, vmax=1.0)
            # ax.quiver(*prospect_plane_points.T, *prospect_plane_normals.T)
            ax_planes.scatter(*proximate_points.T,'.', s=100, alpha=0.5, color='red')
            # ax.scatter(*prospect_quadtree.flattened_points.T)
            set_equal_aspect(ax_planes)
            fig_planes.canvas.draw_idle()
            plt.pause(0.0001)




            surface_trees.append(prospect_quadtree)
            # if len(surface_trees)%5 == 0:
            #     swath_size *= 0.9
            #     # swath_size -= 0.05

            prospect_quadtree.visualize()
            # prospect_quadtree.visualize_separate_planes(with_points=True)
            # prospect_quadtree.visualize_continoues_plane()
            plt.show()


            for leaf in prospect_quadtree.leaf_nodes():
                cell_points = points[leaf.indices]
                cell_normals = normals[leaf.indices]
                cell_size = leaf.size / 2
                cell_center = np.r_[leaf.center, 0.0]  
                cell_corners = leaf.corner_points
                mnv = leaf.normal
                mp = leaf.normal_point
                origodist = leaf.normal_distance
                curvature = leaf.curvature

        
                # is_plane_bool, mp, mnv, mad, mad_angle_dist  = is_plane(cell_points, cell_normals, False)
                if leaf.is_plane:
                    # origodist = plane_distance(mp, np.zeros(3), -mnv)
                    basis = prospect_quadtree.basis
                    plane_dict = {'center': cell_center,'normal': mnv, 'mad': mad ,'origodist': origodist, 'size': cell_size, "basis": basis, 'corners': cell_corners, 'curvature': curvature}
                    planes.append(plane_dict)

                    mad_plane.append(mad)
                    mad_angle.append(mad_angle_dist)
                    # origodistances.append(origodist)
                    plane_bools.append(is_plane_bool)
                
                base_draw_probability[leaf.indices] = 0.0


                # cell_center = np.r_[leaf.center, 0.0] @ plane_basis
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(*cell_points.T, alpha=1.0, s=10, edgecolors=None, cmap='plasma_r')
                # ax.quiver(*cell_points.T, *cell_normals.T)
                # # ax.scatter(*cell_center, c='r', s=200)
                # set_equal_aspect(ax)
                # plt.show()


            # stop the points being drawn again
            # base_draw_probability[inlier_index] = 0.0
            # next_point_index = proximate_inliers[np.random.choice(farthest_index)]

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(*proximate_points.T, alpha=1.0, c=inlier_angular_distances, s=1, edgecolors=None, cmap='plasma_r')
            # ax.quiver(*point, *point_normal)
            # ax.scatter(*point, c='r', s=200)

            # ax.scatter(*inlier_points.T, alpha=0.7, c='g', s=5, edgecolors=None)
            # ax.quiver(*mp, *mnv)
            # ax.scatter(*mp, c='b', s=200)
            # ax.scatter(*proximate_points[farthest_index].T, c='g', s=200)
            # ax.quiver(*proximate_points.T, *proximate_normals.T, length=0.1, color='k')

            # ax.set_xlabel('X axis')
            # ax.set_ylabel('Y axis')
            # ax.set_zlabel('Z axis')

            # set_equal_aspect(ax)

            # plt.show()

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(*points.T, c=base_draw_probability, s=2, alpha=0.5, edgecolors=None, vmax=1.0, vmin=0.0, cmap='jet')
            # set_equal_aspect(ax)
            # plt.show()  

        # next_point_index = draw_from_prob_dist(base_draw_probability)



    fig, ax = plt.subplots()
    ax.scatter(mad_plane, mad_angle, c=plane_bools,  s=10, alpha=0.7, edgecolors=None)


    plane_normals = []
    plane_origodist = []
    curvature = []
    for plane in planes:
        plane_normals.append(plane['normal'])
        plane_origodist.append(plane['origodist'])
        curvature.append(plane['curvature'])
    plane_normals = np.asarray(plane_normals)
    plane_origodist = np.asarray(plane_origodist)

    





    ref = np.eye(3)
    a_x = vector_angular_distance(ref[:,0], plane_normals)
    a_y = vector_angular_distance(ref[:,1], plane_normals)
    a_z = vector_angular_distance(ref[:,2], plane_normals)

    pdf_ax, hist_ax, bin_centers_ax = calculate_probability_density(a_x)
    pdf_ay, hist_ay, bin_centers_ay = calculate_probability_density(a_y)
    pdf_az, hist_az, bin_centers_az = calculate_probability_density(a_z)
    fig, ax = plt.subplots()
    ax.plot(bin_centers_ax[0], hist_ax)
    ax.plot(bin_centers_ay[0], hist_ay)
    ax.plot(bin_centers_az[0], hist_az)
    plt.show()


    # rpy = normals_to_rpy(plane_normals) 

    ### clustering of planes
    # px, py = polar_to_cartesian(thetas, phis)
    a, b, c = plane_normals.T 
    # px, py = plane_normals[:,2].T
    nz = plane_origodist

    d = - np.einsum('ij,ij -> i', plane_normals, plane_origodist)



    p_points = np.c_[a, b, c, plane_origodist]
    # p_points = np.c_[ax, ay, az, nz]
    # p_points = np.c_[rpy[:,1:], nz]

    # p_points = standardize_columns(p_points)

    from DBSCAN import dbscan_clustering

    # cluster_labels = cluster_points(p_points)
    cluster_labels = dbscan_clustering(p_points, eps=0.3, min_samples=5, standardize=False   )
    # cluster_labels = cluster_with_unknown_k(p_points)
    # cluster_labels = cluster_with_unknown_k_gaussian(p_points)

    cluster_masks = [cluster_labels==c for c in np.unique(cluster_labels)]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    fig = plt.figure(figsize=(14, 8))
    axp = fig.add_subplot(121, projection='3d')
    axp_2 = fig.add_subplot(122, projection='3d')
    axp_2.sharex(axp)
    axp_2.sharey(axp)
    fig.tight_layout()
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    def on_move(event):
        if event.inaxes == axp:
            if axp.button_pressed in axp._rotate_btn:
                axp_2.view_init(elev=axp.elev, azim=axp.azim)
            elif axp.button_pressed in axp._zoom_btn:
                axp_2.set_xlim3d(axp.get_xlim3d())
                axp_2.set_ylim3d(axp.get_ylim3d())
                axp_2.set_zlim3d(axp.get_zlim3d())
        elif event.inaxes == axp_2:
            if axp_2.button_pressed in axp_2._rotate_btn:
                axp.view_init(elev=axp_2.elev, azim=axp_2.azim)
            elif axp_2.button_pressed in axp_2._zoom_btn:
                axp.set_xlim3d(axp_2.get_xlim3d())
                axp.set_ylim3d(axp_2.get_ylim3d())
                axp.set_zlim3d(axp_2.get_zlim3d())
        else:
            return
        fig.canvas.draw_idle()

    index = [0,1,2]
    # index = [0,2,3]
    for i,mask in enumerate(cluster_masks):
        sc = p_points[mask,:]
        ax.scatter(*sc[:,index].T)

        for plane in np.asarray(planes)[mask]:
            plot_plane(axp, plane, color=f'C0{i}')

    axp_2.scatter(*points[mask_in_pool,:].T,'.', s=1, alpha=0.2,c='k', edgecolors=None)
    for tree in surface_trees:
        axp_2.scatter(*tree.points[::1,:].T,'.', s=5, alpha=0.8, edgecolors=None)

    set_equal_aspect(axp)
    set_equal_aspect(axp_2)
    # set_equal_aspect(ax)

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()




            # # Plot histogram
            # hist, bins = np.histogram(angular_distances, bins=62)  # Adjust the number of bins as needed
            # fig, ax = plt.subplots()
            # ax.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black')  # Plot bars with correct width
            
            # Plot histogram
            # hist, bins = np.histogram(thetas, bins=62)  # Adjust the number of bins as needed
            # fig, ax = plt.subplots()
            # ax.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black')  # Plot bars with correct width
            
            # # Plot histogram
            # hist, bins = np.histogram(phis, bins=62)  # Adjust the number of bins as needed
            # fig, ax = plt.subplots()
            # ax.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black')  # Plot bars with correct width


            # peak_indices = []
            # for i in range(1, len(hist) - 1):
            #     if hist[i - 1] < hist[i] > hist[i + 1]:
            #         peak_indices.append(i)

            # print(peak_indices)

            # standardized_normal_space_points = np.c_[normalize_points(angular_distances), normalize_points(normal_projected_distance)]


            # best_labels = cluster_points(normal_space_points)
            # cluster_labels = cluster_with_unknown_k(standardized_normal_space_points)
            # cluster_labels = cluster_with_unknown_k_gaussian(standardized_normal_space_points)

            # cluster_masks = [cluster_labels==c for c in np.unique(cluster_labels)]

            # # select cluster closet to origo
            # dist = np.inf
            # c_idx = -1
            # for i, m in enumerate(cluster_masks):
            #     # mean_projected_distance = np.mean(normal_projected_distance[m])
            #     mean_angular_distance = np.mean(angular_distances[m])
            #     # d = np.linalg.norm(np.array([mean_projected_distance, mean_angular_distance]))
            #     d = np.linalg.norm(np.array([mean_angular_distance]))
            #     if d < dist:
            #         dist = d
            #         c_idx = i

            # mask = cluster_masks[c_idx]

            # # Find the mean distance of the points that where "inside" the loss
            # mean_projected_distance = np.mean(normal_projected_distance[mask])
            # mean_angular_distance = np.mean(angular_distances[mask])
            # mean_normal = np.mean(normals[mask], axis=0)
            # mean_normal = mean_normal/np.linalg.norm(mean_normal)
            # print(mean_normal)

            # plt.figure(figsize=(8, 6))
            # plt.scatter(angular_distances[mask], normal_projected_distance[mask], s=3, alpha=0.2,  c='r')
            # plt.scatter(angular_distances[~mask], normal_projected_distance[~mask], s=3, alpha=0.2,  c='b')
            # plt.scatter(mean_angular_distance, mean_projected_distance, s=15, alpha=1,  c='g')
            # plt.xlabel('Angular distance [rad]')
            # plt.ylabel('Planar distance [m]')
            # plt.title('First planar and normal angular distances')
            
            # plt.show()

            # pick the index that is closest
            # k = np.argmin(np.abs(normal_projected_distance[mask] - mean_projected_distance) + (angular_distances[mask] - mean_angular_distance))

            # create a new query that represnt the plane
            # mean_point = point + mean_projected_distance * p_normal
            # mean_point = query_point_normal[:3] + mean_projected_distance * mean_normal
            # mean_point = np.mean(points[mask,:], axis=0) +  mean_projected_distance * p_normal
        #     mean_point = np.mean(points[mask,:], axis=0)
        #     query_point_normal = np.r_[mean_point, mean_normal]
        
        # planes.append(points[mask,:])
        # points = points[~mask,:]
        # normals = normals[~mask,:]

    # plane = points[mask,:]
    # other = points[~mask,:]


    # make loss calculation based on the new index
    # mask, plane_loss, angular_distances, normal_projected_distance, radial_distances = plane_clustering_loss(points, normals, query_point_normal)


    

    # point = mean_point
    # p_normal = np.atleast_2d(mean_normal)




    # cluster_with_unknown_k(np.c_[phis,thetas])


    # plt.figure(figsize=(8, 6))
    # plt.scatter(radial_distances,plane_loss, s=3, alpha=0.2)
    # plt.xlabel('Radial distance [m]')
    # plt.ylabel('plane Loss ')
    # plt.title('Plane Loss')

    # plt.figure(figsize=(8, 6))
    # plt.scatter(radial_distances[mask], normal_projected_distance[mask], s=3, alpha=0.2,  c='r')
    # plt.scatter(radial_distances[~mask], normal_projected_distance[~mask], s=3, alpha=0.2, c='b')
    # plt.xlabel('Radial distance [m]')
    # plt.ylabel('Planar distance [m]')
    # plt.title('Radial and Planar distances')

    # plt.figure(figsize=(8, 6))
    # plt.scatter(angular_distances[mask], normal_projected_distance[mask], s=3, alpha=0.2,  c='r')
    # plt.scatter(angular_distances[~mask], normal_projected_distance[~mask], s=3, alpha=0.2,  c='b')
    # plt.xlabel('Angular distance [rad]')
    # plt.ylabel('Planar distance [m]')
    # plt.title('planar and normal angular distances')

    # plt.figure(figsize=(8, 6))
    # plt.scatter(plane[:,0], plane[:,1], color='r', s=4)
    # plt.scatter(other[:,0], other[:,1], s=4)
    # plt.scatter(points[k,0], points[k,1], s=100, color='g')

    # sd_points = standardize_points(points)

    # nq = normal_to_quaternion(normals)


    # p_n = np.c_[sd_points,normals]



    # # Create a 3D scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for plane in planes:
    #     ax.scatter(plane[:,0], plane[:,1], plane[:,2], marker='o', s=3, alpha=0.2)  # c is color, marker is the shape of the point
    #     # ax.scatter(other[:,0], other[:,1], other[:,2], c='C0', marker='o', s=3, alpha=0.2)  # c is color, marker is the shape of the point
    #     # ax.scatter(point[0], point[1], point[2], c='g', marker='o', s=10, alpha=1.0)  # c is color, marker is the shape of the point
    #     # ax.quiver(point[0], point[1], point[2],p_normal[0][0], p_normal[0][1], p_normal[0][2], color='k')  
    #     ax.set_xlabel('X Label')
    #     ax.set_ylabel('Y Label')
    #     ax.set_zlabel('Z Label')
    #     plt.title('3D Scatter Plot')


    # plt.figure(figsize=(8, 6))
    # plt.plot(np.arange(len(normal_projected_points)), normal_projected_points)
    # plt.title('Normals Projected points Z')
    # plt.xlabel('n_x')
    # plt.ylabel('n_y')

    # plt.figure(figsize=(8, 6))
    # plt.scatter(, np.sort(points[:,0]), alpha=0.05, s= 5)
    # plt.title('Normals X')
    # plt.xlabel('n_x')
    # plt.ylabel('n_y')



    # plt.figure(figsize=(8, 6))
    # plt.scatter(normals[mask,0], normals[mask,1], alpha=0.05, s= 5)
    # plt.title('Normals Space')
    # plt.xlabel('n_x')
    # plt.ylabel('n_y')


    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, polar=True)
    # plt.scatter(phis, np.rad2deg(thetas), alpha=0.05, s=5)
    # plt.title('Normal Spherical Coordinates ')
    # plt.xlabel('Polar Angle')
    # plt.ylabel('Azimuth Angle')



    # plt.figure(figsize=(8, 6))
    # plt.scatter(phis, thetas, alpha=0.05, s= 5)

    # plt.title('Normals Space')
    # plt.xlabel('Azimuth')
    # plt.ylabel('Polar Angle')



