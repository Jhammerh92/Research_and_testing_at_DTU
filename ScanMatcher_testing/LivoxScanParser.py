import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm, multivariate_normal
from scipy.spatial.transform import Rotation
import random
import open3d as o3d

def rotation_matrix_to_align_z_axis(target_vector):
    """
    Generate a 3x3 rotation matrix to align the Z-axis with a target vector.

    Parameters:
    - target_vector (array-like): The target vector to align with the Z-axis.

    Returns:
    - rotation_matrix (array): The 3x3 rotation matrix.
    """
    target_vector = np.array(target_vector)
    target_vector = target_vector / np.linalg.norm(target_vector)  # Normalize the target vector

    # Define the Z-axis
    z_axis = np.array([0, 0, 1])

    # Compute the rotation axis and angle using cross product and dot product
    rotation_axis = np.cross(z_axis, target_vector)
    rotation_angle = np.arccos(np.dot(z_axis, target_vector))

    # Use Rodrigues' rotation formula from scipy
    rotation_matrix = Rotation.from_rotvec(rotation_axis * rotation_angle).as_matrix()

    return rotation_matrix

def compute_eigen_decomposition(point_cloud):
    """
    Compute eigenvalue decomposition of the covariance matrix of a 3D point cloud.

    Parameters:
    - point_cloud (array-like): The 3D point cloud where each row represents a point.

    Returns:
    - eigenvalues (array): The eigenvalues of the covariance matrix.
    - eigenvectors (array): The eigenvectors of the covariance matrix.
    """
    point_cloud = np.array(point_cloud)

    # Compute the mean of the point cloud
    mean_point = np.mean(point_cloud[:, :3], axis=0)

    # Center the point cloud
    centered_points = point_cloud[:, :3] - mean_point

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_eigenvectors = eigenvectors[:, sort_indices]

    return sorted_eigenvectors, sorted_eigenvalues

class KNearestNeighbors:
    def __init__(self, cloud):
        """
        Initialize the KNearestNeighbors class with a 3D point cloud.

        Parameters:
        - point_cloud (array-like): The 3D point cloud where each row represents a point.
        """
        self.point_cloud = np.array(cloud)
        self.tree = cKDTree(self.point_cloud)  # Create a KD-tree based on (x, y, z) coordinates

    def query_k_neighbors(self, query_point=None, point_index=None, k=1, radius=float("inf") ):
        """
        Find the k-nearest neighbors of a query point in the initialized point cloud.

        Parameters:
        - query_point (array-like): The coordinates of the query point.
        - k (int): The number of neighbors to find.

        Returns:
        - neighbors (list): List of indices of the k-nearest neighbors in the point cloud.
        """
        if query_point is None:
            query_point = self.point_cloud[point_index, :]

        # Query the KD-tree to find k-nearest neighbors
        d, indices = self.tree.query(query_point, k=k, distance_upper_bound=radius)
        # if any(d  >= float('inf')):
        #     print('help!')

        indices = indices[np.where(d  < float('inf'))]
        return indices
    
    def query_radius_neighbors(self,radius, query_point=None, point_index=None, ):
        """
        Find the k-nearest neighbors of a query point in the initialized point cloud.

        Parameters:
        - query_point (array-like): The coordinates of the query point.
        - k (int): The number of neighbors to find.

        Returns:
        - neighbors (list): List of indices of the k-nearest neighbors in the point cloud.
        """
        if query_point is None:
            query_point = self.point_cloud[point_index, :]

        # Query the KD-tree to find k-nearest neighbors

        indices = self.tree.query_ball_point(query_point, radius)
        # if any(d  >= float('inf')):
        #     print('help!')

        # indices = indices[np.where(d  < float('inf'))]
        return np.array(indices)

class PCD:
    """
    Docstring
    """
    def __init__(self):
        self.data = None
        self._x = None
        self._y = None
        self._z = None
        self._points = None
        self._normals = None
        self._curvature = None
        self._len = None
        self._headers = None
        self._kdtree = None

        self.o3d_pcd = None

    def _create_pcd_from_points(self, points, normals=None):
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(points)
        if not (normals is None):
            o3d_pcd.normals = o3d.utility.Vector3dVector(normals)
        return o3d_pcd

    def load_from_csv(self, path_to_csv, n=None):
        # self.data = np.genfromtxt(path_to_csv, dtype=float, delimiter=',', names=True)
        cols = [x for x in range(7)]
        self._headers = np.genfromtxt(path_to_csv, dtype='str', delimiter=',', max_rows=1, usecols=cols)
        self.data = np.loadtxt(path_to_csv, dtype="float", delimiter=',', max_rows=n, skiprows=1, usecols=cols)

        # for i,row in enumerate(np.flip(self.data, axis=0)):
        #     if row[3] == 0:
        #         np.delete(self.data, self.data[:,3]==0, 0)
        self.data = np.delete(self.data, self.data[:,3]==0.0 , 0)

        for i, attr in enumerate(self._headers):
            setattr(self, "_"+attr.lower(), self.data[:,i])

        input_points = np.c_[self._x, self._y, self._z]

        self.o3d_pcd = self._create_pcd_from_points(input_points)
        self._set_points()
        # self._points = np.c_[self._x, self._y, self._z]
        # self.remove_zero_points()
        # self._x =  self._points[:,0]
        # self._y =  self._points[:,1]
        # self._z =  self._points[:,2]
        # self._len = len(self._x)

        self._kdtree = KNearestNeighbors(self._points)

    def load_from_o3d(self, filename):

        self.o3d_pcd = o3d.io.read_point_cloud(filename)
        self._set_points()        

        self._kdtree = KNearestNeighbors(self._points)

    def _set_points(self):
        self._points = np.asarray(self.o3d_pcd.points)
        self.remove_zero_points()
        self._x =  self._points[:,0]
        self._y =  self._points[:,1]
        self._z =  self._points[:,2]
        self._len = len(self._x)

    def remove_zero_points(self):
        n = np.linalg.norm(self._points, axis=1)
        self._points = self._points[np.where(n > 1e-8)]


    def parse_data(self, new_data, headers):
        """
        docstring
        """
        self.data = new_data
        self._headers = headers
        
        for i, attr in enumerate(self._headers):
            setattr(self, attr.lower(), self.data[:,i])

        self._points = np.c_[self._x, self._y, self._z]
        self._len = len(self._x)

    def get_points(self, leaf_size=None):
        """
        dasd
        """
        if not leaf_size == None:
            self.o3d_pcd = self.o3d_pcd.voxel_down_sample(leaf_size)

        self._set_points()

        return self._points
    
    def set_points(self, new_points):
        """
        dasd
        """
        self._points = new_points
        self._len = self._points.shape[0]
        self._kdtree = KNearestNeighbors(self._points)
 
    
    def query_k_neighbors(self, query_point=None, query_index=None, k=1, radius=float("inf") ):
        return self._kdtree.query_k_neighbors(query_point=query_point, point_index=query_index, k=k, radius=radius)
    
    def query_radius_neighbors(self,radius, query_point=None, query_index=None):
        return self._kdtree.query_radius_neighbors(query_point=query_point, point_index=query_index, radius=radius)

    
    def compute_normals_and_curvatures(self, kmax=10, kmin=4, search_radius=float("inf"), feature_radius=0.2):
        """
        Compute normals and 3D point curvatures of a point cloud.

        Parameters:
        - point_cloud (array-like): The 3D point cloud where each row represents a point.
        - radius (float): Radius for neighborhood search to compute normals and curvatures.

        Returns:
        - normals (array): Normal vectors for each point in the point cloud.
        - curvatures (array): Principal curvatures for each point in the point cloud.
        """

        point_cloud = self._points
        # num_points = point_cloud.shape[0]

        # if feature_radius is None:
        #     feature_radius = search_radius

        self._normals = np.zeros((self._len, 3))
        self._curvature = np.zeros(self._len)
        index_array = np.arange(self._len)
        not_used_points_index = set(range(self._len))
        # edge_points_index = set()
        # surface_points_index = set()
        counts = np.zeros(self._len)
        prob_sums = np.zeros(self._len)

        while (np.any(prob_sums < 10) and len(not_used_points_index)>0):
        # for i in range(self._len//10):
        # for i in [38500]:
            # i = random.choice(list(not_used_points_index))
            unused_index_list = list(not_used_points_index)
            # if prob_sums[i] > 5.0:
            #     # print('skipped', i)
            #     continue

            # print('Remaining', len(not_used_points_index))

            prob_cumsum = np.cumsum(1.0/((prob_sums[unused_index_list])**2 + 1)) # will pick those points with lowest "probability-sum" to search the set more quickly, and can also only pick from points that have not been used
            prob_cumsum /= prob_cumsum[-1]
            draw = np.random.uniform(0, 1.0)
            # while not ((i:= np.argmin( np.abs( counts_cumsum - draw ))) in not_used_points_index):
            #     draw = np.random.uniform(0, 1.0)
            i_ = np.argmin( np.abs( prob_cumsum - draw ))
            i =  index_array[unused_index_list[i_]]
            not_used_points_index.remove(i)
            
            query_point = self._points[i]


            # Extract neighborhood within the specified radius
            # neighborhood_indices = self.query_k_neighbors(None, i, kmax, radius=search_radius)
            neighborhood_indices = self.query_radius_neighbors(radius=search_radius, query_index=i)
            radius_temp = search_radius
            while len(neighborhood_indices) < kmin:
                # neighborhood_indices = self.query_k_neighbors(None, i, kmax, radius=radius_temp)
                neighborhood_indices = self.query_radius_neighbors(radius=radius_temp, query_index=i)
                radius_temp *= 1.5


            neighborhood = point_cloud[neighborhood_indices, :3]
            # Compute covariance matrix and eigen decomposition
            sorted_eigenvectors, sorted_eigenvalues = compute_eigen_decomposition(neighborhood)
            eigenval_ratio = sorted_eigenvalues[1]/sorted_eigenvalues[2]
            # if  eigenval_ratio == float('inf') :
            #     print('help')

            while  eigenval_ratio > 1e5 and not eigenval_ratio == float('inf') :
                radius_temp *= 1.5
                neighborhood_indices = self.query_radius_neighbors(radius=radius_temp, query_index=i)
                # neighborhood_indices = self.query_k_neighbors(None, i, kmax, radius=radius_temp)
                # indices_in_radius = np.where(np.linalg.norm(point_cloud[:, :3] - point_cloud[i, :3], axis=1) < radius)[0]

                neighborhood = point_cloud[neighborhood_indices, :3]
                # Compute covariance matrix and eigen decomposition
                sorted_eigenvectors, sorted_eigenvalues = compute_eigen_decomposition(neighborhood)
                eigenval_ratio = sorted_eigenvalues[1]/sorted_eigenvalues[2]
                # print(sorted_eigenvalues, eigenval_ratio, radius_temp, len(neighborhood_indices))

            # neighborhood_displacement_vectors = neighborhood - query_point
            # neighborhood_distances = np.linalg.norm(neighborhood_displacement_vectors, axis=1)

            # Normal vector is the eigenvector corresponding to the smallest eigenvalue
            neighborhood_normal = sorted_eigenvectors[:, 2]

            # if normal.dtype == np.complex128:
            #     print('complex normal!')

            neighborhood_normal = align_normal_with_view_position(neighborhood_normal, self._points[i])

            surface_covariance_matrix = create_plane_cov_matrix(neighborhood_normal, feature_radius=feature_radius)

            # neighborhood_prob_dist = norm.pdf(neighborhood_distances, scale=(feature_radius/2)**2 )
            # neighborhood_prob_dist /= neighborhood_prob_dist[0]
            # neighborhood_prob = multivariate_normal.pdf(x=neighborhood_displacement_vectors, mean=None, cov=surface_covariance_matrix)
            # neighborhood_prob /= neighborhood_prob[0]
            neighborhood_prob = prob_of_point_in_plane(neighborhood, surface_covariance_matrix, mean=query_point, normalize=True)
            

            # if len(np.where(neighborhood_prob > 0.01)[0]) < 3 :
            #     print("help!")

            # plane_indices = np.where(neighborhood_prob > 0.01)[0] # should compare to the cfd
            # plane_neighborhood_indices = neighborhood_indices[plane_indices]
            # plane_neighborhood = point_cloud[plane_neighborhood_indices, :3]

            # sorted_eigenvectors, sorted_eigenvalues = compute_eigen_decomposition(plane_neighborhood)
            # plane_normal = sorted_eigenvectors[:, 2]
            # plane_normal = align_normal_with_view_position(plane_normal, self._points[i])

            # plane_covariance_matrix = create_plane_cov_matrix(plane_normal, epsilon_std=0.005, feature_radius=feature_radius)
            # plane_prob = prob_of_point_in_plane(neighborhood, plane_covariance_matrix, mean=query_point, normalize=True)

            self._normals[neighborhood_indices,:] += np.tile(neighborhood_normal, (len(neighborhood_indices), 1)) *  neighborhood_prob[:,np.newaxis]
            # self._normals[neighborhood_indices,:] += np.tile(plane_normal, (len(neighborhood_indices), 1)) *  plane_prob[:,np.newaxis]
            # self._normals[i,:] = neighborhood_normal
            # self._normals[i,:] = plane_normal

            counts[neighborhood_indices] += 1
            # counts[plane_neighborhood_indices] += 1
            # prob_sums[neighborhood_indices] += neighborhood_prob * neighborhood_prob_dist
            # prob_sums[neighborhood_indices] += plane_prob
            prob_sums[neighborhood_indices] += neighborhood_prob
            # prob_sums[plane_neighborhood_indices] += neighborhood_prob[plane_indices]

            # print(np.sum(prob_sums > 1))

            # Calculate principal curvatures
            curvature_ratio = sorted_eigenvalues[1] / sorted_eigenvalues[0]
            curvature = np.sqrt(sorted_eigenvalues[1] / (sorted_eigenvalues[0] + sorted_eigenvalues[1]))
            # self._curvature[i] = np.sqrt(sorted_eigenvalues[1] / (sorted_eigenvalues[0] + sorted_eigenvalues[1]))
            self._curvature[neighborhood_indices] = curvature * neighborhood_prob

        # self._normals /= prob_sums[:,np.newaxis]
        self._normals /= np.linalg.norm(self._normals, axis=1)[:,np.newaxis]
        self._curvature /= prob_sums
        
        print(f"{self._len - len(not_used_points_index)} points used of {self._len} points")

        return self._normals, self._curvature
    
def create_plane_cov_matrix(plane_normal, epsilon_std=0.02, feature_radius=0.2):
    epsilon = epsilon_std**2
    scale = (feature_radius/2)/np.sqrt(2)**2 # 98% drop off at feature_radius
    surface_covariance_matrix  = np.array([[scale,0,0],[0,scale,0],[0,0, epsilon]])
    aligning_rot_mat = rotation_matrix_to_align_z_axis(plane_normal)
    surface_covariance_matrix = aligning_rot_mat @ surface_covariance_matrix @ aligning_rot_mat.T
    
    return surface_covariance_matrix

def align_normal_with_view_position(normal, view_position):
    normal_dir = np.dot(normal,  - view_position)
    # if normal_dir > 0: # REMEMBER TO FLIP THIS BACK
    if normal_dir < 0: 
        normal = - normal
    return normal

def prob_of_point_in_plane(points, plane_cov_matrix, mean=None, normalize=True):
    if mean is None:
        mean = np.zeros(plane_cov_matrix.shape[0])
    # displacement_vectors = points - mean
    prob = multivariate_normal.pdf(x=points, mean=mean, cov=plane_cov_matrix)
    if normalize:
        prob /= prob[0]
    # prob = 1 - multivariate_normal.cdf(x=points, mean=mean, cov=plane_cov_matrix) # is this slow?

    return prob





class LivoxScanParser(PCD):
    def __init__(self):
        self.data = None
        self._x = None
        self._y = None
        self._z = None
        self._points = None
        self._len = None
        self._headers = None

    def load_from_csv(self, path_to_csv, n=None):
        """
        Docstring
        """
        # self = self.__init__()

        # self.data = np.genfromtxt(path_to_csv, dtype=float, delimiter=',', names=True)
        cols = [x for x in range(7,19) if x != 4]
        self._headers = np.genfromtxt(path_to_csv, dtype='str', delimiter=',', max_rows=1, usecols=cols)
        self.data = np.loadtxt(path_to_csv, dtype="float", delimiter=',', max_rows=n, skiprows=1, usecols=cols)

        # for i,row in enumerate(np.flip(self.data, axis=0)):
        #     if row[3] == 0:
        #         np.delete(self.data, self.data[:,3]==0, 0)
        self.data = np.delete(self.data, self.data[:,3]==0.0 , 0)

        for i, attr in enumerate(self._headers):
            setattr(self, "_"+attr.lower(), self.data[:,i])

        self._points = np.c_[self._x, self._y, self._z]
        self._len = len(self._x)

        # remove points with 0-coordinates


if __name__ == "__main__":
    scan = LivoxScanParser()
    scan.load_from_csv(f"Mid-70 Test Data/Livox70_3s150cm.csv")
    
    print(scan.get_points()[0:5,:])
    

