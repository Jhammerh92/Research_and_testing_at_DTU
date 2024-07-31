import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3d

from LivoxScanParser import PCD, compute_eigen_decomposition
from PlotPointCloud3d import plot_cloud_3d_o3d_path, plot_cloud_3d_o3d, _create_pcd_from_points



class ProbalisticRANSAC():
    def __init__(self, points, test_function, samplesize=1) -> None:
        self.points = points
        self.in_model_prob = np.full((len(points), 1), 1.0/len(points))
        self.samplesize = samplesize
        self.test_function = test_function
        self.inliers = np.zeros((0), dtype=np.int64)
        self.drawn_index = set()

    def run_ransac(self, iterations):
        best_inlier_count = 0
        for i in range(iterations):
            inliers_ = self.test_model()
            inlier_count = len(inliers_)
            if inlier_count > best_inlier_count:
                self.inliers = inliers_
                best_inlier_count = inlier_count

        return self.inliers

    def draw_index(self):
        idx = []
        for i in range(self.samplesize):
            draw = np.random.uniform(0, 1)
            in_model_prob_cumsum = np.cumsum(self.in_model_prob)

            idx.append(np.argmin( np.abs( in_model_prob_cumsum - draw )))

            # if idx[0] in self.drawn_index:
            #     idx[0] += 1

        idx = np.asarray(idx)
        self.drawn_index.add(idx[0])
        return idx
    
    def test_model(self):
        test_idx = self.draw_index()
        inliers = self.test_function(test_idx, self.points)
        self.adjust_in_model_prob(inliers)
        return inliers
   
    def adjust_in_model_prob(self, inlier_index):
        inlier_mask = np.zeros(len(self.points), dtype=bool)
        inlier_mask[inlier_index] = True
        # how the probability of found inliers and outliers should be scaled/calculated/updated can be explored in many ways I think
        # But it should most likely incorporate the deviation from the found model 
        self.in_model_prob[inlier_mask] *= (1. + 0.05)
        # self.in_model_prob[~inlier_mask] *= (1. - 0.01)
        self.in_model_prob /= np.sum(self.in_model_prob)
        self.in_model_prob.flatten()

    def get_inlier_index(self):
        return self.inliers
    
    def get_inlier_probs(self, prob_threshold=None):
        if prob_threshold is None:
            prob_threshold = 1.0/len(self.points)
        prob_idx = np.array(np.where(self.in_model_prob[:,0] > prob_threshold)[0]).ravel()
        return self.in_model_prob, prob_idx


def points_near_vector(cloud, origin, direction_vector, distance_threshold):
    """
    Find all points within a distance threshold along a vector from a specific origin point.

    Parameters:
    - point_cloud (array-like): The 3D point cloud where each row represents a point.
    - origin (array-like): The coordinates of the origin point.
    - direction_vector (array-like): The direction vector along which to search for points.
    - distance_threshold (float): The maximum distance allowed from the origin along the vector.

    Returns:
    - nearby_points (list): List of indices of points within the distance threshold.
    """
    cloud = np.array(cloud)
    origin = np.array(origin)
    direction_vector = np.array(direction_vector) / np.linalg.norm(direction_vector)  # Normalize the direction vector

    vectors =  cloud[:, :3] - origin[:3]
    # Project points onto the vector
    
    # projections = np.tile(direction_vector, [n,1]) * np.resize(np.dot(vectors, direction_vector), (n,1))
    # projections = np.dot(vectors, direction_vector)
    # rejections = (vectors / np.resize(projections, (n,1)))
    # rejections -= np.tile(direction_vector, (n,1))
    # distances = np.linalg.norm(rejections, axis=1)

    rejection = np.linalg.norm(np.cross(vectors, direction_vector), axis=1)
    distances = rejection
    # Find points within the distance threshold
    nearby_points_indices = np.where((distances <= distance_threshold))[0]

    return nearby_points_indices


def detect_planes_and_edges(point_cloud:PCD, plane_distance_threshold=0.1, edge_threshold=1.0):
    # Convert the point cloud to a numpy array
    points = np.array(point_cloud.get_points())
    # KNN = KNearestNeighbors(points)
    # Detect planes using RANSAC
    plane_model = RANSACRegressor(residual_threshold=plane_distance_threshold)
    plane_model.fit(points[:, :2], points[:, 2])

    # Extract inliers (points on the detected plane)
    plane_inliers = points[plane_model.inlier_mask_]

    def ransac_test_near_vector(idx, points):
        # KNN = KNearestNeighbors(points)
        point_group_idx = point_cloud.query_k_neighbors(query_index=idx, k=20).flatten()
        point_group = points[point_group_idx, :]
        group_mean = np.mean(point_group, axis=0)
        eigen_vec, eigen_val = compute_eigen_decomposition(point_group)
        
        if not (eigen_val[0] > eigen_val[1]*100):
            return np.zeros((0,0), dtype=np.int8)
        edge_vector = eigen_vec[:,0]

        edge_points_idx = points_near_vector(points, group_mean, edge_vector, edge_threshold)
        return edge_points_idx

    PRANSAC = ProbalisticRANSAC(points, ransac_test_near_vector, samplesize=1)

    PRANSAC.run_ransac(300)
    edge_points_idx = PRANSAC.get_inlier_index()
    edge_points_prob, edge_points_prob_idx = PRANSAC.get_inlier_probs()
    edge_points = points[edge_points_idx, :]
    # edge_points = points[edge_points_prob>0.02, :]


    return plane_inliers, edge_points, edge_points_prob, edge_points_prob_idx
    # return plane_inliers




if __name__ == "__main__":
    # Example usage
    # np.random.seed(42)

 
    pcd = PCD()
    # pcd.load_from_csv("/Users/jhh/Desktop/ASL Dataset/Stairs/local_frame/Hokuyo_0.csv")
    pcd.load_from_o3d("sphere_cube_random_10000.pcd")


    # pcd._points = np.vstack([pcd._points, np.array([0,0,0])])
    point_cloud = pcd.get_points()
    # pcd.set_points(point_cloud[np.where(point_cloud[:,2] >= 0.0)[0], :]) #- np.array([0,0,5])) 
    point_cloud = pcd.get_points()
    # plot_cloud_3d_o3d(point_cloud[np.where(point_cloud[:,2] <= 0.0)[0], :])

    pcd.compute_normals_and_curvatures(kmax=100000, kmin=5, search_radius=0.2, feature_radius=0.2)

    # Detect planes and edges
    detected_planes, detected_edges, detected_edges_prob, prob_idx = detect_planes_and_edges(pcd, 0.1, 0.1)
    # detected_planes = detect_planes_and_edges(point_cloud, 0.2)

    detected_edges_prob = point_cloud[prob_idx, :]

    plot_cloud_3d_o3d(pcd.get_points(), normals=pcd._normals)
    # plot_cloud_3d_o3d(pcd.get_points(), normals=pcd._normals, colors=pcd._curvature)
    # plot_cloud_3d_o3d(detected_planes)
    # plot_cloud_3d_o3d(detected_edges)

    # plot_cloud_3d_o3d(np.vstack((detected_planes, detected_edges)))


    # fig = plt.figure(figsize=(12, 6))
    # plt.plot(prob*100)



    plt.show()
