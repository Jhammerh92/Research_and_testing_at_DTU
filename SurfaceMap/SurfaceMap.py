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


class SurfaceMap:
    def __init__(self) -> None:
        self.surface_trees = []
        self.tree_kd = None

    def visualize_map(self):
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')

        for tree in self.surface_trees:
            tree.visualize_separate_planes(ax)


        plt.show()

    def query_point(self, query_point, normals=None, normal_centers=None):
        if normals is None and normal_centers is None:
            normals = np.array([surface.surface_normal for surface in self.surface_trees])
            normal_centers = np.array([surface.center for surface in self.surface_trees])

        # dists = [np.abs(plane_distance(query_point, surface.center, surface.surface_normal)) for surface in self.surface_trees]
        
        dists = normal_distances(query_point, normals, normal_centers)
        idx = np.argmin(np.abs(dists))

        is_within = self.surface_trees[idx].point_within_tree(query_point)

        if is_within:
            corresponding_node = self.surface_trees[idx].get_node(query_point)
            if corresponding_node:


                plane_normal, plane_point = corresponding_node.get_plane_information()

                error = - plane_distance(query_point, plane_point, plane_normal)
                error_vector =  plane_normal * error

                # fig = plt.figure(figsize=(14, 8))
                # ax = fig.add_subplot(111, projection='3d')
                # self.surface_trees[idx].visualize_separate_planes(ax, with_points=False)
                # self.surface_trees[idx].visualize()
                # ax.scatter(*query_point, c='r', s=100)
                # ax.quiver(*query_point, *error_vector)
                # set_equal_aspect(ax)
                # plt.show()

                return error_vector

    def query_points(self, query_points):
        normals = np.array([surface.surface_normal for surface in self.surface_trees])
        normal_centers = np.array([surface.center for surface in self.surface_trees])

        error_vectors = []
        inlier_points = []
        for query_point in query_points:
            if not (error_vector:= self.query_point(query_point, normals, normal_centers)) is None:
                error_vectors.append(error_vector)
                inlier_points.append(query_point)

        return np.array(error_vectors), np.array(inlier_points)


    def calc_error_vectors(self, query_points):

        

        for q_point in query_points:
            self.query_point(q_point)



    def create_initial_map(self, points, normals):

        swath_size = 2.0
        desired_swath_part = 0.002
        base_draw_probability = np.ones(len(points))
        iterations = 0
        kdtree = build_kd_tree(points)
        alpha = 3
        total_points = points.shape[0]

        while not np.all(base_draw_probability <= 0.5**2) and swath_size >= 0.05 and iterations < 10000:
            iterations += 1
            mask_in_pool = base_draw_probability > 0.0
            count_in_pool = np.sum(mask_in_pool)

            # pick a random point from pdf
            next_point_index = draw_from_prob_dist(base_draw_probability)
            # next_point_index = sample

            point = points[next_point_index,:]

            proximate_inliers = np.array(points_within_distance(kdtree, points, point, swath_size))
            
            # half the probability of these points being drawn, maybe make it less likely the closer they are to the point

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


            if is_plane_bool:
                print("Plane found...")
                # find point that could be in the same plane the found plane by plane distane and normal angular distance

                normal_projected_distance = plane_distance(points, mp, mnv)
                angular_distances = vector_angular_distance(mnv, normals) 

                norm_dist_limit = alpha*mad *3 + 1e-5
                angle_dist_limit = np.deg2rad(30)# alpha*mad_angle_dist *2 + 1e-3

                print(f"search limits: {norm_dist_limit}, {np.rad2deg(angle_dist_limit)}")
                
                prospect_plane_indices = np.where(np.logical_and( np.abs(normal_projected_distance) <= norm_dist_limit,  angular_distances < angle_dist_limit ))[0]
                # prospect_plane_indices = np.where( np.abs(normal_projected_distance) < 0.3)[0]
                prospect_plane_indices = prospect_plane_indices[mask_in_pool[prospect_plane_indices]]
                prospect_plane_points = points[prospect_plane_indices]
                prospect_plane_normals = normals[prospect_plane_indices]


                if len(prospect_plane_indices) < 100:
                    print("Not enough points within prospect plane to make tree")
                    continue


                prospect_quadtree = SurfaceQuadtree(points=prospect_plane_points, normals=prospect_plane_normals, indices=prospect_plane_indices, min_density=0.5, max_points=200)
                
                self.surface_trees.append(prospect_quadtree)
        
                base_draw_probability[prospect_plane_indices] = 0.0

                    
                    