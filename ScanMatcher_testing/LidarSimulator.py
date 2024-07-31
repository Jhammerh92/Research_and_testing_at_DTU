from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.spatial import cKDTree
from scipy.optimize import minimize

from ScanMatcher2D import *



class LidarSimulator:
    FIGSIZE = (8,8)


    def __init__(self, room_size=(10, 10), num_rays=360, max_range=100):
        self.room_size = room_size
        self.num_rays = num_rays
        self.max_range = max_range


    def simulate_scan(self, robot_pose, ray_sigma=0.0):
        lidar_readings = []
        transformation_matrix = create_2d_transformation_matrix(*robot_pose)
        for angle in np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False):
            # ray_direction = np.array([np.cos(angle + robot_pose[2]), np.sin(angle + robot_pose[2])])
            ray_direction = np.array([np.cos(angle), np.sin(angle)])
            # ray_start = np.array([0,0]) #robot_pose[:2]
            ray_start = robot_pose[:2]
            ray_end = ray_start + ray_direction * self.max_range 


            intersection_point = self.cast_ray(ray_start, ray_end) #- robot_pose[:2]
            if isinstance(intersection_point, np.ndarray):
                intersection_point += ray_direction * np.random.normal(0.0,ray_sigma)
                lidar_readings.append(intersection_point)

        lidar_readings = np.array(lidar_readings)
        lidar_readings = transform_2d_points(lidar_readings, np.linalg.inv(transformation_matrix)) # transform to body frame
        # lidar_readings = transform_2d_points(lidar_readings, transformation_matrix) # transform to body frame
        return lidar_readings

    def cast_ray(self, start, end):
        # Intersect the ray with the walls of the room
        x1, y1, x2, y2 = 0, 0, self.room_size[0], 0  # Bottom wall
        intersection = self.intersect_segment(start, end, (x1, y1), (x2, y2))

        x1, y1, x2, y2 = self.room_size[0], 0, self.room_size[0], self.room_size[1]  # Right wall
        intersection = self.update_intersection(start, end, (x1, y1), (x2, y2), intersection)

        x1, y1, x2, y2 = self.room_size[0], self.room_size[1], 0, self.room_size[1]  # Top wall
        intersection = self.update_intersection(start, end, (x1, y1), (x2, y2), intersection)

        x1, y1, x2, y2 = 0, self.room_size[1], 0, 0  # Left wall
        intersection = self.update_intersection(start, end, (x1, y1), (x2, y2), intersection)

        return intersection

    def intersect_segment(self, p1, p2, q1, q2):
        # Find the intersection point of two line segments
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = q1
        x4, y4 = q2

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # Parallel lines

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return np.array([intersection_x, intersection_y])
        else:
            return None

    def update_intersection(self, start, end, wall_start, wall_end, intersection):
        new_intersection = self.intersect_segment(start, end, wall_start, wall_end)
        if new_intersection is not None:
            if intersection is None or np.linalg.norm(new_intersection - start) < np.linalg.norm(intersection - start):
                return new_intersection
        return intersection

    def visualize_scan(self, robot_pose):
        lidar_readings = self.simulate_scan(robot_pose)

        plt.figure(figsize=LidarSimulator.FIGSIZE)
        plt.plot([0, self.room_size[0], self.room_size[0], 0, 0],
                 [0, 0, self.room_size[1], self.room_size[1], 0], 'k-', linewidth=2)

        transformation_matrix =  create_2d_transformation_matrix(*robot_pose)

        # pose_projected_readings = lidar_readings
        pose_projected_readings = transform_2d_points(lidar_readings, transformation_matrix)

        plt.scatter(pose_projected_readings[:, 0], pose_projected_readings[:, 1], color='r', marker='o', label='Lidar Readings')
        plt.scatter(robot_pose[0], robot_pose[1], color='b', marker='x', label='Robot Pose')

        # Plot robot pose as an arrow
        arrow_length = 0.5
        plt.arrow(robot_pose[0], robot_pose[1],
                  arrow_length * np.cos(robot_pose[2]), arrow_length * np.sin(robot_pose[2]),
                  head_width=0.2, head_length=0.2, fc='b', ec='b', label='Robot Pose')
        
        plt.axis('equal')
        plt.title('2D Lidar Scan Simulation')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_lidar_readings(self, lidar_readings, labels=None):
        if not isinstance(lidar_readings, list):
            lidar_readings = [lidar_readings]

        if not isinstance(labels, list):
            labels = []
            for i in range(len(lidar_readings)):
                labels.append(f"Cloud {i}")

        fig, ax = plt.subplots(figsize=LidarSimulator.FIGSIZE)
        for i, lidar_reading in enumerate(lidar_readings):
            ax.scatter(lidar_reading[:, 0], lidar_reading[:, 1], marker='o', label=labels[i])
            
        plt.title('Lidar Readings')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')


if __name__ == "__main__":
    # Create a LidarSimulator
    lidar_simulator = LidarSimulator(room_size=(10, 7), num_rays=300, max_range=7)

    # Set the initial robot pose
    robot_pose_0 = np.array([2, 2, 0])  # [x, y, theta]
    robot_pose_1 = robot_pose_0 + np.array([0.7, 1.3, 0.2])  # [x, y, theta]

    # Simulate a lidar scan
    lidar_readings_0 = lidar_simulator.simulate_scan(robot_pose_0, 0.0)
    lidar_readings_1 = lidar_simulator.simulate_scan(robot_pose_1, 0.0)

    # Visualize the lidar scan
    # lidar_simulator.visualize_scan(robot_pose_0)
    # lidar_simulator.visualize_scan(robot_pose_1)

    # lidar_simulator.plot_lidar_readings([lidar_readings_0,lidar_readings_1 ])


    scan_matcher = ScanMatcher2D(ScanMatcher2D.BAYES)
    # scan_matcher.set_bayesian_prior([0.6, 1.2, 0.17], [1.5,1.5,1.5] )
    scan_matcher.set_bayesian_prior([0,0,0], [1,1,1])
    scan_matcher.set_source(lidar_readings_1)
    scan_matcher.set_target(lidar_readings_0)

    
    scan_matcher.calculate_normals(radius=1.0)
    scan_matcher.plot_source_target()
    plt.show()
    



    # scan_matcher.registrate()
    scan_matcher.vizualize_registrate()

    # scan_matcher.plot_normal_space()

    T = scan_matcher.get_transformation()
    transformed_lidar_readings_1 = transform_2d_points(lidar_readings_1, T)
    
    lidar_simulator.plot_lidar_readings([lidar_readings_0, lidar_readings_1, transformed_lidar_readings_1], ['Target', 'Source', 'T*Source'])

    scan_matcher.plot_losses()
    scan_matcher.plot_translation_epsilon()

    scan_matcher.plot_bayes_kde()
    scan_matcher.plot_bayes_translation_cov()

    # Display the plot
    plt.show()
