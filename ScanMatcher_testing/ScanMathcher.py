from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


from scipy.spatial import cKDTree
from scipy.optimize import minimize

def create_3d_transformation_matrix(x, y, z, roll, pitch, yaw):
    """
    Create a 3D transformation matrix from translation in x, y, z, and rotation (roll, pitch, yaw).

    Parameters:
    - x (float): Translation in the x-direction.
    - y (float): Translation in the y-direction.
    - z (float): Translation in the z-direction.
    - roll (float): Rotation angle around the x-axis (roll) in radians.
    - pitch (float): Rotation angle around the y-axis (pitch) in radians.
    - yaw (float): Rotation angle around the z-axis (yaw) in radians.

    Returns:
    - transformation_matrix (array): The 3D transformation matrix.
    """
    transformation_matrix = np.array([[1, 0, 0, x],
                                      [0, 1, 0, y],
                                      [0, 0, 1, z],
                                      [0, 0, 0, 1]])

    rotation_matrix = rotation_matrix_3d(roll,'x') @ (rotation_matrix_3d(pitch,'y') @ rotation_matrix_3d(yaw,'z'))

    transformation_matrix[:3,:3] =  rotation_matrix

    return transformation_matrix

def rotation_matrix_3d(angle, axis='z'):
    """
    Create a 3D rotation matrix for the specified axis and angle.

    Parameters:
    - axis (str): Axis of rotation ('x', 'y', or 'z').
    - angle (float): Rotation angle in radians.

    Returns:
    - rotation_matrix (array): The 3D rotation matrix.
    """
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle), -np.sin(angle)],
                                    [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                    [0, 1, 0],
                                    [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")

    return rotation_matrix

def transform_3d_points(points, transformation_matrix):
    """
    Transform 3D points using a 3D transformation matrix.

    Parameters:
    - points (array-like): 3D points in the form [[x1, y1, z1], [x2, y2, z2], ...].
    - transformation_matrix (array): The 3D transformation matrix.

    Returns:
    - transformed_points (array): Transformed 3D points.
    """
    points = np.array(points)
    homogenous_points = np.column_stack((points, np.ones(len(points))))

    transformed_homogenous = np.dot(transformation_matrix, homogenous_points.T).T

    transformed_points = transformed_homogenous[:, :3]

    return transformed_points

def angles_between_vectors(vectors1, vectors2):
    """
    Calculate the angles in radians between pairs of vectors.

    Parameters:
    - vectors1 (array-like): First set of vectors (each row represents a vector).
    - vectors2 (array-like): Second set of vectors (each row represents a vector).

    Returns:
    - angles (array): Array of angles in radians between corresponding pairs of vectors.
    """
    vectors1 = np.array(vectors1)
    vectors2 = np.array(vectors2)

    # Calculate dot products
    dot_products = np.sum(vectors1 * vectors2, axis=1)

    # Calculate magnitudes
    magnitudes1 = np.linalg.norm(vectors1, axis=1)
    magnitudes2 = np.linalg.norm(vectors2, axis=1)

    # Calculate cosine of the angles
    cos_thetas = dot_products / (magnitudes1 * magnitudes2)

    # Ensure cos_thetas are within the valid range [-1, 1]
    cos_thetas = np.clip(cos_thetas, -1.0, 1.0)

    # Calculate the angles in radians
    angles = np.arccos(cos_thetas)

    return angles


class AdamScheduler:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0      # Time step
        self.last_step_size = lr
        
    def step(self, gradient):
        self.t += 1
        
        # Initialize first and second moment estimates if not yet initialized
        if self.m is None:
            self.m = np.zeros_like(gradient)
        if self.v is None:
            self.v = np.zeros_like(gradient)
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        # Compute bias-corrected first and second moment estimates
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        step_size = self.lr / (np.sqrt(v_hat) + self.epsilon) * m_hat
        # print(step_size)
        self.last_step_size = - step_size

        parameters = -step_size 
        
        return parameters
    
    def get_last_step_size(self):
        return self.last_step_size



class ScanMatcher:
    ITERATIONS = "iterations"
    TRANSLATION_EPSILON = "translation_epsilon"
    P2P = 'P2P'
    P2PL = 'P2Pl'
    N2N = 'N2N'
    BAYES = 'BAYES'
    SGD = 'SGD'

    FIGSIZE = (8,8)

    def __init__(self, method=P2P, distance_threshold=np.float64('inf'), break_conditions=None) -> None:
        self.distance_threshold = distance_threshold
        self.source_transformed = None
        self.final_transformation_matrix = np.eye(4)
        self.transformation_matrix = np.eye(4)
        # self.transformation_matrix[:2,2] = np.float64('inf')
        self.set_conditions(break_conditions)
        self.did_converge = False
        self.iter = 0
        self.losses = []
        self.translation_epsilons = []

        self.N = None

        self.batch_size = 150

        self.linear_adam_scheduler = AdamScheduler(lr=0.1, beta1=0.6, beta2=0.99999999999)
        self.rotional_adam_scheduler = AdamScheduler(lr=0.1, beta1=0.3, beta2=0.9999999999)

        self.bayes_step_size_t = 0.0001
        self.bayes_step_size_r = 0.00005
        self.bayes_epoch_decay = 0.8

        # self.not_picked_indices_initial = set(range(self.source.shape[0]))
        
        self.rescale_points_a = None
        self.rescale_points_b = None

        self.method = method

        self.X_log = []
        self.X = np.zeros(6)
        self.X_std = np.ones(6) * 2
        self.X_mean = np.zeros(6)

        self.V = 0
        self.beta = 0.9
        self.lmbda = 1e-8

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


    def set_conditions(self, break_conditions):
        # set default conditions
        self.conditions = {ScanMatcher.ITERATIONS: np.int64(500),
                           ScanMatcher.TRANSLATION_EPSILON: 1e-8,
                           }
        if not break_conditions is None:
            self.conditions.update(break_conditions)

    def set_bayesian_prior(self, means, stds):
        self.prior = np.array(means)
        self.prior_std = np.array(stds)
        self.prior_covariance_matrix_t = np.diag(np.array(stds[:3])**2)
        self.prior_covariance_matrix_r = np.diag(np.array(stds[3:])**2)
        # self.prior = np.array(means)
        self.X = np.array(means)
        self.X_std = np.array(stds) 
        self.X_mean = np.array(means) 


    def find_point_correspondences(self, source_indicies=None, N=None):
        """
        source_indices: if we want to find pairs to specific indices in sources   

        N: if want to find pick 'N' random indecies to use / find point pairs for 
        """

        self.not_picked_indices

        if not isinstance(source_indicies, np.ndarray):
            source_indicies = np.arange(self.source.shape[0]) # all points are used
        if not (N is None):
            # source_indicies = np.random.randint(0, self.source.shape[0], N) # N random points are picked from the source
            n = len(self.not_picked_indices)
            if n >= N:
                source_indicies = np.random.choice(list(self.not_picked_indices), N, replace=False) # N random points are picked from the source
                self.not_picked_indices -= set(source_indicies)
            else: 
                source_indicies = self.not_picked_indices 
                self.not_picked_indices = self.not_picked_indices_initial - self.not_picked_indices
                remainder_indices = np.random.choice(list(self.not_picked_indices), N - n, replace=False)
                source_indicies = np.asarray(list(source_indicies | set(remainder_indices)))


        query_points = self.source_transformed[source_indicies,:]

        dd ,target_indicies = self.target_kd_tre.query(query_points, k=1, distance_upper_bound=self.distance_threshold)

        distance_mask = np.where(dd != np.float64('inf'))[0] # remove points that are beyond the threshold criteria

        target_indicies = target_indicies[distance_mask]
        source_indicies = source_indicies[distance_mask]

        self.point_pairs_idx = np.c_[source_indicies, target_indicies] # create point pairs array

        return
    
    def update_final_transformation(self):
        # TODO make case swith for method
        # self.final_transformation_matrix = self._find_rigid_transform_svd(self.source, self.target)
        # if self.method == ScanMatcher2D.P2P:
        #     self.final_transformation_matrix = self._find_rigid_transform_p2p(self.source, self.target)
        # elif self.method == ScanMatcher2D.P2PL:
        #     self.final_transformation_matrix = self._find_rigid_transform_p2pl(self.source, self.target, self.source_normals, self.target_normals)
        # elif self.method == ScanMatcher2D.N2N:
        #     self.final_transformation_matrix = self._find_rigid_transform_n2n(self.source, self.target, self.source_normals, self.target_normals)

        # if self.method == ScanMatcher2D.SGD:
        #     self.final_transformation_matrix = self.transformation_matrix
        # else:
        self.final_transformation_matrix = self.transformation_matrix @ self.final_transformation_matrix


    def solve_rigid_transform(self):
    #    self.transformation_matrix = self._find_rigid_transform_svd(self.source_transformed, self.target)
        if self.method == ScanMatcher.P2P:
            self.transformation_matrix = self._find_rigid_transform_p2p(self.source_transformed, self.target)
        elif self.method == ScanMatcher.P2PL:
            #TODO if no normals self.calculate_normals()
            self.transformation_matrix = self._find_rigid_transform_p2pl(self.source_transformed, self.target, self.source_normals_transformed, self.target_normals)
        elif self.method == ScanMatcher.N2N:
            self.transformation_matrix = self._find_rigid_transform_n2n(self.source_transformed, self.target, self.source_normals_transformed, self.target_normals)
        elif self.method == ScanMatcher.SGD:
            # self.transformation_matrix = self._find_rigid_transform_sgd(self.source_normalized, self.target_normalized, self.source_normals_transformed, self.target_normals)
            self.transformation_matrix = self._find_rigid_transform_sgd(self.source, self.target)
        elif self.method == ScanMatcher.BAYES:
            # self.transformation_matrix = self._find_rigid_transform_bayes(self.source, self.target, self.target_normals)
            self.transformation_matrix = self._find_rigid_transform_bayes(self.source, self.target)


    def _find_rigid_transform_svd(self, source=None, target=None):
        """
        Find the 2D rigid body transformation (translation, rotation) between two sets of corresponding points.

        Parameters:
        - source_points (array-like): Source points in the form [[x1, y1], [x2, y2], ...].
        - target_points (array-like): Target points in the same form as source_points.

        Returns:
        - transformation_matrix (array): The 2D rigid body transformation matrix.
        """
        source_points = np.array(source[self.point_pairs_idx[:,0], :])
        target_points = np.array(target[self.point_pairs_idx[:,1], :])

        # Centroid of source and target points
        centroid_source = np.mean(source_points, axis=0)
        centroid_target = np.mean(target_points, axis=0)

        # Center the points
        centered_source = source_points - centroid_source
        centered_target = target_points - centroid_target

        # Singular Value Decomposition
        matrix_w = np.dot(centered_source.T, centered_target)
        # matrix_w = np.dot(centered_target.T, centered_source)
        U, _, Vt = np.linalg.svd(matrix_w)

        # Ensure proper rotation (no reflection)
        rotation_matrix = np.dot(U, Vt)
        if np.linalg.det(rotation_matrix) < 0:
            Vt[-1, :] *= -1
            rotation_matrix = np.dot(U, Vt)

        # Translation vector
        translation_vector = centroid_target - np.dot(rotation_matrix, centroid_source)

        # Construct the transformation matrix
        transformation_matrix = np.eye(3)
        transformation_matrix[:2, :2] = rotation_matrix
        transformation_matrix[:2, 2] = translation_vector

        return transformation_matrix
    
    def _find_rigid_transform_p2p(self, source=None, target=None):
        def P2P_LOSS(x):
            transformation  = create_3d_transformation_matrix(*x)
            s = transform_3d_points(source[self.point_pairs_idx[:,0]], transformation)
            r = target[[self.point_pairs_idx[:,1]]]
            N = s.shape[0]
            loss = np.sum(np.square(np.linalg.norm(s-r, axis=1))) / N
            return loss
        
        result = self.minimize_loss(P2P_LOSS)
        x = result.x
        transformation = create_3d_transformation_matrix(*x)
        
        return transformation
    
    def p2pl_loss(self, s, r, r_n):
        m = s.shape[0]
        d = s-r
        d_n = np.einsum('ij,ij->i', d, r_n)
        # loss = np.mean(np.square(d_n))
        loss = np.square(d_n) / m
        return loss

    def _find_rigid_transform_p2pl(self, source, target, source_normals, target_normals):
        def P2PL_LOSS(x):
            transformation  = create_3d_transformation_matrix(*x)
            s = transform_3d_points(source[self.point_pairs_idx[:,0]], transformation)
            r = target[self.point_pairs_idx[:,1]]
            d = s-r
            d_n = np.einsum('ij,ij->i', d, target_normals[self.point_pairs_idx[:,1]])
            N = s.shape[0]
            loss = np.sum(np.square(d_n)) / N
            return loss
        
        result = self.minimize_loss(P2PL_LOSS)
        x = result.x
        transformation = create_3d_transformation_matrix(*x)
        
        return transformation
    
    
    def _find_rigid_transform_n2n(self, source, target, source_normals, target_normals):
        def N2N_LOSS(x):
            transformation  = create_3d_transformation_matrix(*x)
            s = transform_3d_points(source[self.point_pairs_idx[:,0]], transformation)
            r = target[self.point_pairs_idx[:,1]]
            d = np.linalg.norm(s-r, axis=1)
            # d_p2p = np.einsum('ij,ij->i', d, target_normals)

            s_n = transform_3d_points(source_normals[self.point_pairs_idx[:,0]], transformation)
            r_n = target_normals[self.point_pairs_idx[:,1]]
            d_n = angles_between_vectors(s_n, r_n)
            N = s_n.shape[0]
            # loss = np.sum(np.square(d_n) + np.square(d)) / N
            loss = np.sum(np.square(d_n)) / N
            return loss
        
        result = self.minimize_loss(N2N_LOSS)
        x = result.x
        transformation = create_3d_transformation_matrix(*x)
        
        return transformation
    
    def linear_gradient(self):
        return np.array([1,1,1])

    def simple_rotational_gradient(self, angle, axis='yaw'):

    
        if axis == 'roll':
            gradient_rot_mat_roll = np.array([ [1, 0, 0, 0],
                                                [0, -np.sin(angle), -np.cos(angle), 0],
                                                [0, np.cos(angle), -np.sin(angle), 0],
                                                [0, 0, 0, 1]])
            return gradient_rot_mat_roll
        if axis == 'pitch':
            gradient_rot_mat_pitch = np.array([[-np.sin(angle), 0, np.cos(angle), 0],
                                                [0, 1, 0, 0],
                                                [-np.cos(angle), 0, -np.sin(angle), 0],
                                                [0, 0, 0, 1]])
            return gradient_rot_mat_pitch
        if axis == 'yaw':
            gradient_rot_mat_yaw = np.array([ [-np.sin(angle), -np.cos(angle), 0, 0],
                                                [np.cos(angle),  -np.sin(angle), 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]])
            return gradient_rot_mat_yaw
        else:
            raise ValueError("Invalid grad_type. Use 'all', 'roll', 'pitch', or 'yaw'.")
        
    def rotational_gradient(self, g,b,a , axis='yaw'):
        """
        g,b,a is gamma, beta , alpha from https://web.mit.edu/2.05/www/Handout/HO2.PDF
        the combined matrix has been differentiated with respect to each parameter
        """
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cg, sg = np.cos(g), np.sin(g)


        if axis == 'roll': # gamma
            gradient_rot_mat_roll = np.array([ [0, ca*sb*cg + sa*sg, sa*cg - ca*sb*sg, 0],
                                                [0, sa*sb*cg - ca*sg, - ca*cg - sa*sb*sg, 0],
                                                [0, cb*cg, -cb*sg, 0],
                                                [0, 0, 0, 1]])
            return gradient_rot_mat_roll
        if axis == 'pitch': # beta
            gradient_rot_mat_pitch = np.array([[-sb*ca, ca*cb*sg, ca*cb*cg, 0],
                                                [-sa*sb, sa*cb*sg, sa*cb*cg, 0],
                                                [-cb, 0, 0, 0],
                                                [0, 0, 0, 1]])
            return gradient_rot_mat_pitch
        if axis == 'yaw': # alpha
            gradient_rot_mat_yaw = np.array([[-sa*cb, - sa*sb*sg - ca*cg, ca*sg - sa*sb*cg, 0],
                                             [ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg, 0],
                                             [0, 0, 0, 0],
                                             [0, 0, 0, 1]])
            return gradient_rot_mat_yaw
        else:
            raise ValueError("Invalid axis. Use 'roll', 'pitch', or 'yaw'.")


    def _find_rigid_transform_sgd(self, source, target):

        x = self.X
        m = self.point_pairs_idx.shape[0]
        
        # transformation  = create_2d_transformation_matrix(x[0]/self.rescale_points_a[0], x[1]/self.rescale_points_a[1], x[2])
        transformation  = create_3d_transformation_matrix(*x)

        # s = source[self.point_pairs_idx[:,0]]
        s = source[self.point_pairs_idx[:,0]]
        s_prime = transform_3d_points(s, transformation)
        r = target[self.point_pairs_idx[:,1]]


        batch_loss = (s_prime - r) 



        translation_mean_gradient = np.sum(batch_loss, axis=0) * self.linear_gradient()
        # rotational_mean_gradient =  np.sum(np.multiply( (s_prime - r)  , transform_3d_points(s, self.rotational_gradient(*x[3:]))  ), axis=0) / m
        rotational_mean_gradient_r = np.sum(np.sum(np.multiply(batch_loss , transform_3d_points(s, self.rotational_gradient(*x[3:], 'roll'))), axis=1 ), axis=0)
        rotational_mean_gradient_p = np.sum(np.sum(np.multiply(batch_loss , transform_3d_points(s, self.rotational_gradient(*x[3:], 'pitch'))), axis=1 ), axis=0)
        rotational_mean_gradient_y = np.sum(np.sum(np.multiply(batch_loss , transform_3d_points(s, self.rotational_gradient(*x[3:], 'yaw'))), axis=1 ), axis=0)
        rotational_mean_gradient = np.r_[rotational_mean_gradient_r, rotational_mean_gradient_p, rotational_mean_gradient_y]

        if True: # use ADAM step size scheduling
            linear_update_adam  = self.linear_adam_scheduler.step( 1/2 * translation_mean_gradient/ m )
            rotational_update_adam = self.rotional_adam_scheduler.step(1/2 * rotational_mean_gradient/ m )
            new_x_t = x[:3] + linear_update_adam
            new_x_r = x[3:] + rotational_update_adam 
        else:
            new_x_t = x[:2] - 0.1 * 1/(2*m) * np.sum(s_prime - r, axis=0) * linear_gradient()
            new_x_r = x[2:] - 0.03 * 1/(2*m) * np.sum(np.sum(np.multiply(s_prime - r , transform_2d_points(s, rotational_gradient(x[2]))),axis=1  ), axis=0)


        self.X = np.r_[new_x_t, new_x_r]
        # print(self.X)

        new_transformation = create_3d_transformation_matrix(*self.X)


        loss = np.sum(np.square(np.linalg.norm(transform_3d_points(s, new_transformation) - r, axis=1))) / m
        self.losses.append(loss)

        # transformation_delta = new_transformation.T @ transformation
        # transformation_delta = np.linalg.inv(new_transformation) @ transformation
        transformation_delta =  new_transformation @ np.linalg.inv(transformation)

        # print(self.X)

        return transformation_delta
        # return new_transformation

    # def _find_rigid_transform_bayes(self, source, target, target_normals):
    def _find_rigid_transform_bayes(self, source, target):
        x = self.X
        mu = self.X_mean
        sigma = self.X_std
        m = self.point_pairs_idx.shape[0]

        transformation  = create_3d_transformation_matrix(*x)

        s = source[self.point_pairs_idx[:,0]]
        s_prime = transform_3d_points(s, transformation)
        r = target[self.point_pairs_idx[:,1]]
        # r_n = target_normals[self.point_pairs_idx[:,1]]

        batch_loss = (s_prime - r) / m    # p2p loss
        # batch_loss = r_n * self.p2pl_loss(s, r, r_n)[:, np.newaxis] # p2pl loss


        # compute mean gradients
        translation_mean_gradient = np.sum(batch_loss, axis=0) * self.linear_gradient()
        # rotational_mean_gradient =  np.sum( np.multiply( (s_prime - r)  , transform_3d_points(s, self.rotational_gradient(*x[3:]))  ), axis=0) / m
        rotational_mean_gradient_r = np.sum(np.sum(np.multiply(batch_loss , transform_3d_points(s, self.rotational_gradient(*x[3:], 'roll'))), axis=1 ), axis=0)
        rotational_mean_gradient_p = np.sum(np.sum(np.multiply(batch_loss , transform_3d_points(s, self.rotational_gradient(*x[3:], 'pitch'))), axis=1 ), axis=0)
        rotational_mean_gradient_y = np.sum(np.sum(np.multiply(batch_loss , transform_3d_points(s, self.rotational_gradient(*x[3:], 'yaw'))), axis=1 ), axis=0)
        rotational_mean_gradient = np.r_[rotational_mean_gradient_r,rotational_mean_gradient_p,rotational_mean_gradient_y]

        mean_gradients = np.r_[translation_mean_gradient, rotational_mean_gradient]

        # compute prior negative log gradient
        negative_log_prior_gradient_t = (x[:3] - mu[:3]) / sigma[:3]
        negative_log_prior_gradient_r = np.sin(x[3:] - mu[3:]) / sigma[3:]

        # N = self.N
        N = 6000
        if False: # use ADAM step size scheduling
            linear_update_adam = self.linear_adam_scheduler.step( (negative_log_prior_gradient_t + N * translation_mean_gradient)  )
            rotational_update_adam = self.rotional_adam_scheduler.step( (negative_log_prior_gradient_r + N * rotational_mean_gradient) )
            step_t = self.linear_adam_scheduler.get_last_step_size()
            step_r = self.rotional_adam_scheduler.get_last_step_size()
            injected_noise = np.r_[np.array([np.random.normal(0, np.abs(step)) for i,step in enumerate(step_t) ]), np.array([np.random.normal(0, np.abs(step)) for i,step in enumerate(step_r) ]) ]
            new_x_t = x[:3] +  linear_update_adam + injected_noise[:3]
            new_x_r = x[3:] +  rotational_update_adam + injected_noise[3:]
        else:
            # compute preconditioners 
            self.V = self.beta * self.V + (1-self.beta) * mean_gradients**2
            A = ( 1 /(self.lmbda + np.sqrt(self.V)))

            epoch = (self.batch_size * self.iter) / self.N
            # print(epoch//1)
            print(f"Epoch: {epoch:.2f}")

            step_t = self.bayes_step_size_t * self.bayes_epoch_decay**(epoch) + self.bayes_step_size_t * 0.05
            step_r = self.bayes_step_size_r * self.bayes_epoch_decay**(epoch) + self.bayes_step_size_r * 0.05
     
            injected_noise = np.r_[np.array([np.random.normal(0, np.abs(step_t*A[i])) for i in range(3) ]), np.array([np.random.normal(0, np.abs(step_r*A[i+3])) for i in range(3) ]) ]
            new_x_t = x[:3] - step_t/2 * A[:3] * (negative_log_prior_gradient_t + N * translation_mean_gradient) + injected_noise[:3]
            new_x_r = x[3:] - step_r/2 * A[3:] * (negative_log_prior_gradient_r + N * rotational_mean_gradient) + injected_noise[3:]


        self.X = np.r_[new_x_t, new_x_r]
        self.X_log.append(self.X.copy())

        # mean_X = np.mean(self.X_log, axis=0)
        # print(mean_X)

        # new_transformation = create_2d_transformation_matrix(*mean_X)
        new_transformation = create_3d_transformation_matrix(*self.X)


        loss = np.sum(np.square(np.linalg.norm(transform_3d_points(s, new_transformation) - r, axis=1))) / m
        self.losses.append(loss)

        # transformation_delta = new_transformation.T @ transformation
        # transformation_delta = np.linalg.inv(new_transformation) @ transformation
        transformation_delta =  new_transformation @ np.linalg.inv(transformation)

        # print(self.X)

        return transformation_delta
        # return new_transformation

    def minimize_loss(self, loss_function):
        initial_guess = np.array([0,0,0,0,0,0]) # x, y, yaw
        result = minimize(loss_function, initial_guess, method='L-BFGS-B') # scipy minimizer
        self.losses.append(result.fun)
        return result

    def transform_source(self):
        # self.source_transformed = transform_2d_points(self.source_transformed, self.transformation_matrix)
        self.source_transformed = transform_3d_points(self.source, self.final_transformation_matrix)
        try:
            normal_transform = self.final_transformation_matrix.copy()
            normal_transform[:2, 2] = 0.0
            self.source_normals_transformed = transform_3d_points(self.source_normals_transformed, normal_transform)
        except:
            pass
            # print('')

    def rescale_points(self, points, inverse=False):
        if inverse == False:
            self.rescale_points_a =  (np.max(points, axis=0) - np.min(points, axis=0))
            self.rescale_points_b =  np.min(points, axis=0)
            scaled_points = (points - self.rescale_points_b) / self.rescale_points_a
            return scaled_points
        elif inverse == True:
            rescaled_points = points * self.rescale_points_a #+ self.rescale_points_b
            return rescaled_points
        elif inverse == None:
            scaled_points = (points - self.rescale_points_b) / self.rescale_points_a 
            return scaled_points

    def set_source(self, source):
        self.source = np.array(source)

        self.final_transformation_matrix = create_3d_transformation_matrix(*self.X)
        # self.source_transformed = np.array(source)
        self.transform_source()
        self.source_kd_tre = cKDTree(self.source)

        # self.source_normalized = (source - np.min(source, axis=0)) / (np.max(source, axis=0) - np.min(source, axis=0))
        self.source_normalized = self.rescale_points(source)

        self.N = self.source.shape[0]

        self.not_picked_indices_initial = set(range(self.N))
        self.not_picked_indices = set(range(self.N))

    def set_target(self, target):
        self.target = np.array(target)
        self.target_kd_tre = cKDTree(self.target)

        # self.target_normalized = (target - np.min(target, axis=0)) / (np.max(target, axis=0) - np.min(target, axis=0))
        self.target_normalized = self.rescale_points(target, None)

    def set_initial_guess(self, initial_guess=None):
        if initial_guess == None:
            self.initial_guess = np.array([0,0,0])
        self.initial_guess

    def get_transformation(self):
        if self.method == ScanMatcher.BAYES:
            mean_X = np.mean(self.X_log, axis=0)
            self.final_transformation_matrix = create_3d_transformation_matrix(*mean_X)
        return self.final_transformation_matrix


    def assert_break_conditions(self):
        if self.method == ScanMatcher.BAYES:
            return False

        translation_epsilon = np.linalg.norm(self.transformation_matrix[:2,2])
        # print(f": {translation_epsilon}")

        if self.iter == 0:
            self.did_converge = False
            return
        
        self.translation_epsilons.append(translation_epsilon)

        if translation_epsilon < self.conditions.get(ScanMatcher.TRANSLATION_EPSILON):
            self.did_converge = True

        return self.did_converge

    def run_iterator(self):
        for _ in range(self.conditions.get(ScanMatcher.ITERATIONS)):
            self.run_step()
            yield

    def run_step(self):
        if self.method == ScanMatcher.SGD or self.method == ScanMatcher.BAYES:
            self.find_point_correspondences(None, N=self.batch_size)
        else:
           self.find_point_correspondences(None)
        self.solve_rigid_transform()
        self.update_final_transformation()
        self.transform_source()
        self.iter += 1

    def registrate(self):
        self.did_converge = False
        for _ in self.run_iterator():
            if self.assert_break_conditions():
                break
        
        self.update_final_transformation()
        self.print_convrergence_info()
        

    def vizualize_registrate(self):
        # Create a figure and axis

        # Create the animation
        def update(frame):
            self.did_converge = False
            if self.assert_break_conditions():
                animation.event_source.stop() 
                return
            self.run_step()
            # print(self.final_transformation_matrix)
            # ax.clear()  # Clear the previous plot
            # ax.scatter(self.target[:, 0], self.target[:, 1], marker='o', label='Target', s=10, alpha=0.3)
            # ax.scatter(self.source[:, 0], self.source[:, 1], marker='o', label='Source', s=5, alpha=0.3)
            # tar.set_offsets(self.target[:, 0], self.target[:, 1])
            # sou.set_offsets(self.source[:, 0], self.source[:, 1])
            sout_xy.set_offsets(self.source_transformed[:, :2].copy())
            sout_zy.set_offsets(self.source_transformed[:, 3:0:-1].copy())
            # ax.scatter(self.source_transformed[:, 0].copy(), self.source_transformed[:, 1].copy(), marker='o', label='Source Transformed', s=8, alpha=0.3)
            fig.suptitle(f'Scan Matching - Iteration {self.iter}')
            # ax.set_title(f'Scan Matching - Iteration {self.iter}')

            
        fig, [ax_xy, ax_zy] = plt.subplots(1,2,figsize =(12,8), layout='constrained', sharey=True)
        tar_xy = ax_xy.scatter(self.target[:, 0], self.target[:, 1], marker='o', label='Target', s=10, alpha=0.3)
        sou_xy = ax_xy.scatter(self.source[:, 0], self.source[:, 1], marker='o', label='Source', s=5, alpha=0.3)
        sout_xy = ax_xy.scatter(self.source_transformed[:, 0].copy(), self.source_transformed[:, 1].copy(), marker='o', label='Source Transformed', s=8, alpha=0.3)
        tar_xz = ax_zy.scatter(self.target[:, 2], self.target[:, 1], marker='o', label='Target', s=10, alpha=0.3)
        sou_xz = ax_zy.scatter(self.source[:, 2], self.source[:, 1], marker='o', label='Source', s=5, alpha=0.3)
        sout_zy = ax_zy.scatter(self.source_transformed[:, 2].copy(), self.source_transformed[:, 1].copy(), marker='o', label='Source Transformed', s=8, alpha=0.3)
        
        fig.suptitle(f'Scan Matching - Iteration {self.iter}')
        ax_xy.set_xlabel('X [m]')
        ax_xy.set_ylabel('Y [m]')
        ax_xy.axis('equal')
        ax_zy.set_xlabel('Z [m]')
        ax_zy.set_ylabel('Y [m]')
        ax_zy.axis('equal')
        ax_xy.legend()
        ax_zy.legend()

        animation = FuncAnimation(fig, update, frames=self.conditions[ScanMatcher.ITERATIONS], interval=1, repeat=False)
        # Display the plot
        plt.show()
        self.update_final_transformation()
        self.print_convrergence_info()


    def print_convrergence_info(self):
        print(f"Convergence: {self.did_converge}")
        print(f"Iterations: {self.iter}")
        print(f"Final loss ({self.method}): {self.losses[-1]}")
        print(f"Final transformation:\n", self.get_transformation())

    def plot_losses(self):
        fig, ax = plt.subplots(figsize=ScanMatcher.FIGSIZE)
        ax.set_yscale('log')
        ax.plot(self.losses)

    def plot_translation_epsilon(self):
        fig, ax = plt.subplots(figsize=ScanMatcher.FIGSIZE)
        ax.set_yscale('log')
        ax.plot(self.translation_epsilons)


    def calculate_eigen_normal(self, points):
        """
        Calculate normals for a set of 2D points using eigen decomposition.

        Parameters:
        - points (array-like): 2D points in the form [[x1, y1], [x2, y2], ...].

        Returns:
        - normals (array): Normal vectors corresponding to each point.
        """
        points = np.array(points)
        centroid = np.mean(points, axis=0)

        # Center the points
        centered_points = points - centroid

        # Compute covariance matrix
        covariance_matrix = np.cov(centered_points, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Get the eigenvector corresponding to the smallest eigenvalue as the normal
        normal = eigenvectors[:, np.argmin(eigenvalues)]

        # Flip the normal if it points towards the centroid
        if np.dot(normal, centroid) > 0:
            normal *= -1

        return normal


    def calculate_normals(self,radius=3.0, update_target_normals=False):
        # source normals
        ii = self.source_kd_tre.query_ball_point(self.source, radius)
        source_normals_list = []
        groups = [self.source[i] for i in ii]
        for group in groups:
            source_normals_list.append(self.calculate_eigen_normal(group))

        self.source_normals = np.array(source_normals_list)
        self.source_normals_transformed = np.array(source_normals_list)

        # target normals
        ii = self.target_kd_tre.query_ball_point(self.target, radius)
        target_normals_list = []
        groups = [self.target[i] for i in ii]
        for group in groups:
            target_normals_list.append(self.calculate_eigen_normal(group))

        self.target_normals = np.array(target_normals_list)


        # dd, ii = self.source_kd_tre.query_ball_point(self.target, 0.5)
    
    def _plot_source(self, ax):
        ax.scatter(self.source[:, 0], self.source[:, 1], marker='o', label='Source', color='C0')
        try:
            self._plot_normals(ax, self.source, self.source_normals, label='Source Normals', angles='xy', color='C0')
        except:
            print('no normals on source')

    def _plot_source_final(self, ax):
        self.transform_source()
        ax.scatter(self.source_transformed[:, 0], self.source_transformed[:, 1], marker='o', label='Source', color='C0')
        try:
            self._plot_normals(ax, self.source, self.source_normals, label='Source Normals', angles='xy', color='C0')
        except:
            print('no normals on source')
        
    def _plot_target(self, ax):
        ax.scatter(self.target[:, 0], self.target[:, 1], marker='o', label='Target', color='C1')
        try:
            self._plot_normals(ax, self.target, self.target_normals, label='Target Normals', angles='xy', color='C1')
        except:
            print('no normals on target')
        
    def _plot_normals(self, ax, points, normals, **kwargs):
        try:
            ax.quiver(points[:, 0], points[:, 1], normals[:, 0], normals[:, 1],headwidth=2, headlength=2.0,headaxislength=2.0, width = 5 ,units='dots', **kwargs)
        except:
            print('Normals given does not exist')

    def plot_source(self):
        fig, ax = plt.subplots(figsize=(10,8))
        self._plot_source(ax)
            
        plt.title('Source')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

    def plot_target(self):
        fig, ax = plt.subplots(figsize=(10,8))
        self._plot_target(ax)
            
        plt.title('Target')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

    def plot_source_target(self):
        fig, ax = plt.subplots(figsize=(10,8))
        self._plot_target(ax)
        self._plot_source(ax)
            
        plt.title('Source and Target')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

    def plot_source_target_final(self):
        fig, ax = plt.subplots(figsize=(10,8))
        self.transform_source()
        self._plot_target(ax)
        self._plot_source_final(ax)
            
        plt.title('Source and Target')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

    def plot_normal_space(self):
        fig, ax = plt.subplots(figsize=(10,8))
            
        noise_x = np.random.normal(0.0,0.02,self.source_normals.shape[0])
        noise_y = np.random.normal(0.0,0.02,self.source_normals.shape[0])

        ax.scatter(self.target_normals[:,0] + noise_x, self.target_normals[:,1] + noise_y, s=5, label='Target')
        ax.scatter(self.source_normals[:,0] + noise_x, self.source_normals[:,1] + noise_y, s=5, label='Source')
        ax.scatter(self.source_normals_transformed[:,0] + noise_x, self.source_normals_transformed[:,1] + noise_y, s=5, label='Source')

        plt.title('Normal Space')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')


    def plot_bayes_kde(self):

        if not self.method is ScanMatcher.BAYES:
            return
        # Gaussian function to fit to KDE
        def gaussian(x, mean, std):
            return 1/(std * np.sqrt(2 *np.pi)) * np.exp(-0.5 * ((x - mean) / std)**2)

        burn_in = 40
        data = np.asarray(self.X_log)

        # Get the number of columns
        num_columns = data.shape[1]

        vars = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

        # Create subplots for each column
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 8), layout='constrained')
        axs = axs.flatten()
        # Iterate over each column
        self.posterior = []
        self.posterior_std = []
        self.posterior_covariance_matrix_t = np.eye(3)
        self.posterior_covariance_matrix_r = np.eye(3)
        for i, var in enumerate(vars):
            # Extract data for the current column
            column_data = data[burn_in:, i]
            
            # Create KDE estimator for the current column
            kde = gaussian_kde(column_data)
            
 
            x = np.linspace(min(column_data), max(column_data), 100)
            kde_values = kde(x)

            max_index = np.argmax(kde_values)
            peak_location = x[max_index]

            # Fit Gaussian curve to KDE
            popt, pcov = curve_fit(gaussian, x, kde_values, p0=[peak_location, 0.1])  # Initial guess for parameters
            
            # Get standard deviation from fitted curve
            std = popt[1]
            mean = popt[0]
            self.posterior.append(mean) 
            self.posterior_std.append(std) 
            
            # Plot KDE for the current column
            axs[i].plot(x, kde_values, label='KDE')
            axs[i].plot(peak_location, kde_values[max_index], 'ro', label=f'Peak at {peak_location:.2f}')  # Plot peak
            axs[i].plot(x, gaussian(x, *popt), 'r--', label=f'Gaussian Fit, $\mu$={mean:.2f}, $\sigma$={std:.2f}')  # Plot fitted curve
            axs[i].set_title(f'KDE {var}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Density')
            axs[i].legend()

        self.posterior = np.array(self.posterior)
        self.posterior_std = np.array(self.posterior_std)
        self.posterior_covariance_matrix_t = np.diag(self.posterior_std[:3]**2)
        self.posterior_covariance_matrix_r = np.diag(self.posterior_std[3:]**2)



    def plot_bayes_translation_cov(self):
        if not self.method is ScanMatcher.BAYES:
            return
        
        def multivariate_gaussian(x, mu1, mu2, mu3, s1, s2, s3, s4, s5, s6):
        # def multivariate_gaussian(x, mu1, mu2, mu3, s1, s2, s3):
                """
                sigma_row consist of the 6 elements of the covaraince matrix, the diagonal and the upper off elements
                """
                sigma_row = np.array([s1, s2, s3, s4, s5, s6])
                # sigma_row = np.array([s1, s2, s3])
                sigma = np.diag(sigma_row[:3])
                upper_triangular = np.triu(-sigma_row[3:],)
                lower_triangular = np.tril(sigma_row[3:])
                sigma = sigma + upper_triangular + lower_triangular
                mu = np.array([[mu1, mu2, mu3]]).T
                return 1/(np.sqrt(2 *np.pi)**3 *np.linalg.det(sigma)) * np.exp(-0.5 * np.einsum('ij,ji->i', (x - mu).T, np.linalg.inv(sigma) @ (x - mu))) 
                # return 1/(np.sqrt(2 *np.pi)**3 *np.linalg.det(sigma)) * np.exp(-0.5 * ((x - mu).T @ np.linalg.inv(sigma) @ (x - mu)))
        
        # xy data
        data = np.asarray(self.X_log)
        # kde3 = gaussian_kde(data[:,:3].T)
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]
        # x_3, y_3, z_3 = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(y), max(y), 10), np.linspace(min(z), max(z), 10))
        # positions_3 = np.vstack([x_3.ravel(), y_3.ravel(), z_3.ravel()])
        # kde_values = kde3(positions_3).T
        # popt, pcov = curve_fit(multivariate_gaussian, positions_3, kde_values, p0=[*self.posterior[:3],*[0.1,0.1,0.1,0.01,0.01,0.01]], maxfev=50000)



        covariance_matrix_t = self.posterior_covariance_matrix_t
        # covariance_matrix_r = self.prior_covariance_matrix_r
        # x = data[:,0]
        # y = data[:,1]
        # z = data[:,2]
        xy = data[:,:2]
        xy_mean = self.posterior[:2]
        covariance_matrix_xy = covariance_matrix_t[:2,:2]

        # Transpose the data to fit the expected shape
        # xy = xy.T

        # Create 2D KDE
        kde = gaussian_kde(xy.T)

        # Compute covariance matrix

        # Compute standard deviations
        std_devs = np.sqrt(np.diag(covariance_matrix_xy))
        print(f"Translation standard deviation: {std_devs}")

        # Define a grid of points for evaluation
        x_, y_ = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
        positions = np.vstack([x_.ravel(), y_.ravel()])

        # Evaluate the KDE at each point in the grid
        z_ = np.reshape(kde(positions).T, x_.shape)
    

        # Plot the KDE
        fig, ax = plt.subplots(figsize=(8, 6))
        # confidence_ellipse(popt[0], popt[1], np.diag(popt[3:5]), ax, n_std=1.0, edgecolor='blue')
        # confidence_ellipse(popt[0], popt[1], np.diag(popt[3:5]), ax, n_std=2.0, edgecolor='blue')

        plt.contour(x_, y_, z_, cmap='viridis', linestyles='--')
        # plt.contourf(x, y, z, cmap='viridis')
        plt.scatter(x, y, s=10, alpha=0.2, color='b', marker='.')
        confidence_ellipse(xy_mean[0], xy_mean[1], covariance_matrix_xy, ax, n_std=1.0, edgecolor='red')
        confidence_ellipse(xy_mean[0], xy_mean[1], covariance_matrix_xy, ax, n_std=2.0, edgecolor='red')


        confidence_ellipse(self.prior[0], self.prior[1], self.prior_covariance_matrix_t[:2,:2],ax, n_std=1.0, edgecolor='green')
        confidence_ellipse(self.prior[0], self.prior[1], self.prior_covariance_matrix_t[:2,:2],ax, n_std=2.0, edgecolor='green')


        plt.axis('equal')

        # plt.colorbar(label='Density')
        plt.title('2D Kernel Density Estimation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

def confidence_ellipse(x, y, cov, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    # cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if __name__ == "__main__":
    from LivoxScanParser import PCD

    downsample_leaf_size = 0.1

    pcd1 = PCD()
    pcd2 = PCD()
    pcd1.load_from_csv("/Users/jhh/Desktop/ASL Dataset/Stairs/local_frame/Hokuyo_0.csv")
    pcd2.load_from_csv("/Users/jhh/Desktop/ASL Dataset/Stairs/local_frame/Hokuyo_1.csv")
    transformation_params = np.array([0,0,0,0,0,0])

    # pcd1.load_from_o3d("sphere_cube_random_00_10000.pcd")
    # pcd2.load_from_o3d("sphere_cube_random_01_10000.pcd")
    # transformation_params = np.array([0.7, 1.3, 0.3, 0.0, 0.5, 0.2])

    # pcd1.load_from_o3d("mid360_06_static_cloud.pcd")
    # pcd2.load_from_o3d("mid360_07_static_cloud.pcd")
    # transformation_params = np.array([0,0,0,0.7,0.2,0.5])
    # transformation_params = np.array([0,0,0,  *np.random.uniform(-1.0, 1.0 , 3)])


    # pcd1.load_from_o3d("hap_05_static_cloud.pcd")
    # pcd2.load_from_o3d("hap_06_static_cloud.pcd")
    # transformation_params = np.array([0,0,0,0.0,0,0])


    point_cloud_00 = pcd1.get_points(downsample_leaf_size)
    point_cloud_01 = transform_3d_points(pcd2.get_points(downsample_leaf_size) , create_3d_transformation_matrix(*transformation_params))

    conditions = {ScanMatcher.ITERATIONS: np.int64(5000),
                 ScanMatcher.TRANSLATION_EPSILON: 1e-8,
                           }

    scan_matcher = ScanMatcher(ScanMatcher.P2P, break_conditions=conditions)

    scan_matcher.set_batch_size(50)


    scan_matcher.set_bayesian_prior([-0.0,-0.00,0.0,0,0,0], [1.,1.,1.,.1,.1,.1])
    # scan_matcher.set_bayesian_prior([0.0,0.0,0.0,0,0,0], [.1,.1,.1,.1,.1,.1])
    # scan_matcher.set_bayesian_prior([-0.7,0.07,0.1,0,0,0], [1,1,1,1,1,1])
    # scan_matcher.set_bayesian_prior([0.8, 1.4, 0.2, 0.1, -0.1, 0.3], [.1,.1,.1,.1,.1,.1])
    scan_matcher.set_source(point_cloud_00)
    scan_matcher.set_target(point_cloud_01)

    
    # scan_matcher.calculate_normals(radius=1.0)
    

    scan_matcher.plot_source_target()

    # scan_matcher.registrate()
    scan_matcher.vizualize_registrate()

    scan_matcher.plot_source_target_final()

    scan_matcher.plot_losses()

    scan_matcher.plot_bayes_kde()
    scan_matcher.plot_bayes_translation_cov()

    plt.show()