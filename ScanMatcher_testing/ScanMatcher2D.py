from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit


from scipy.spatial import cKDTree
from scipy.optimize import minimize

def rotation_matrix_2d(yaw):
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                [np.sin(yaw), np.cos(yaw), 0],
                                [0, 0, 1]])
    return rotation_matrix

def create_2d_transformation_matrix(x, y, yaw):
    """
    Create a 2D transformation matrix from translation in x and y, and rotation (yaw).

    Parameters:
    - x (float): Translation in the x-direction.
    - y (float): Translation in the y-direction.
    - yaw (float): Rotation angle around the z-axis (yaw) in radians.

    Returns:
    - transformation_matrix (array): The 2D transformation matrix.
    """
    translation_matrix = np.array([[1, 0, x],
                                   [0, 1, y],
                                   [0, 0, 1]])

    rotation_matrix = rotation_matrix_2d(yaw)

    transformation_matrix = np.dot(translation_matrix, rotation_matrix)

    return transformation_matrix

def transform_2d_points( points, transformation_matrix):
        """
        Transform 2D points using a 2D transformation matrix.

        Parameters:
        - points (array-like): 2D points in the form [[x1, y1], [x2, y2], ...].
        - transformation_matrix (array): The 2D transformation matrix.

        Returns:
        - transformed_points (array): Transformed 2D points.
        """
            
        points = np.array(points)
        homogenous_points = np.column_stack((points, np.ones(len(points))))

        transformed_homogenous = np.dot(transformation_matrix, homogenous_points.T).T

        transformed_points = transformed_homogenous[:, :2]

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


import numpy as np

class AdamScheduler:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0      # Time step
        
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
        parameters = -step_size 
        
        return parameters



class ScanMatcher2D:
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
        self.final_transformation_matrix = np.eye(3)
        self.transformation_matrix = np.eye(3)
        # self.transformation_matrix[:2,2] = np.float64('inf')
        self.set_conditions(break_conditions)
        self.did_converge = False
        self.iter = 0
        self.losses = []
        self.translation_epsilons = []

        self.N = None

        self.batch_size = 15

        self.linear_adam_scheduler = AdamScheduler(lr=0.1, beta1=0.8, beta2=0.9999999)
        self.rotional_adam_scheduler = AdamScheduler(lr=0.1, beta1=0.5, beta2=0.9999999)
        # self.not_picked_indices_initial = set(range(self.source.shape[0]))
        
        self.rescale_points_a = None
        self.rescale_points_b = None

        self.method = method

        self.X_log = []
        self.X = np.zeros(3)
        self.X_std = np.ones(3) * 2
        self.X_mean = np.zeros(3)

        self.V = 0
        self.beta = 0.9
        self.lmbda = 1e-8


    def set_conditions(self, break_conditions):
        # set default conditions
        self.conditions = {ScanMatcher2D.ITERATIONS: np.int64(2500),
                           ScanMatcher2D.TRANSLATION_EPSILON: 1e-6,
                           }
        if not break_conditions is None:
            self.conditions.update(break_conditions)

    def set_bayesian_prior(self, means, stds):
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
        if self.method == ScanMatcher2D.P2P:
            self.transformation_matrix = self._find_rigid_transform_p2p(self.source_transformed, self.target)
        elif self.method == ScanMatcher2D.P2PL:
            #TODO if no normals self.calculate_normals()
            self.transformation_matrix = self._find_rigid_transform_p2pl(self.source_transformed, self.target, self.source_normals_transformed, self.target_normals)
        elif self.method == ScanMatcher2D.N2N:
            self.transformation_matrix = self._find_rigid_transform_n2n(self.source_transformed, self.target, self.source_normals_transformed, self.target_normals)
        elif self.method == ScanMatcher2D.SGD:
            # self.transformation_matrix = self._find_rigid_transform_sgd(self.source_normalized, self.target_normalized, self.source_normals_transformed, self.target_normals)
            self.transformation_matrix = self._find_rigid_transform_sgd(self.source, self.target, self.source_normals_transformed, self.target_normals)
        elif self.method == ScanMatcher2D.BAYES:
            self.transformation_matrix = self._find_rigid_transform_bayes(self.source, self.target, self.source_normals_transformed, self.target_normals)


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
            transformation  = create_2d_transformation_matrix(x[0], x[1], x[2])
            s = transform_2d_points(source[self.point_pairs_idx[:,0]], transformation)
            r = target[[self.point_pairs_idx[:,1]]]
            N = s.shape[0]
            loss = np.sum(np.square(np.linalg.norm(s-r, axis=1))) / N
            return loss
        
        result = self.minimize_loss(P2P_LOSS)
        x = result.x
        transformation = create_2d_transformation_matrix(x[0], x[1], x[2])
        
        return transformation
    
    def _find_rigid_transform_p2pl(self, source, target, source_normals, target_normals):
        def P2PL_LOSS(x):
            transformation  = create_2d_transformation_matrix(x[0], x[1], x[2])
            s = transform_2d_points(source[self.point_pairs_idx[:,0]], transformation)
            r = target[self.point_pairs_idx[:,1]]
            d = s-r
            d_n = np.einsum('ij,ij->i', d, target_normals[self.point_pairs_idx[:,1]])
            N = s.shape[0]
            loss = np.sum(np.square(d_n)) / N
            return loss
        
        result = self.minimize_loss(P2PL_LOSS)
        x = result.x
        transformation = create_2d_transformation_matrix(x[0], x[1], x[2])
        
        return transformation
    
    def _find_rigid_transform_n2n(self, source, target, source_normals, target_normals):
        def N2N_LOSS(x):
            transformation  = create_2d_transformation_matrix(x[0], x[1], x[2])
            s = transform_2d_points(source[self.point_pairs_idx[:,0]], transformation)
            r = target[self.point_pairs_idx[:,1]]
            d = np.linalg.norm(s-r, axis=1)
            # d_p2p = np.einsum('ij,ij->i', d, target_normals)

            s_n = transform_2d_points(source_normals[self.point_pairs_idx[:,0]], transformation)
            r_n = target_normals[self.point_pairs_idx[:,1]]
            d_n = angles_between_vectors(s_n, r_n)
            N = s_n.shape[0]
            # loss = np.sum(np.square(d_n) + np.square(d)) / N
            loss = np.sum(np.square(d_n)) / N
            return loss
        
        result = self.minimize_loss(N2N_LOSS)
        x = result.x
        transformation = create_2d_transformation_matrix(x[0], x[1], x[2])
        
        return transformation
    
    def _find_rigid_transform_sgd(self, source, target, source_normals, target_normals):

        x = self.X
        N = self.point_pairs_idx.shape[0]

        def linear_gradient():
            return np.array([1,1])

        def rotational_gradient(yaw):
            gradient_rot_mat = np.array([[-np.sin(yaw), -np.cos(yaw), 0],
                                         [np.cos(yaw),  -np.sin(yaw), 0],
                                         [0, 0, 0]])
            return gradient_rot_mat
        
        # transformation  = create_2d_transformation_matrix(x[0]/self.rescale_points_a[0], x[1]/self.rescale_points_a[1], x[2])
        transformation  = create_2d_transformation_matrix(x[0], x[1], x[2])

        # s = source[self.point_pairs_idx[:,0]]
        s = source[self.point_pairs_idx[:,0]]
        s_prime = transform_2d_points(s, transformation)
        r = target[self.point_pairs_idx[:,1]]

        sample_loss = np.sum(s_prime - r, axis=0)

        if True: # use ADAM step size scheduling
            linear_update_adam  = self.linear_adam_scheduler.step( 2/(3*N) * sample_loss * linear_gradient() )
            rotational_update_adam = self.rotional_adam_scheduler.step(1/(3*N) * np.sum(np.sum(np.multiply(s_prime - r , transform_2d_points(s, rotational_gradient(x[2]))), axis=1), axis=0) )
            new_x_t = x[:2] + linear_update_adam
            new_x_r = x[2:] + rotational_update_adam 
        else:
            new_x_t = x[:2] - 0.1 * 1/(2*N) * np.sum(s_prime - r, axis=0) * linear_gradient()
            new_x_r = x[2:] - 0.03 * 1/(2*N) * np.sum(np.sum(np.multiply(s_prime - r , transform_2d_points(s, rotational_gradient(x[2]))),axis=1  ), axis=0)


        self.X = np.r_[new_x_t, new_x_r]
        print(new_x_t)

        new_transformation = create_2d_transformation_matrix(*self.X)


        loss = np.sum(np.square(np.linalg.norm(transform_2d_points(s, new_transformation) - r, axis=1))) / N
        self.losses.append(loss)

        # transformation_delta = new_transformation.T @ transformation
        # transformation_delta = np.linalg.inv(new_transformation) @ transformation
        transformation_delta =  new_transformation @ np.linalg.inv(transformation)

        # print(self.X)

        return transformation_delta
        # return new_transformation

    def _find_rigid_transform_bayes(self, source, target, source_normals, target_normals):
        x = self.X
        mu = self.X_mean
        sigma = self.X_std
        m = self.point_pairs_idx.shape[0]

        def linear_gradient():
            return np.array([1,1])

        def rotational_gradient(yaw):
            gradient_rot_mat = np.array([[-np.sin(yaw), -np.cos(yaw), 0],
                                         [np.cos(yaw),  -np.sin(yaw), 0],
                                         [0, 0, 0]])
            return gradient_rot_mat
        
        # transformation  = create_2d_transformation_matrix(x[0]/self.rescale_points_a[0], x[1]/self.rescale_points_a[1], x[2])
        transformation  = create_2d_transformation_matrix(x[0], x[1], x[2])

        # s = source[self.point_pairs_idx[:,0]]
        s = source[self.point_pairs_idx[:,0]]
        s_prime = transform_2d_points(s, transformation)
        r = target[self.point_pairs_idx[:,1]]

        # compute mean gradients
        translation_mean_gradient = np.sum(s_prime - r, axis=0) / m * linear_gradient()
        rotational_mean_gradient = np.sum(np.sum(np.multiply((s_prime - r)/ m , transform_2d_points(s, rotational_gradient(x[2]))),axis=1  ), axis=0)
        
        mean_gradients = np.r_[translation_mean_gradient, rotational_mean_gradient]

        # compute preconditioners 
        self.V = self.beta * self.V + np.multiply((1-self.beta)*mean_gradients, mean_gradients)
        A = ( 1 /(self.lmbda + np.sqrt(self.V)))
        # A = np.diag( 1 /(self.lmbda + np.sqrt(self.V)))

        # compute prior negative log gradient
        negative_log_prior_gradient_t = (x[:2] - mu[:2]) / sigma[:2]
        negative_log_prior_gradient_r = np.sin(x[2:] - mu[2:]) / sigma[2:]

        step_t = 0.0005
        step_r = 0.0001
        injected_noise = np.c_[np.random.normal(0,step_t*A[0]), np.random.normal(0,step_t*A[1]), np.random.normal(0,step_r*A[2])].flatten()

        # if True: # use ADAM step size scheduling
        #     linear_update_adam  = self.linear_adam_scheduler.step(  self.N * sample_loss * linear_gradient() )
        #     rotational_update_adam = self.rotional_adam_scheduler.step(1/(3*N) * np.sum(np.sum(np.multiply(s_prime - r , transform_2d_points(s, rotational_gradient(x[2]))), axis=1), axis=0) )
        #     new_x_t = x[:2] + linear_update_adam
        #     new_x_r = x[2:] + rotational_update_adam 
        # else:
        new_x_t = x[:2] - step_t/2 * A[:2] * (negative_log_prior_gradient_t + self.N * translation_mean_gradient) + injected_noise[:2]
        new_x_r = x[2:] - step_r/2 * A[2:] * (negative_log_prior_gradient_r + self.N * rotational_mean_gradient) + injected_noise[2:]


        self.X = np.r_[new_x_t, new_x_r]
        self.X_log.append(self.X.copy())

        mean_X = np.mean(self.X_log, axis=0)
        # print(mean_X)

        # new_transformation = create_2d_transformation_matrix(*mean_X)
        new_transformation = create_2d_transformation_matrix(*self.X)


        loss = np.sum(np.square(np.linalg.norm(transform_2d_points(s, new_transformation) - r, axis=1))) / m
        self.losses.append(loss)

        # transformation_delta = new_transformation.T @ transformation
        # transformation_delta = np.linalg.inv(new_transformation) @ transformation
        transformation_delta =  new_transformation @ np.linalg.inv(transformation)

        # print(self.X)

        return transformation_delta
        # return new_transformation

    def minimize_loss(self, loss_function):
        initial_guess = np.array([0,0,0]) # x, y, yaw
        result = minimize(loss_function, initial_guess, method='L-BFGS-B') # scipy minimizer
        self.losses.append(result.fun)
        return result

    def transform_source(self):
        # self.source_transformed = transform_2d_points(self.source_transformed, self.transformation_matrix)
        self.source_transformed = transform_2d_points(self.source, self.final_transformation_matrix)
        try:
            normal_transform = self.final_transformation_matrix.copy()
            normal_transform[:2, 2] = 0.0
            self.source_normals_transformed = transform_2d_points(self.source_normals_transformed, normal_transform)
        except:
            print('')

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
        self.source_normals = None

        self.final_transformation_matrix = create_2d_transformation_matrix(*self.X)
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
        self.target_normals = None

        # self.target_normalized = (target - np.min(target, axis=0)) / (np.max(target, axis=0) - np.min(target, axis=0))
        self.target_normalized = self.rescale_points(target, None)

    def set_initial_guess(self, initial_guess=None):
        if initial_guess == None:
            self.initial_guess = np.array([0,0,0])
        self.initial_guess

    def get_transformation(self):
        if self.method == ScanMatcher2D.BAYES:
            mean_X = np.mean(self.X_log, axis=0)
            self.final_transformation_matrix = create_2d_transformation_matrix(*mean_X)
        return self.final_transformation_matrix


    def assert_break_conditions(self):
        translation_epsilon = np.linalg.norm(self.transformation_matrix[:2,2])
        # print(f": {translation_epsilon}")

        if self.iter == 0:
            self.did_converge = False
            return
        
        self.translation_epsilons.append(translation_epsilon)

        if translation_epsilon < self.conditions.get(ScanMatcher2D.TRANSLATION_EPSILON):
            self.did_converge = True

        return self.did_converge

    def run_iterator(self):
        for _ in range(self.conditions.get(ScanMatcher2D.ITERATIONS)):
            self.run_step()
            yield

    def run_step(self):
        if self.method == ScanMatcher2D.SGD or self.method == ScanMatcher2D.BAYES:
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
            ax.clear()  # Clear the previous plot
            ax.scatter(self.target[:, 0], self.target[:, 1], marker='o', label='Target')
            ax.scatter(self.source[:, 0], self.source[:, 1], marker='o', label='Source')
            ax.scatter(self.source_transformed[:, 0].copy(), self.source_transformed[:, 1].copy(), marker='o', label='Source Transformed')
            ax.set_title(f'Scan Matching - Iteration {self.iter}')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.legend()



        fig, ax = plt.subplots(figsize = ScanMatcher2D.FIGSIZE)
        ax.axis('equal')
        animation = FuncAnimation(fig, update, frames=self.conditions[ScanMatcher2D.ITERATIONS], interval=1, repeat=False)
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
        fig = plt.figure(figsize=ScanMatcher2D.FIGSIZE)
        plt.plot(self.losses)

    def plot_translation_epsilon(self):
        fig = plt.figure(figsize=ScanMatcher2D.FIGSIZE)
        plt.plot(self.translation_epsilons)


    def calculate_eigen_normal_2d(self, points):
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
            source_normals_list.append(self.calculate_eigen_normal_2d(group))

        self.source_normals = np.array(source_normals_list)
        self.source_normals_transformed = np.array(source_normals_list)

        # target normals
        ii = self.target_kd_tre.query_ball_point(self.target, radius)
        target_normals_list = []
        groups = [self.target[i] for i in ii]
        for group in groups:
            target_normals_list.append(self.calculate_eigen_normal_2d(group))

        self.target_normals = np.array(target_normals_list)


        # dd, ii = self.source_kd_tre.query_ball_point(self.target, 0.5)
    
    def _plot_source(self, ax):
        ax.scatter(self.source[:, 0], self.source[:, 1], marker='o', label='Source', color='C0')
        self._plot_normals(ax, self.source, self.source_normals, label='Source Normals', angles='xy', color='C0')
        
    def _plot_target(self, ax):
        ax.scatter(self.target[:, 0], self.target[:, 1], marker='o', label='Target', color='C1')
        self._plot_normals(ax, self.target, self.target_normals, label='Target Normals', angles='xy', color='C1')

        
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

        if not self.method is ScanMatcher2D.BAYES:
            return
        # Gaussian function to fit to KDE
        # def gaussian(x, amp, mean, std):
        #     return amp * np.exp(-0.5 * ((x - mean) / std)**2)
        def gaussian(x, mean, std):
            return 1/(std * np.sqrt(2 *np.pi)) * np.exp(-0.5 * ((x - mean) / std)**2)

        burn_in = 40
        data = np.asarray(self.X_log)

        # Get the number of columns
        num_columns = data.shape[1]

        # Create subplots for each column
        fig, axs = plt.subplots(nrows=1, ncols=num_columns, figsize=(10, 4), layout='constrained')

        # Iterate over each column
        for i in range(num_columns):
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
            
            # Plot KDE for the current column
            axs[i].plot(x, kde_values, label='KDE')
            axs[i].plot(peak_location, kde_values[max_index], 'ro', label=f'Peak at {peak_location:.2f}')  # Plot peak
            axs[i].plot(x, gaussian(x, *popt), 'r--', label=f'Gaussian Fit, $\mu$={mean:.2f}, $\sigma$={std:.2f}')  # Plot fitted curve
            axs[i].set_title(f'KDE for Column {i+1}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Density')
            axs[i].legend()


    def plot_bayes_translation_cov(self):

        if not self.method is ScanMatcher2D.BAYES:
            return
        
        # Example 2D data
        data = np.asarray(self.X_log)[:,:2]

        # Transpose the data to fit the expected shape
        data = data.T

        # Create 2D KDE
        kde = gaussian_kde(data)

        # Compute covariance matrix
        covariance_matrix = np.cov(data)

        # Compute standard deviations
        std_devs = np.sqrt(np.diag(covariance_matrix))
        print(f"Translation standard deviation: {std_devs}")

        # Define a grid of points for evaluation
        x, y = np.meshgrid(np.linspace(min(data[0]), max(data[0]), 100), np.linspace(min(data[1]), max(data[1]), 100))
        positions = np.vstack([x.ravel(), y.ravel()])

        # Evaluate the KDE at each point in the grid
        z = np.reshape(kde(positions).T, x.shape)
    

        # Plot the KDE
        plt.figure(figsize=(8, 6))
        plt.contour(x, y, z, cmap='viridis', linestyles='--')
        # plt.contourf(x, y, z, cmap='viridis')
        # plt.scatter(data[0,:],data[1,:], s=5, alpha=0.1, color='k', marker='.')

        # Plot ellipse for 1 standard deviation
        ellipse = plt.Circle((np.mean(data[0]), np.mean(data[1])), std_devs[0], edgecolor='red', fill=False, linestyle='-')
        plt.gca().add_patch(ellipse)

        # Plot ellipse for 2 standard deviations
        ellipse = plt.Circle((np.mean(data[0]), np.mean(data[1])), 2 * std_devs[0], edgecolor='red', fill=False, linestyle='-')
        plt.gca().add_patch(ellipse)

        plt.colorbar(label='Density')
        plt.title('2D Kernel Density Estimation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


        