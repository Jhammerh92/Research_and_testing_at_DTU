import numpy as np
from plane_analysis_functions import *
from NormalSpaceClustering import *


def normal_clustering_grad_descent(normals):


    def loss_f(test_normal, normals):
        test_normal /= np.linalg.norm(test_normal)
        angular_distances, direction = vector_angular_distance_and_direction(test_normal, normals)
        # angular_distances = vector_angular_distance(test_normal, normals)
        # n = np.sum(~(angular_distances > np.pi/2))
        # angular_distances[angular_distances > np.pi/2] = 0.0
        loss = np.sum(angular_distances**2)
        return loss
    
    def random_normal():
        normal = np.random.uniform(-1, 1, (1,3))
        normal /= np.linalg.norm(normal)
        return normal

    dx = 0.01
    step = 0.01
    cardinals = np.eye(3)
    n = normals.shape[0]

    indices = np.array([i for i in range(n)])
    c_index = np.zeros(n, dtype=int) -1

    clusters = []
    cluster_centers = []

    normals_in_pool = normals.copy()

    for k in range(4):
        cluster_normal = random_normal()
        loss = loss_f(cluster_normal, normals_in_pool)
        for j in range(1000):
            grads = np.zeros_like(cluster_normal)
            for i in range(3):
                step_vector = cardinals[i,:] * dx
                test_normal = cluster_normal + step_vector
                
                grad_loss = loss - loss_f(test_normal, normals_in_pool)
                grads[0,i] = grad_loss

            cluster_normal += grads * step
            cluster_normal /= np.linalg.norm(cluster_normal)
            loss = loss_f(cluster_normal, normals_in_pool)

            print(loss, cluster_normal)


        ang_dist = vector_angular_distance(test_normal, normals_in_pool)

        cluster_mask = ang_dist < np.deg2rad(15)
        normals_in_pool = normals_in_pool[~cluster_mask.T[0],:]

        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(*normals_in_pool.T, s=1, alpha=0.5)
        plt.show()

        clusters.append(indices[cluster_mask.T[0]])
        indices = indices[~cluster_mask.T[0]]
        c_index[clusters[-1]] = k
        cluster_centers.append(cluster_normal)



    # cluster_normal /= np.linalg.norm(cluster_normal)
    return clusters, cluster_centers


def mean_normal_grad_descent(normals, norm=2, weights=None):

    def loss_f(test_normal, normals):
        test_normal /= np.linalg.norm(test_normal)
        # angular_distances, direction = vector_angular_distance_and_direction(test_normal, normals)
        angular_distances = vector_angular_distance(test_normal, normals)
        # n = np.sum(~(angular_distances > np.pi/2))
        # angular_distances[angular_distances > np.pi/2] = 0.0
        loss = np.mean((angular_distances**norm * weights))
        return loss
    
    normals = np.array(normals)
    if weights is None:
        weights = np.ones((normals.shape[0],1))

    dx = 0.001
    step = 0.001
    cardinals = np.eye(3)
    n = normals.shape[0]

    cluster_normal = np.mean(normals, axis = 0, keepdims=True)
    cluster_normal /= np.linalg.norm(cluster_normal)
    loss = loss_f(cluster_normal, normals)
    grads_norm = float('inf')
    # while grads_norm > 1e-4:
    for _ in range(20):
        # print(loss, cluster_normal)
        grads = np.zeros_like(cluster_normal)
        for i in range(3):
            step_vector = cardinals[i,:] * dx
            test_normal = cluster_normal + step_vector
            
            grad_loss = loss - loss_f(test_normal, normals)
            grads[0,i] = grad_loss

        new_cluster_normal = cluster_normal + grads * step
        new_cluster_normal /= np.linalg.norm(new_cluster_normal)
        loss = loss_f(cluster_normal, normals)
        grads_norm = np.linalg.norm(new_cluster_normal - cluster_normal)
        cluster_normal = new_cluster_normal

    # cluster_normal /= np.linalg.norm(cluster_normal)

    # ax = plt.figure().add_subplot(111, projection='3d')
    # ax.scatter(*normals.T, s=1, alpha=0.5)
    # ax.scatter(*cluster_normal.T, s=25, alpha=0.5, color='r')
    # plt.show()




    return cluster_normal

def mean_normal_density(normals, densities):

    def loss_f(test_normal, normals):
        test_normal /= np.linalg.norm(test_normal)
        # angular_distances, direction = vector_angular_distance_and_direction(test_normal, normals)
        angular_distances = vector_angular_distance(test_normal, normals)
        # n = np.sum(~(angular_distances > np.pi/2))
        # angular_distances[angular_distances > np.pi/2] = 0.0
        # loss = np.mean((densities / (angular_distances+1e-6)))
        loss = np.mean((angular_distances / (densities)))
        return loss
    
    normals = np.array(normals)
    if densities is None:
        densities = np.ones((normals.shape[0],1))

    dx = 0.001
    step = 1e-0
    cardinals = np.eye(3)
    n = normals.shape[0]

    cluster_normal = np.mean(normals, axis = 0, keepdims=True)
    cluster_normal /= np.linalg.norm(cluster_normal)
    loss = loss_f(cluster_normal, normals)
    grads_norm = float('inf')
    # while grads_norm > 1e-4:
    for _ in range(20):
        # print(loss, cluster_normal)
        grads = np.zeros_like(cluster_normal)
        for i in range(3):
            step_vector = cardinals[i,:] * dx
            test_normal = cluster_normal + step_vector
            
            grad_loss = loss - loss_f(test_normal, normals)
            grads[0,i] = grad_loss

        # step = 1.0 / (np.max(grads)*1e2)

        new_cluster_normal = cluster_normal + grads * step
        new_cluster_normal /= np.linalg.norm(new_cluster_normal)
        loss = loss_f(cluster_normal, normals)
        grads_norm = np.linalg.norm(new_cluster_normal - cluster_normal)
        cluster_normal = new_cluster_normal

    # cluster_normal /= np.linalg.norm(cluster_normal)

    # ax = plt.figure().add_subplot(111, projection='3d')
    # ax.scatter(*normals.T, s=1, alpha=0.5)
    # ax.scatter(*cluster_normal.T, s=25, alpha=0.5, color='r')
    # plt.show()




    return cluster_normal

def _normal_clustering(normals):
    n = normals.shape[0]
    indices = np.array([i for i in range(n)])
    c_index = np.zeros(n, dtype=int) - 1.0 

    clusters = [[]]
    cluster_indices = [[]]
    cluster_centers = []

    cluster_dist_threshold = np.deg2rad(30)

    densities = np.array([np.mean(vector_angular_distance(normal, normals)) for normal in normals])
    argsort = np.argsort(densities)
    normals = normals[argsort]

    for index, normal in enumerate(normals):
        if not cluster_centers:
            cluster_centers.append(normal)
            cluster_indices[0].append(index)
            clusters[0].append(normal)
            continue
        # dist_to_centers = []
        min_dist = float('inf')
        min_k = -1
        for k, center in enumerate(cluster_centers):
            dist = vector_angular_distance(normal, center)
            # dist_to_centers.append(dist)
            if dist < min_dist:
                min_dist = dist
                min_k = k

        if min_dist < cluster_dist_threshold:
            cluster_indices[min_k].append(index)
            clusters[min_k].append(normal)
            cluster_centers[k] = mean_normal_grad_descent(clusters[min_k])
            ## make a function to calculate cluster center
        else:
            clusters.append([normal])
            cluster_centers.append(normal)
            cluster_indices.append([index])


    return cluster_indices, cluster_centers

        
def __normal_clustering(normals):
    def random_normal():
        normal = np.random.uniform(-1, 1, (1,3))
        normal /= np.linalg.norm(normal)
        return normal

    n = normals.shape[0]
    indices = np.array([i for i in range(n)])
    c_index = np.zeros(n, dtype=int) - 1.0 

    clusters = []
    cluster_indices = []
    cluster_centers = []

    cluster_dist_threshold = np.deg2rad(30)

    normals_in_pool = normals.copy()
    # distances = np.array([vector_angular_distance(normal, normals) for normal in normals])

    while len(indices) > 0:
        dist = vector_angular_distance(random_normal(), normals_in_pool)

        for i in range(10):
            cluster_mask = dist < cluster_dist_threshold
            cluster = normals_in_pool[cluster_mask.T[0]]
            cluster_center = mean_normal_grad_descent(cluster)
            dist = vector_angular_distance(cluster_center, normals_in_pool)

        cluster_indices.append(indices[cluster_mask.T[0]])
        cluster_centers.append(cluster_center)
        normals_in_pool = normals_in_pool[~cluster_mask.T[0],:]

        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(*normals_in_pool.T, s=1, alpha=0.5)
        plt.show()

        clusters.append(indices[cluster_mask.T[0]])
        indices = indices[list(~cluster_mask.T[0])]
        # c_index[clusters[-1]] = k
        

    return cluster_indices, cluster_centers


def normal_clustering(normals, densities=None):
    def random_normal():
            normal = np.random.uniform(-1, 1, (1,3))
            normal /= np.linalg.norm(normal)
            return normal
    
    if densities is None:
        densities = np.ones((normals.shape[0],1))

    n, _ = normals.shape

    c = 20
    loss = []
    cluster_centers = [[] for i in range(c)]
    min_indices = [[] for i in range(c)]
    for k in range(c):
        if k > 0:
            cluster_centers[k].extend(cluster_centers[k-1]) 
        cluster_centers[k].append(random_normal())
        for i in range(25): # should run until a change of centers is less than some delta..
            distances = np.array([vector_angular_distance(center, normals) for center in cluster_centers[k]])
            min_indices[k] = np.argmin(distances, axis=0).ravel()

            for j, center in enumerate(cluster_centers[k]):
                mask = min_indices[k] == j
                # mask = mask.ravel()
                # cluster_centers[k][j] = mean_normal_grad_descent(normals[mask],norm=2, weights=densities[mask])
                # cluster_centers[k][j] = mean_normal_density(normals[mask], densities=densities[mask])
                average_center = np.average(normals[mask], weights=densities[mask].ravel(), axis=0)
                cluster_centers[k][j] = np.array([average_center / np.linalg.norm(average_center)])


        loss_i = 0.0
        for j, center in enumerate(cluster_centers[k]):
            mask = min_indices[k] == j
            # densities[mask] * 
            loss_i += np.mean(( vector_angular_distance(center, normals[mask])/ densities[mask])**2 )

        loss.append(loss_i)

    best_loss_index = np.argmin(loss)
    distances = np.array([vector_angular_distance(center, normals) for center in cluster_centers[best_loss_index]])
    min_indices = np.argmin(distances, axis=0)

    cluster_indices = [np.where(min_indices==i)[0] for i in range(len(cluster_centers[best_loss_index]))]

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(loss)),loss)
    plt.show()

    return cluster_indices, cluster_centers[best_loss_index]

def triangle_area_from_medians(medians):
    a , b, c = 3/2*medians.T
    return 0.5 * np.sqrt(2*a**2 + 2*b**2*c**2 + 2*c**2+a**2 - a**4 - b**4 - c**4)

def triangle_area_circular(medians):
    # assume the average of the medians are like a radius of a circle
    radii = np.mean(medians, axis=1, keepdims=True)
    return radii**2 * np.pi

if __name__ == "__main__":

    plane_normals = np.load('test_normals.npy')
    angs, dirs = vector_angular_distance_and_direction([0.0,0.0,1.0], plane_normals)
    nearest_dists = np.array([np.partition(vector_angular_distance(normal, plane_normals), 3)[1:4] for normal in plane_normals])
    areas = triangle_area_from_medians(nearest_dists)
    densities = np.atleast_2d(1.0/(areas + 1e-6)).T


    # fig_3d = plt.figure()
    # ax_3d = fig_3d.add_subplot(111, projection='3d')
    # ax_3d.scatter(*plane_normals.T, s=2, alpha=0.5, c=densities, vmin=np.min(densities), vmax=100.0)
    # ax_polar = plt.figure().add_subplot(111, projection=None)
    # ax_polar.scatter( np.rad2deg(dirs), np.rad2deg(angs), s=2, alpha=0.5, c=densities, vmin=np.min(densities), vmax=100.0)
    # plt.show()

    mask = areas < 0.05
    tested_normals = plane_normals[mask]
    tested_densities = densities[mask]

    cluster_index, cluster_normals = normal_clustering(tested_normals, densities=tested_densities)

    
    ref = np.eye(3)
    fig, ax = plt.subplots()
    # fig_2, ax_2 = plt.subplots()
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    for ci, cn in zip(cluster_index, cluster_normals):
        angs = vector_angular_distance(cn, tested_normals[ci,:])
        pdf, hist, bin_centers = calculate_probability_density(angs)
        ax.plot(np.rad2deg(bin_centers[0]), hist)
        # ax_2.scatter(angs, np.log10(densities[ci,:]), s=2)

        ax_3d.scatter(*tested_normals[ci,:].T, s=1, alpha=0.5)
        ax_3d.scatter(*cn.T*1.001, s=100, alpha=1.0, color='r')


    # test_normals = plane_normals[cluster_index[0],:]
    # test_densities = areas[cluster_index[0]]
    # cn = cluster_normals[0]
    # losses = []

    # fig, ax = plt.subplots()
    # fig_3d = plt.figure()
    # ax_3d = fig_3d.add_subplot(111, projection='3d')
    # for i in range(10):
    #     mask = test_densities < 1/(10-i) * np.max(test_densities)
    #     d = test_densities[mask.T]
    #     n = test_normals[mask.T]

    #     angs = vector_angular_distance(cn, n)
    #     pdf, hist, bin_centers = calculate_probability_density(angs)
    #     ax.plot(np.rad2deg(bin_centers[0]), hist)

    #     ax_3d.scatter(*n.T, s=1, alpha=0.5)
    #     plt.show()
    #     ax_3d.clear()
    #     loss = np.mean(vector_angular_distance(cn, n)**2)
    #     losses.append(loss)

    # ax.scatter(np.array(losses))
    # plt.show()





    plt.show()