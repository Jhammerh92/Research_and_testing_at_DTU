import numpy as np
import matplotlib.pyplot as plt


# make some points and normals in 2D

def interpolate(p1, p2, level):
    """ Linearly interpolate between points p1 and p2 at the given level. """
    (x1, y1, z1), (x2, y2, z2) = p1, p2
    t = (level - z1) / (z2 - z1)
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

def find_contour_points(x, y, z, level):
    contour_points = []
    rows, cols = z.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            corners = [(x[i, j], y[i, j], z[i, j]), 
                       (x[i+1, j], y[i+1, j], z[i+1, j]), 
                       (x[i, j+1], y[i, j+1], z[i, j+1]), 
                       (x[i+1, j+1], y[i+1, j+1], z[i+1, j+1])]
            
            for k in range(4):
                p1, p2 = corners[k], corners[(k+1) % 4]
                if (p1[2] < level <= p2[2]) or (p2[2] < level <= p1[2]):
                    contour_points.append(interpolate(p1, p2, level))
    
    return np.array(contour_points)


def make_error_gradient_space(normals):
    # make gradient plane
    # first make the grid points
    resolution = 25
    size = 5
    grid = np.linspace(-size/2, size/2, resolution)
    if normals.shape[1] > 2:
        X,Y,Z = np.meshgrid(grid, grid, grid)
    else:
        X,Y = np.meshgrid(grid, grid)
        Z = np.zeros_like(X)

    # gradient_points = np.c_[X.ravel(), Y.ravel()]
    # then calcluate elevation for each normal
    # Z = normals[0,0]*X +  normals[0,1] *Y
    # Z =  gradient_points @ normals
    # test_z = normals @ test_delta.T**2
    # test_z = np.einsum("ij,ij->i", normals, test_delta**2)
    # Z = np.zeros_like(X)

    vals = np.zeros_like(Z)
    for normal in normals:
        vals += np.abs(normal[0])*X**2 + np.abs(normal[1])*Y**2
        try:
            vals += np.abs(normal[2])*Z**2
        except:
            pass
        # Z += np.sqrt((X**2 + Y**2))
    # Z /= n


    return X,Y,Z, vals

if __name__ == "__main__":

    n = 3
    arc = np.linspace(-np.pi, 1*np.pi, n, endpoint=False)
    points = np.array([[np.cos(x), np.sin(y)] for x,y in zip(arc, arc) ])
    test_points = points + np.array([0.2,0.2])
    normals = - points / np.linalg.norm(points, axis = 1, keepdims=True)

    diff_vector = test_points - points
    # test_delta = (points - test_points).T @ normals
    test_grads = np.einsum("ij,ij->i",- diff_vector, normals)
    test_delta = normals * np.atleast_2d(test_grads).T

    total_delta = np.mean(test_delta,axis=0)

    # plot to check points
    fig, ax = plt.subplots()
    ax.scatter(*points.T)
    ax.quiver(*points.T, *normals.T)
    ax.quiver(*points.T, *diff_vector.T)
    ax.quiver(*test_points.T, *test_delta.T)
    ax.scatter(*test_points.T)
    ax.scatter(*np.mean(test_points, axis=0).T)
    ax.quiver(*np.mean(test_points, axis=0).T,*total_delta.T)
    # plt.show()



    X,Y,Z, vals = make_error_gradient_space(normals)

    normal_gradient_contour = find_contour_points(X,Y,vals, level=1)


    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X,Y,vals, alpha=0.5)
    ax.scatter(0,0,0, s=20, c='r')
    ax.scatter(*test_delta.T, test_grads, s=20, c='b')
    ax.scatter(*normal_gradient_contour.T,np.ones(normal_gradient_contour.shape[0]), s=20, c='g')
    plt.show()