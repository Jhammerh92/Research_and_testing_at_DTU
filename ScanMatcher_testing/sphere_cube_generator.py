import numpy as np
from select import KQ_FILTER_AIO
import numpy as np
import matplotlib.pyplot as plt
from PlotPointCloud3d import plot_cloud_3d_o3d, save_o3d_pcd

# Script to create an overlapping sphere and box that has known normals for normal estimation testing 

## create points on sphere

def create_cube_points(points=150, method="grid"):
    side_length = 2 # don't change!
    def create_face(i, j, plane_level, face_plane="xy"):
        if face_plane=="xy":
            face = np.meshgrid(i, j, plane_level, sparse=False)
        elif face_plane== "xz":
            face = np.meshgrid(plane_level, i, j, sparse=False)
        elif face_plane== "yz":
            face = np.meshgrid(j, plane_level, i, sparse=False)

        face = np.c_[face[0].ravel(), face[1].ravel(), face[2].ravel()]
        return face

    def create_random_face(n, limits, plane_level, face_plane="xy"):
        random_points = np.random.uniform(min(limits), max(limits), (n,2) )
        if face_plane=="xy":
            face = np.c_[random_points,np.tile(plane_level, n)]
        elif face_plane=="xz":
            face = np.c_[np.tile(plane_level, n), random_points]
        elif face_plane=="yz":
            face = np.c_[random_points[:,0],np.tile(plane_level, n), random_points[:,1]]

        return face

    points_per_face = points // 6 # 6 faces on a cube
    points_per_length = int(round(np.sqrt(points_per_face)))
    # dl = side_length/(points_per_length+2)/2
    dl = side_length/(points_per_length)/2
    limits = [-side_length/2, side_length/2]
    
    if method=="grid":
        linear_points = np.linspace(limits[0], limits[1], points_per_length, endpoint=True)
        linear_points_dl = np.linspace(limits[0] + dl, limits[1] - dl, points_per_length, endpoint=True)
        linear_points_small = np.linspace(limits[0]+ 2*dl, limits[1] - 2*dl, points_per_length-1, endpoint=True)
        
        bottom_face = create_face(linear_points, linear_points_dl - dl, limits[0], "xy")
        top_face    = create_face(linear_points, linear_points_dl + dl, limits[1], "xy")

        left_face   = create_face(linear_points_dl + dl, linear_points, limits[0], "yz")
        right_face  = create_face(linear_points_dl - dl, linear_points, limits[1], "yz")

        front_face  = create_face(linear_points_small, linear_points_small, limits[0] ,"xz")
        back_face   = create_face(linear_points_small, linear_points_small, limits[1], "xz")

    elif method == "random":
        bottom_face = create_random_face(points_per_face,limits, limits[0], "xy")
        top_face = create_random_face(points_per_face,limits, limits[1], "xy")
        left_face = create_random_face(points_per_face,limits, limits[0], "yz")
        right_face = create_random_face(points_per_face,limits, limits[1], "yz")
        front_face = create_random_face(points_per_face,limits, limits[0], "xz")
        back_face = create_random_face(points_per_face,limits, limits[1], "xz")

    
    cube_points = np.r_[bottom_face, top_face, left_face, right_face, front_face, back_face]
    cube_points = np.unique(cube_points, axis=0) # removes duplicate points i.e. in the corners and edges.

    return cube_points


def project_cube_to_sphere(cube, method="even"):
    x = np.copy(cube[:,0])
    y = np.copy(cube[:,1])
    z = np.copy(cube[:,2])
    x2 = x **2
    y2 = y **2
    z2 = z **2
    r = np.sqrt(x2 + y2 + z2)
    if method == "direct":
        x /= r
        y /= r
        z /= r
    # even only works if the cube is 2x2x2, needs some scaling to make it works for all cube sizes.
    elif method == "even":
        x = x * np.sqrt(1 - (y2 + z2) /2 + (y2 * z2) / 3 )
        y = y * np.sqrt(1 - (z2 + x2) /2 + (z2 * x2) / 3 )
        z = z * np.sqrt(1 - (x2 + y2) /2 + (x2 * y2) / 3 )

    sphere = np.c_[x,y,z]
    sphere /= np.linalg.norm(sphere, axis=1)[:,np.newaxis]/np.sqrt(2)
    return sphere

def cartesian_to_latlon(cart):
    #https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # ignores the radius of the spherical coordinates
    def calc_lon(x,y):
        def xy_to_lon(x,y):
            # if x > 0:
            #     return np.arctan2(y, x)
            # if x < 0 and y >= 0:
            #     return np.arctan2(y, x) #+ np.pi
            # if x < 0 and y < 0:
            #     return np.arctan2(y, x) #- np.pi

            # edge cases, rarely happens with floats, but just to cover div by 0.
            if x == 0.0 and y > 0.0:
                return np.pi/2
            if x == 0.0 and y < 0.0:
                return - np.pi/2
            return np.arctan2(y, x)

        lon = np.empty_like(x)
        for i,(xy) in enumerate(zip(x,y)):
            lon[i] = xy_to_lon(*xy)
        return lon

    x = cart[:,0]
    y = cart[:,1]
    z = cart[:,2]

    lat = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    lon = np.rad2deg(calc_lon(x,y))

    return np.c_[lat,lon]

def create_sphere_points(n_points=150, method="grid"):
    # create a 3d cube with points evenly distributed on each face in cartesian coordinates
    cube = create_cube_points(n_points, method=method)
    # project the points to a sphere (needs source to method)
    sphere = project_cube_to_sphere(cube)
    return sphere

def even_points_on_sphere_lat_long(n_points=150):
    sphere = create_sphere_points(n_points)
    # lat lon is then calculated by sperical coordinates and igoring the radius
    lat_lon = cartesian_to_latlon(sphere)
    return lat_lon

def demo(points=2000):
    s = 1.5
    cube = create_cube_points(points, 'grid')
    sphere = project_cube_to_sphere(cube, 'even')
    lat_lon = cartesian_to_latlon(sphere)
    print(lat_lon.shape)

    fig = plt.figure()
    plt.suptitle("Cube and projected Sphere")
    ax=[]
    ax.append(fig.add_subplot(121,projection='3d'))
    ax.append(fig.add_subplot(122,projection='3d'))

    ax[0].scatter(cube[:,0], cube[:,1], cube[:,2], s=1.0)
    ax[0].set_xlim3d([-s, s])
    ax[0].set_ylim3d([-s, s])
    ax[0].set_zlim3d([-s, s])
    ax[0].set_aspect("auto")

    ax[1].scatter(sphere[:,0], sphere[:,1], sphere[:,2], s=1.0)
    ax[1].set_xlim3d([-s, s])
    ax[1].set_ylim3d([-s, s])
    ax[1].set_zlim3d([-s, s])
    ax[1].set_aspect("auto")


    fig2 =plt.subplots(1,1)
    plt.scatter(lat_lon[:,1], lat_lon[:,0], s=2.0)
    plt.title("Lat and Lon of the points")

    plt.show()

def create_sphere_normals(sphere_points):
    normals = sphere_points / np.linalg.norm(sphere_points, axis = 1)[:, np.newaxis]
    return normals

def create_cube_normals(cube_points):
    normals = cube_points.copy().ravel()
    # normals[np.where(np.abs(cube_points[:,0]) < 0.99)[0],:] = 0.0
    normals[np.where(np.abs(cube_points.ravel()) < 1.0)[0]] = 0.0
    # normals[np.where(np.abs(cube_points[:,2]) < 1.0)[0],:] = 0.0
    normals = normals.reshape((-1,3))
    normals = normals / np.linalg.norm(normals, axis = 1)[:,np.newaxis]
    return normals

def create_intersecting_cube_sphere(m, method="grid", offset=[1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(2)]):
    n = m//2
    cube_points = create_cube_points(n,method)
    cube_normals = create_cube_normals(cube_points)
    sphere_points = create_sphere_points(n, method)
    sphere_normals = create_sphere_normals(sphere_points)

    cube_center = np.array([0.0,0.0,0.0])
    sphere_center = np.array(offset)
    cube_face_distances = np.array([1.0,1.0,1.0])

    cube_keep_index = np.where(np.linalg.norm(cube_points + cube_center - sphere_center, axis=1) > np.sqrt(2))
    sphere_keep_index = np.where(np.abs(sphere_points + sphere_center - cube_center) - cube_face_distances > 0.0)[0]

    cube_points = cube_points[cube_keep_index]
    cube_normals = cube_normals[cube_keep_index]

    sphere_points = sphere_points[sphere_keep_index]
    sphere_normals = sphere_normals[sphere_keep_index]

    cube_sphere_points = np.r_[cube_points+cube_center, sphere_points+sphere_center]
    cube_sphere_normals = np.r_[cube_normals, sphere_normals]

    return cube_sphere_points, cube_sphere_normals
    
if __name__ == "__main__":

    method = "random"
    n_points = 10000
    points, normals = create_intersecting_cube_sphere(n_points, method)

    save_o3d_pcd(f"sphere_cube_{method}_{str(n_points)}_00.pcd", points, normals)
    plot_cloud_3d_o3d(points, normals)