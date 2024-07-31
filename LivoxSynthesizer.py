import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as path

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def noise_profile_radial(t, random_sigma= 0.01, amp = 0.02):
    return np.sin(t*30)**2* (1-np.exp(-3*t)) * amp + random_sigma

#TODO def noise_profile_azimuth(t, random_sigma= 0.01, amp = 0.02):
    # return np.sin(t*15)**2* (1-np.exp(-3*t)) * amp + random_sigma


def livox_scan_synthetic_plane(t=1, depth=1, sigma1=0.005, sigma2=0.005):
    def vec_from_angle(ang):
        x = np.cos(ang)
        y = np.sin(ang)
        return np.array([x,y])
    
    def shooting_vec_from_angles_depth(ang,theta):
        z = np.cos(ang) * np.sin(theta)     # old x, the actual scanner has x as depth and z veritcal
        y = -np.sin(ang) * np.sin(theta)    # old y
        x = np.cos(theta)                   # old z
        vec = np.array([x, y, z])
        vec = vec/np.linalg.norm(vec)
        return vec
    
    def add_random_noise(vec,mu=0.0, sigma=0.02):
        nvec = vec/np.linalg.norm(vec)
        vec += nvec * np.random.normal(mu, sigma)
        return vec
    
    def add_systemic_noise(vec, theta, amp = 0.05):
        nvec = vec/np.linalg.norm(vec)
        noise = np.random.normal(0.0, np.sin(theta*15)**2* (1-np.exp(-3*theta)))
        vec += nvec * noise
        return vec
    
    def add_noise(vec, theta, random_sigma= 0.01, amp = 0.02):
        nvec = vec/np.linalg.norm(vec)
        theta = np.abs(theta)
        # systemic_sigma = np.sin(theta*15)**2* (1-np.exp(-3*theta)) * amp
        # sigma = random_sigma + systemic_sigma
        sigma = noise_profile_radial(theta, random_sigma, amp)
        noise = np.random.normal(0.0, sigma)
        vec += nvec * noise
        return vec
    


        

    N = 100_000 # 10k points pr 0.1 sec so 100k pr sec
    # alfa = 11.1 #  5  4.95
    # beta = 73 # 23
    # angles = np.linspace(0, np.pi * alfa * sec, int(N*sec))
    # theta = np.linspace(0, np.pi * beta * sec, int(N*sec))

    w = 3*2*np.pi/0.02
    alfa = 1/6.5536 #  5  4.95 # RT
    beta = 1 # 23 # T
    rho = np.linspace(0, w * alfa * t, int(N*t)) # rot 
    t_theta = np.linspace(0, w * beta * t, int(N*t)) # timing of theta angles

    theta = np.cos(t_theta) * np.deg2rad(40/2) # radial angle from center, max radius is 20 deg from center for the mid40
    # theta = np.cos(t_theta) * np.deg2rad(60)
    

    

    # pos = np.array([0.0,0.0])
    shooting_vectors = []

    for i, (ang,thet) in enumerate(zip(rho, theta)):
        # vec = vec_from_angle(ang)  * np.cos(thet)
        vec = shooting_vec_from_angles_depth(ang, thet)
        vec = (vec / vec[0]) * depth
        if np.linalg.norm(vec) > 10:
            print("help")

        # vec = add_systemic_noise(vec,thet)
        # vec = add_random_noise(vec)
        vec = add_noise(vec,thet,sigma1, sigma2)
        shooting_vectors.append(vec.copy()) 

    coords = np.asarray(shooting_vectors)


    return coords,theta

def load_synthetic(folder):

    points = np.load(os.path.join(folder,'scan_points.npy'))
    std_profile_radial = np.load(os.path.join(folder,'std_profile.npy'))
    ## TODO std_profile_azimuth = np.load(os.path.join(folder,'std_profile.npy')) 
    return points, std_profile_radial


if __name__ == "__main__":

    SCAN_TIME = 0.1
    SCAN_PLANE_DEPTH = 3
    SYS_SIGMA = 0.00071 # sqrt(0.01)
    RAND_SIGMA = 0.0071
    # points,theta = livox_scan_synthetic_plane(0.025, 1)
    points, theta = livox_scan_synthetic_plane(SCAN_TIME, SCAN_PLANE_DEPTH, sigma1=SYS_SIGMA, sigma2=RAND_SIGMA)
    t = np.linspace(0, np.max(theta), 100)
    std_profile = noise_profile_radial(t, RAND_SIGMA, SYS_SIGMA)
    std_profile = np.c_[t, std_profile]


    fig = plt.figure()
    plt.plot(np.linspace(0, SCAN_TIME,len(theta)), theta)
    # plt.scatter(points[:,0], points[:,2])
    # plt.axis('equal')

    fig = plt.figure()
    plt.scatter(points[:,1], points[:,2], s=5)
    plt.axis('equal')

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # ax.scatter(points[:,0], points[:,1],points[:,2], alpha= 1.0, s=2, c=points[:,2], cmap='bwr')
    ax.scatter(points[:,0], points[:,1],points[:,2], alpha= 0.5, s=2)
    # ax.scatter(points2[:,0], points2[:,1],points2[:,2], alpha= 1.0, s=2)
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
  



    success = False
    i = 0
    # while not success:
    # try:
    #     path_ = f'livox40_synthetic_{SCAN_TIME}s{SCAN_PLANE_DEPTH}m_{i}'
    #     os.makedirs(path_)
    #     success = True
    # except:
    #     i += 1

    # np.save(path.join(path_, 'scan_points'), points)
    # np.save(path.join(path_, 'std_profile'), std_profile)


    plt.show()