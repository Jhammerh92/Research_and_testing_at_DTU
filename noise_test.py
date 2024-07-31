import numpy as np
import matplotlib.pyplot as plt



t = np.linspace(np.deg2rad(40/2),0,10000)
sigma = 0.00
amp =  np.sin(t*30)**2* (1-np.exp(-3*t)) *0.005
noise = np.random.normal(0.0,amp + sigma)

fig = plt.figure()
plt.scatter(np.rad2deg(t),noise, s=2, alpha=0.1, color="C0")
plt.plot(np.rad2deg(t),amp+ sigma, color="C1", lw=2)
plt.title('Radial synthetic error')
plt.ylabel(r'$\theta$ Dependent Std $\sigma, \, [m]$')
plt.xlabel(r'Angle from center $\theta$ $[{}^\circ]$')
plt.show()
