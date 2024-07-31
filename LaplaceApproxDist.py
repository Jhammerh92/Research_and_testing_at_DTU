import numpy as np
import matplotlib.pyplot as plt

def reverse_fourier_transform(fft_result):
    # Reverse Fourier Transform
    ifft_result = np.fft.ifft2(np.fft.ifftshift(fft_result))
    ifft_result = np.real(ifft_result)  # Take the real part to remove potential imaginary artifacts

    return ifft_result

def crop_fft(fft_result, x):
    # Get the shape of the FFT result
    M, N = fft_result.shape

    # Calculate the indices to crop
    start_idx = M // 2 - x // 2
    end_idx = start_idx + x

    # Crop the FFT result
    cropped_fft = fft_result[start_idx:end_idx, start_idx:end_idx]

    return cropped_fft

def add_zero_padding(cropped_fft, y):
    # Get the shape of the cropped FFT
    M, N = cropped_fft.shape

    # Calculate the padding needed
    pad_width = ((y - M) // 2, (y - N) // 2)

    # Pad the cropped FFT with zeros
    padded_fft = np.pad(cropped_fft, pad_width, mode='edge',)

    return padded_fft


# sample_size=1000
grid_size=100
sigma=0.02
# Generate random 2D points
# points = np.random.uniform(-10,10, [sample_size,2])
# points = np.random.normal(-2,2, [sample_size,2])
max = [-2.5,-15] 
min = [-3.5, -16]

points = np.load("test_data/HAP_sweep_ds.npy")
points = points[points[:,2] < -5]
points = points[::10,:2]
# max = [np.max(points[:,0]), np.max(points[:,1])]
# min = [np.min(points[:,0]), np.min(points[:,1])]


# Create a grid
x = np.linspace(min[0], max[0], grid_size)
y = np.linspace(min[1], max[1], grid_size)
X, Y = np.meshgrid(x, y)

# Compute KDE (Kernel Density Estimation)
# kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))  # Gaussian kernel
density = np.zeros_like(X)
# X_diff = X - points[:,0]
# Y_diff = X - points[:,0]


# [density:= density + (np.exp(-((X - point[0])**2 + (Y - point[1])**2) / (2 * sigma**2))) for point in points]

def gaussian_kernel(grid, point, variance ):
    # h = 2*variance
    X,Y = grid
    density = np.exp(-((X - point[0])**2 + (Y - point[1])**2) / (2 * variance))/ (np.sqrt(2*np.pi))
    return density

for point in points:
    density += gaussian_kernel((X,Y), point, sigma**2)


# density = np.log(density)
# density /= np.sum(density)


# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(density, cmap='hot', origin='lower', extent=[min[0], max[0], min[1], max[1]])
plt.colorbar(label='Density')
# plt.scatter(points[:,0], points[:,1], alpha=0.1, color='b')
plt.title('Heatmap of Sampled Points')
plt.xlabel('X')
plt.ylabel('Y')

# Perform Fourier Transformation
fft_result = np.fft.fft2(density)
# crop fft 
fft_result = np.fft.fftshift(fft_result)  # Shift zero frequency component to the center
reversed_density = reverse_fourier_transform(fft_result)

fft_cropped = crop_fft(fft_result, grid_size//2)
fft_cropped = add_zero_padding(fft_cropped, grid_size)
# fft_cropped = np.fft.fftshift(fft_cropped)  # Shift zero frequency component to the center



reversed_density_cropped = reverse_fourier_transform(fft_cropped)




# Plot Fourier Transformation
plt.figure(figsize=(8, 6))
plt.imshow(np.log(np.abs(fft_result)), cmap='hot', origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi])
plt.colorbar(label='Magnitude')
plt.title('Fourier Transformation')
plt.xlabel('Frequency (kx)')
plt.ylabel('Frequency (ky)')

plt.figure(figsize=(8, 6))
plt.imshow(np.log(np.abs(fft_cropped)), cmap='hot', origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi])
plt.colorbar(label='Magnitude')
plt.title('Fourier Transformation')
plt.xlabel('Frequency (kx)')
plt.ylabel('Frequency (ky)')

plt.figure(figsize=(8, 6))
plt.imshow(reversed_density, cmap='hot', origin='lower', extent=[min[0], max[0], min[1], max[1]])
plt.colorbar(label='Density')
plt.title('Reversed Heatmap')
plt.xlabel('X')
plt.ylabel('Y')


plt.figure(figsize=(8, 6))
plt.imshow(reversed_density_cropped, cmap='hot', origin='lower', extent=[min[0], max[0], min[1], max[1]])
plt.colorbar(label='Density')
plt.title('Cropped Reversed Heatmap')
plt.xlabel('X')
plt.ylabel('Y')

# plt.figure(figsize=(8, 6))
# plt.imshow(phase, cmap='hsv', origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi])
# plt.colorbar(label='Phase')
# plt.title('Phase of Fourier Coefficients')
# plt.xlabel('Frequency (kx)')
# plt.ylabel('Frequency (ky)')

plt.show()

