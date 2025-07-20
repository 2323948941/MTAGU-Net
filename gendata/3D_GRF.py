import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fftind_3d(N):
    """Generate the 3D grid of frequency indices for FFT."""
    kx = np.fft.fftfreq(N, 1.0)  # Frequency indices for x-direction
    ky = np.fft.fftfreq(N, 1.0)  # Frequency indices for y-direction
    kz = np.fft.fftfreq(N, 1.0)  # Frequency indices for z-direction

    # Generate the 3D grid for kx, ky, kz
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    # Return the grid of indices
    return KX, KY, KZ


def gaussian_random_field_3d(alpha=10.0,  # smooth factor
                             size=32,  # size of the field
                             mode='bound',  # 'random' or 'bound'
                             set_1=1.0,  # if 'random', mean; else if 'bound', lower bound
                             set_2=10000.0):  # if 'random', standard derivation; else if 'bound', upper bound
    # Get the 3D momentum indices
    k_idx = fftind_3d(size)

    # Compute the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power(k_idx[0] ** 2 + k_idx[1] ** 2 + k_idx[2] ** 2 + 1e-10, -alpha / 4.0)
    amplitude[0, 0, 0] = 0  # Prevent division by zero at the origin

    # Generate complex Gaussian random noise
    noise = np.random.normal(size=(size, size, size)) + 1j * np.random.normal(size=(size, size, size))

    # Perform the inverse Fourier transform to get the Gaussian random field in real space
    gfield = np.fft.ifftn(noise * amplitude).real

    # Normalize the field to have zero mean and unit standard deviation
    gfield = gfield - np.mean(gfield)
    gfield = gfield / np.std(gfield)

    # Scale and shift the field according to the mode
    if mode == 'random':
        set_mean = set_1
        set_std = set_2
        gfield = gfield * set_std
        gfield = gfield + set_mean
    elif mode == 'bound':
        set_lower = set_1
        set_upper = set_2
        g_max = np.max(gfield)
        g_min = np.min(gfield)
        gfield = (set_upper - set_lower) / (g_max - g_min) * gfield + \
                 (set_lower * g_max - set_upper * g_min) / (g_max - g_min)
    else:
        raise KeyError("mode must be 'random' or 'bound'")

    return gfield


def plot_3d_surface(gfield, threshold=0.0):
    """ Visualizes a 3D surface of the Gaussian Random Field using matplotlib """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of the points where gfield is above a certain threshold
    x, y, z = np.where(gfield > threshold)

    # Scatter plot of the coordinates
    ax.scatter(x, y, z, c=gfield[x, y, z], cmap='jet', s=1)

    ax.set_title('3D Visualization of Gaussian Random Field')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def main():
    import scipy.io
    size = 22  # Set the field size (smaller size for quick visualization)
    num=10  #Set the sample size
    data=np.zeros((size,size,size,num))
    for i in range(num):
        example = gaussian_random_field_3d(size=size)
        data[:,:,:,i] = example.real
    scipy.io.savemat('multiple_arrays.mat', {
        'data': data
    })


    # Plot 3D surface visualization
    plot_3d_surface(example)


if __name__ == '__main__':
    main()
