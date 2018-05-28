import numpy as np
import numpy.linalg as la
from scipy.stats import norm
import matplotlib.pyplot as plt

xmin, xmax = -10, 10
n_particles = 100

particles = np.random.uniform(xmin, xmax, n_particles)
weights = np.ones(n_particles) / n_particles

bandwidth = 1.
def kernel(z, particles):
    return norm.pdf((z - particles) / bandwidth)

def pdf(z, particles, weights):
    return np.sum(norm.pdf((z - particles) / bandwidth) * weights)

#def resample()

def ess(weights):
    return la.norm(weights, 1) ** 2 / la.norm(weights, 2) ** 2

def mean(particles, weights):
    return np.sum(particles * weights)

def variance(particles, weights):
    return np.sum(weights * (particles - mean(particles, weights)) ** 2)

n_samples = 10000
mu = 5.
sigma = 1.
ess_limit = n_particles * 0.1

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
xs = np.linspace(xmin, xmax, 100)

linetrue, = ax.plot(xs, [norm.pdf((x - mu)/sigma) for x in xs], 'g')
linetrue, = ax.plot(xs, [norm.pdf((x + mu)/sigma) for x in xs], 'y')

line1, = ax.plot(particles, weights, linestyle='None', marker='o')
ax.set_ylim(0, 1)

for i in range(n_samples):
    if np.random.randint(0, 2, 1) % 2 == 1:
        z = np.random.normal(mu, sigma, 1)
    else:
        z = np.random.normal(-mu, sigma, 1)

    likelihood = kernel(z, particles)

    weights = weights * likelihood + 1e-300
    weights /= np.sum(weights)

    line1.set_ydata(weights)
    fig.canvas.draw()
    fig.canvas.flush_events()

    print(variance(particles, weights))
