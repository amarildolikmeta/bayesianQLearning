import numpy as np
import numpy.linalg as la
from scipy.stats import norm
import matplotlib.pyplot as plt


xmin, xmax = -10, 10
n_prior_particles = 1
n_samples = 10000
min_particles = 20
max_particles = 21

particles = np.random.uniform(xmin, xmax, min_particles)
n_particles = len(particles)
counter = np.ones(n_particles) * n_prior_particles / n_particles
weights = counter / np.sum(counter)

print('Sarting with %s effective PRIOR particles' % sum(counter))

def mean(particles, weights):
    return np.sum(particles * weights)

def variance(particles, weights):
    return np.sum(weights * (particles - mean(particles, weights)) ** 2)

mu = 5.
sigma = 1.

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

    particles = np.append(particles, z)
    n_particles = len(particles)
    counter = np.append(counter, 1)
    weights = counter / np.sum(counter)

    if n_particles == max_particles:
        #Subsample
        particles = np.random.choice(particles, min_particles, p=weights, replace=False)
        n_particles = len(particles)
        counter = np.ones(min_particles) * np.sum(counter) / min_particles
        weights = counter / np.sum(counter)
        print('SUBSAMPLING, now %s effective POSTERIOR particles' % sum(counter))



    line1.set_ydata(weights)
    line1.set_xdata(particles)
    fig.canvas.draw()
    fig.canvas.flush_events()

    #print(variance(particles, weights))
