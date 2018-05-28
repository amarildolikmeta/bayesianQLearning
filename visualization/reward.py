import numpy as np
import numpy.linalg as la
from scipy.stats import norm
import matplotlib.pyplot as plt

gamma=0.99
xmin, xmax = -10, 10
n_prior_particles = 1
nr_tries=1000
n_timesteps = 1000
min_particles = 20
max_particles = 21
NUM_STATES=2
NUM_ACTION=2

def mean(particles, weights):
    return np.sum(particles * weights)

def variance(particles, weights):
    return np.sum(weights * (particles - mean(particles, weights)) ** 2)

mu = 5.
sigma = 1.
'''
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
xs = np.linspace(xmin, xmax, 100)
line1, = ax.plot([], [], linestyle='None', marker='o')
linetrue, = ax.plot(xs, [norm.pdf((x - mu)/sigma) for x in xs], 'g')
linetrue, = ax.plot(xs, [norm.pdf((x + mu)/sigma) for x in xs], 'y')

ax.set_ylim(0, 1)
particles=np.zeros(nr_tries)
for i in range(nr_tries):
    sum=0
    for j in range(n_timesteps):
        z = np.random.normal(mu, sigma, 1)
        sum+=z*np.power(gamma, j)
    particles[i]=sum
print("Mean :%f; STD:%d" %(np.mean(particles), np.std(particles)))
line1.set_xdata(particles)
line1.set_ydata(np.zeros(nr_tries))
fig.canvas.draw()
fig.canvas.flush_events()'''
sum=0
sum1=0
for j in range(n_timesteps):
    sum+=np.power(gamma, j)
    sum1+=np.power(gamma, 2*j)
print("sum :%f; sum2:%d" %(sum, sum1))

    
