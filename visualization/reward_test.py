import numpy as np
import numpy.linalg as la
from scipy.stats import norm
import matplotlib.pyplot as plt

gamma=0.99
xmin, xmax = 0, 1000
n_prior_particles = 1
n_timesteps = 10000
min_particles = 20
max_particles = 50
particles = np.random.uniform(xmin, xmax, min_particles)
n_particles = len(particles)
counter = np.ones(n_particles) * n_prior_particles / n_particles
weights = counter / np.sum(counter)
print('Sarting with %s effective PRIOR particles' % sum(counter))

def mean(particles, weights):
    return np.sum(particles * weights)

def variance(particles, weights):
    return np.sum(weights * (particles - mean(particles, weights)) ** 2)

#discounted sum of rewards
mu = 495
sigma = 49.25

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
xs = np.linspace(xmin, xmax, 1000)

linetrue, = ax.plot(xs, [norm.pdf((x - mu)/sigma) for x in xs], 'g')


line1, = ax.plot(particles, weights, linestyle='None', marker='o')
ax.set_ylim(0, 1)
reward_mu=5
reward_sigma=1
for i in range(n_timesteps):
    reward = np.random.normal(reward_mu, reward_sigma)
    old_particles=np.zeros(len(particles))
    old_particles[:]=particles
    old_counter=np.zeros(len(counter))
    old_counter[:]=counter
    new_particles=np.zeros(len(particles))
    new_particles[:]=particles
    new_counter=np.zeros(len(counter))
    new_counter[:]=counter
    new_weights=new_counter/np.sum(new_counter)
    ##sample a particle and its count
    index=np.random.choice(a=range(len(new_particles)), p=new_weights)
    z=new_particles[index]
    count=new_counter[index]
    target=reward+gamma*z
    particles= np.append(old_particles, target)
    counter = np.append(old_counter, count)
    weights = counter / np.sum(counter)
    #Subsample
    particles = np.random.choice(particles, min_particles, p=weights, replace=False)
    counter = np.ones(min_particles) * np.sum(counter) / len(counter)
    weights = counter / np.sum(counter)
    '''
    if n_particles == max_particles:
        #Subsample
        particles = np.random.choice(particles, min_particles, p=weights, replace=False)
        n_particles = len(particles)
        counter = np.ones(min_particles) * np.sum(counter) / min_particles
        weights = counter / np.sum(counter)
        print('SUBSAMPLING, now %s effective POSTERIOR particles' % sum(counter))'''

    line1.set_ydata(weights)
    line1.set_xdata(particles)
    fig.canvas.draw()
    fig.canvas.flush_events()

    #print(variance(particles, weights))
