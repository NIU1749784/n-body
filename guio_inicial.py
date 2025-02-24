# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:08:41 2025

@author: carme
"""

import numpy as np
import matplotlib.pyplot as plt

positions = np.array([[0,0],[0,4.5e10],[0, -4.5e10]])
velocities = np.array([[5e02,0],[3e04,0],[-3e04,0]])
masses = np.array([5.97e24, 1.989e30, 1.989e30])
dt = 5000
num_steps = 2000


num_bodies = len(masses)
plt.figure()
colors = "rgb"
for t in range(num_steps):
    forces = np.zeros((num_bodies,2))
    for i in range(num_bodies):
        for j in range(num_bodies):
            if j != i:
                forces[i] += force(masses[i], masses[j], positions[i], positions[j])
    for i in range(num_bodies):
        acceleration = forces[i]/masses[i]
        velocities[i] += acceleration*dt
        positions[i] += velocities[i]*dt
        color = colors[i % len(colors)]
        plt.plot(positions[i,1], positions[i,0], color+".")

plt.axis("equal")
plt.grid()

class Body:
    G = 6.67e-11

    def __init__(self, mass, initial_position, initial_velocity):
        self.mass = mass
        self.position = np.array(initial_position)
        self.velocity = np.array(initial_velocity)

    def update(self, force, dt):
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

    def total_force(self, other_bodies):
        force = np.zeros(2)
        for body in other_bodies:
            force += self._force(body)
        return force

    def _force(self, another_body):
        distance12 = self._distance_to(another_body)
        magnitude = self.mass * another_body.mass * Body.G / distance12**2
        direction = (another_body.position - self.position) / distance12
        return magnitude * direction

    def _distance(self, p, q):
        return np.sqrt(np.square(p - q).sum())

    def _distance_to(self, another_body):
        return self._distance(self.position, another_body.position)


# class Star(Body):
#     def __init__(self, mass, initial_position, initial_velocity, percent_mass):
#         super().__init__(mass, initial_position, initial_velocity)
#         self.percent_mass = percent_mass

#     def update(self, force, dt):
#         super().update(force, dt)
#         u = 2 * (np.random.rand() - 0.5)  # Uniform in [-1,1]
#         self.mass = self.mass * (1 + (self.percent_mass / 100.0) * u)


class Simulator:
    def __init__(self, bodies, seed=1234):
        self.bodies = bodies
        self.num_bodies = len(bodies)
        self.colors = "rgbymk"
        np.random.seed(seed)
    
    def simulate(self, dt, num_steps):
        plt.figure()
        for _ in range(num_steps):
            self._plot_bodies()
            forces = self._compute_forces()
            self._update_bodies(forces, dt)
    
    def _plot_bodies(self):
        for i in range(self.num_bodies):
            b = self.bodies[i]
            color = self.colors[i % len(self.colors)]
            plt.plot(b.position[1], b.position[0], color + '.')
    
    def _compute_forces(self):
        forces = []
        for i in range(self.num_bodies):
            body = self.bodies[i]
            other_bodies = self.bodies[:i] + self.bodies[i+1:]
            forces.append(body.total_force(other_bodies))
        return np.array(forces)
    
    def _update_bodies(self, forces, dt):
        for i in range(self.num_bodies):
            self.bodies[i].update(forces[i], dt)
            
class Universe:
    def __init__(self, bodies, radius, name):
        self._bodies = bodies
        self._radius = radius
    @property
    def bodies(self):
        return self._bodies
    def radius(self):
        return self.radius

    @classmethod
    def from_file(cls, fname):
        bodies = []
        with open(fname, 'r') as f:
            num_bodies = int(f.readline())
            for _ in range(num_bodies):
                y = f.readline()
                m, px, py, vx, vy = [float(z) for z in y.strip().split(' ')]
                bodies.append(Body(m, [px, py], [vx, vy]))
        return cls(bodies)
    
import numpy as np
import pygame
import random

class PygameExample:
    def __init__(self, window_size=600):
        self.window_size = window_size  # pixels
        self.space_radius = 1e12  # meters of half window
        self.factor = self.window_size / 2 / self.space_radius  # pixels/meter
        
        # Initial position and velocity of a body in space
        self.position = np.array([
            random.uniform(self.space_radius / 4, self.space_radius / 2),
            random.uniform(self.space_radius / 4, self.space_radius / 2)
        ])  # meters
        self.velocity = 2e-3 * np.array([-self.position[1], self.position[0]])
        # meters / second, perpendicular to the vector from origin to the initial position

    def animate(self, time_step, trace=False):
        pygame.init()
        self.screen = pygame.display.set_mode([
            self.window_size, self.window_size
        ])
        pygame.display.set_caption(f'Example, timestep {time_step}')
        
        running = True  # Run until the user asks to quit
        color_background = (128, 128, 128)
        color_body = (0, 0, 0)
        color_trace = (192, 192, 192)
        
        self.screen.fill(color_background)
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if trace:
                self._draw(self.position, color_trace)
            
            self._update(time_step)
            self._draw(self.position, color_body)
            pygame.display.flip()
        
        pygame.quit()
    
    def _draw(self, position_space, color, size=5):
        position_pixels = self.factor * position_space + self.window_size / 2.
        pygame.draw.circle(self.screen, color, position_pixels, size)
    
    def _update(self, time_step):
        self.position += self.velocity * time_step
        self.velocity = 2e-3 * np.array([-self.position[1], self.position[0]])
        # The velocity is always perpendicular to the position vector => trajectory is a circle

if __name__ == '__main__':
    example = PygameExample()
    time_step = 1  # seconds
    example.animate(time_step, trace=True)
    
if __name__ == "__main__":
    bodies = [
        Body(5.97e24, [0, 0], [5e2, 0]),
        Star(1.989e30, [0, 4.5e10], [3.0e4, 0], 1.0),
        Star(1.989e30, [0, -4.5e10], [-3.0e4, 0], 1.0)
    ]
    dt = 5000
    num_steps = 2000
    sim = Simulator(bodies)
    sim.simulate(dt, num_steps)


u = Universe()
sim = Simluator(u)
b = sim.u.bodies()  # metodo body es usado para leer/obtener (getter) el valor
                    # de la variable priv. _bodies





















