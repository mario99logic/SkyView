import math
import numpy as np

G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2

def distance(object1, object2):     # calculate distance between two objects
    dx = object2.location[0] - object1.location[0]
    dy = object2.location[1] - object1.location[1]
    dz = object2.location[2] - object1.location[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)

def gravitational_force(object1, object2):    # newton law function (page 4)
    """Update gravitational_force of two bodies."""
    r = distance(object1, object2)
    force_magnitude = G * object1.mass * object2.mass / r**2
    # This gives the magnitude. Direction has to be calculated based on positions.
    return force_magnitude

def update_velocity(body, force, dt):
    """Update velocity of a body given a force and time step."""
    acceleration = force / body.mass  # a = F/m
    new_velocity = np.array(body.speed) + acceleration * dt  # v = u + at
    return list(new_velocity)

def update_position(body, dt):
    """Update position of a body given its velocity and a time step."""
    new_position = np.array(body.location) + np.array(body.speed) * dt
    return list(new_position)

""" This function is for calculation the gravitational force of the planet with all other celestial bodies"""
def compute_net_gravitational_force(planet, celestial_bodies):
    net_force = np.array([0.0, 0.0, 0.0])
    for body in celestial_bodies:
        if body != planet:
            r = distance(planet, body)
            force_magnitude = gravitational_force(planet, body)
            direction = (np.array(body.location) - np.array(planet.location)) / r
            force_vector = force_magnitude * direction
            net_force += force_vector
    return net_force

""" These functions are for the time function """

def simulate_motion(celestial_bodies, dt, total_time):
    time_elapsed = 0
    while time_elapsed < total_time:
        for planet in celestial_bodies:
            if planet.object_type == "Planet":  # Only update planets
                net_force = compute_net_gravitational_force(planet, celestial_bodies)
                planet.speed = update_velocity(planet, net_force, dt)
                planet.location = update_position(planet, dt)
        time_elapsed += dt



