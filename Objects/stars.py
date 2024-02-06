import random
import numpy as np


class Star:
    def __init__(self, name, magnitude, location, color):
        self.name = name
        self.magnitude = magnitude  # Brightness of the star
        self.location = location  # x, y, z coordinates of the star
        self.color = color  # Color of the star

    def __repr__(self):
        return f"{self.name}: Magnitude {self.magnitude}, Location {self.location}"


stars = [
    Star(name="Star1", magnitude=5, location=[2e11, 0, 0], color=(255, 255, 255)),
    Star(name="Star2", magnitude=4, location=[3e11, 0, 0], color=(255, 255, 255)),
    Star(name="star3", magnitude=3,location=[1e11,0,0],color=(255, 255, 0)),
    Star(name="star3", magnitude=3,location=[3e11,3e11,0],color=(0, 0, 255))
    # Add more stars as needed
]

# def generate_star_positions(num_stars, min_distance, max_distance):
#     stars = []
#     for i in range(num_stars):
#         r = random.uniform(min_distance, max_distance)  # distance from the Sun
#         theta = random.uniform(0, 2 * np.pi)  # azimuthal angle
#         phi = random.uniform(0, np.pi)  # polar angle
#
#         # Convert spherical to Cartesian coordinates
#         x = r * np.sin(phi) * np.cos(theta)
#         y = r * np.sin(phi) * np.sin(theta)
#         z = r * np.cos(phi)
#
#         star = Star(f"Star{i}", 1, [x, y, z])
#         stars.append(star)
#     return stars
#
# # Generate stars with distances ranging from 1 light year to 100 light years from the Sun
# # 1 light year is approximately 9.461e15 meters
# stars = generate_star_positions(100, 9.461e15, 9.461e16)






"""
class Star:
    def __init__(self, name, magnitude, location):
        self.name = name
        self.magnitude = magnitude
        self.location = location

    def __repr__(self):
        return f"{self.name}: Magnitude {self.magnitude}, Location {self.location}"

stars = [
    Star("Sirius", -1.46, [8.6e16, 2.0e17, 0]),
    Star("Canopus", -0.72, [1.1e17, 3.0e17, 0]),
    Star("Arcturus", -0.04, [3.6e16, 1.1e17, 0]),
    Star("Alpha Centauri", -0.01, [4.1e16, 1.3e17, 0]),
    Star("Vega", 0.03, [2.4e16, 7.8e16, 0]),
    Star("Capella", 0.08, [5.0e16, 1.6e17, 0]),
    Star("Rigel", 0.12, [6.7e16, 2.1e17, 0]),
    Star("Procyon", 0.34, [7.3e16, 2.3e17, 0]),
    Star("Betelgeuse", 0.45, [8.0e16, 2.5e17, 0]),
    Star("Achernar", 0.46, [8.8e16, 2.7e17, 0]),
    # ... Add more stars as needed
]

  """

