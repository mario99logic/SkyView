import random
import numpy as np
from tranformation.transformations import dec_to_radians
from tranformation.transformations import hms_to_radians

class Star:

    def __init__(self, name, magnitude, location, color,ra,dec):
        self.name = name
        self.magnitude = magnitude  # Brightness of the star
        self.location = location  # x, y, z coordinates of the star
        self.color = color  # Color of the star
        self.ra = ra
        self.dec = dec

    def __repr__(self):
        return f"{self.name}: Magnitude {self.magnitude}, Location {self.location}"


stars = [
    Star(name="Mimosa",
         magnitude=-0.5,
         location=[-2.5e17, 0, 0],
         color=(0, 0, 120),
         ra=hms_to_radians(12, 47, 43.26877),
         dec=dec_to_radians(50, 41, 19)),
    Star(name="Vega",
         magnitude=0.03,
         location=[2.5e17, -1.2e17, 0],
         color=(244, 245, 255),
         ra=hms_to_radians(18, 36, 56),
         dec=dec_to_radians(38, 40, 1)),
    Star(name="Altair",
         magnitude=0.77,
         location=[1.7e17, -1e17, 0],
         color=(255, 255, 255),
         ra=hms_to_radians(19, 50, 46.9),
         dec=dec_to_radians(8, 52, 5)),
    # Add more stars as needed
    Star(name="Spica",
         magnitude=0.98,
         location=[2.6e17, -1.5e17, 0],
         color=(0, 0, 255),
         ra=hms_to_radians(13, 25, 11.5),
         dec=dec_to_radians(-11, 9, 40)),
    Star(name="Rigel",
         magnitude=0.12,
         location=[7.7e16, 1.2e16, 0],
         color=(180, 180, 255),
         ra=hms_to_radians(5, 14, 32),
         dec=dec_to_radians(-8, 12, 6)),
    Star(name="Pollux",
         magnitude=1.14,
         location=[1.2e17, 4.4e16, 0],
         color=(255, 255, 0),
         ra=hms_to_radians(7, 45, 18.9),
         dec=dec_to_radians(28, 1, 34.3)),
    Star(name="Achernar",
         magnitude=0.45,
         location=[3.2e17, -7e17, 0],
         color=(190, 190, 255),
         ra=hms_to_radians(1, 37, 42.8),
         dec=dec_to_radians(-57, 14, 12)),
    Star(name="Aldebaran",
         magnitude=0.85,
         location=[5.8e16, 1.2e17, 0],
         color=(255, 69, 0),
         ra=hms_to_radians(4, 35, 55),
         dec=dec_to_radians(16, 30, 33)),
    Star(name="Fomalhaut",
         magnitude=1.16,
         location=[7.7e16, -2.5e17, 0],
         color=(255, 255, 255),
         ra=hms_to_radians(22, 57, 39),
         dec=dec_to_radians(-29, 37, 20)),
    Star(name="Capella",
         magnitude=0.08,
         location=[5e16, 5e16, 0],
         color=(255, 255, 0),
         ra=hms_to_radians(5, 16, 41),
         dec=dec_to_radians(45, 55, 50)),
    Star(name="Deneb",
         magnitude=1.25,
         location=[3.2e17, -5.3e17, 0],
         color=(160, 160, 255),
         ra=hms_to_radians(20, 41, 25.9),
         dec=dec_to_radians(45, 16, 49)),
    Star(name="Antares",
         magnitude=1.06,
         location=[5.5e16, -2.9e16, 0],
         color=(190, 0, 0),
         ra=hms_to_radians(16, 29, 24.4),
         dec=dec_to_radians(-26, 25, 55)),
    Star(name="Sirius",
         magnitude=-1.46,
         location=[8.611e16, 0, 0],
         color=(240, 240, 240),
         ra=hms_to_radians(6, 45, 8.9),
         dec=dec_to_radians(-16, 42, 58)),
    Star(name="Arcturus",
         magnitude=-0.05,
         location=[-3.7e17, 2.1e17, 0],
         color=(255, 90, 0),
         ra=hms_to_radians(14, 15, 39.7),
         dec=dec_to_radians(19, 10, 57)),
    Star(name="Canopus",
         magnitude=-0.74,
         location=[-5.1e17, 1.9e17, 0],
         color=(190, 190, 190),
         ra=hms_to_radians(6, 23, 57),
         dec=dec_to_radians(-52, 41, 45)),
    Star(name="Regulus",
         magnitude=1.35,
         location=[7.9e16, 0, 0],
         color=(0, 0, 255),
         ra=hms_to_radians(10, 8, 22),
         dec=dec_to_radians(11, 58, 1)),
    Star(name="Proxima Centauri",
         magnitude=11.13,
         location=[-3.710e16, 0, 0],
         color=(255, 0, 0),
         ra=hms_to_radians(14, 29, 43),
         dec=dec_to_radians(-62, 40, 46)),
    Star(name="Alpha Centauri",
         magnitude=-0.27,
         location=[4.367e16, 0, 0],
         color=(255, 255, 0),
         ra=hms_to_radians(14, 39, 37),
         dec=dec_to_radians(-60, 50, 2)),
    Star(name="Betelgeuse",
         magnitude=0.42,
         location=[1.642e17, 0, 0],
         color=(255, 0, 0),
         ra=hms_to_radians(5, 55, 10),
         dec=dec_to_radians(7, 24, 26)),
    # Star(name="Orion Nebula",
    #      magnitude=4.0,
    #      location=[4e17, 0, 0],
    #      color=(0, 0, 255),
    #      ra=hms_to_radians(5, 35, 17),
    #      dec=dec_to_radians(-5, 23, 28)),

    Star(
        name="Pistol Star",
        magnitude=-10.0,
        location=[-1e17, 3e17, 0],  # Fictional representation for illustrative purposes
        color=(255, 189, 111),
        ra=hms_to_radians(17, 46, 15),
        dec=dec_to_radians(-28, 50, 4)),
    Star(
        name="BAT99-98",
        magnitude=-6.5,
        location=[3e17, 4e17, 0],  # Fictional representation for illustrative purposes
        color=(255, 215, 0),
        ra=hms_to_radians(5,38,39),# Color representation for a Wolf-Rayet star
        dec=dec_to_radians(-69,6,21)
    ),

    Star(
         name="R136a1",  # R136a1 is among the most massive stars known
         location=[5e17, -1e18, 0],  # Fictional representation for illustrative purposes
         magnitude=-7.0,  # Estimated visual magnitude, highly luminous
         color=(255, 255, 255),  # Color representation for a very hot, luminous star
         ra=hms_to_radians(5,38,42),
         dec=dec_to_radians(-69,6,3)
    ),

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






  """

