import os
import pickle

import pygame
import datetime
from datetime import timezone
from astropy.coordinates import EarthLocation, AltAz, get_body, solar_system_ephemeris
from astropy.time import Time
import astropy.units as u
from src.Objects.stars import stars
from astropy.coordinates import SkyCoord
# from datetime import datetime,timedelta


pygame.init()
WIDTH, HEIGHT = 800, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Night Sky Simulation")

# start_time = datetime(2000,1,1)
"""
# create current time = datetime.now(),then time_elapsed = current_time - start_time 
# then elapsed time is time delta object
# we can call the total_seconds method from it
"""

def load_planets():
    # Load the planets from a file
    with open('planets_data.pkl', 'rb') as f:
        planets_flask = pickle.load(f)
    return planets_flask

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LAND_COLOR = (34, 139, 34)  # Dark green, feel free to choose any color that represents land to you

# Observer's location: Haifa
observer_location = EarthLocation(lat=32.7940 * u.deg, lon=34.9896 * u.deg, height=0 * u.m)

# Initialize pygame font
pygame.font.init()
FONT = pygame.font.SysFont('arial', 15)

background_image = pygame.image.load('static/images/ground2.png')
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT // 2))


star_image = pygame.image.load('static/images/star2.png')  # Ensure the image path is correct
star_image = pygame.transform.scale(star_image, (25, 25))  # Scale the image to an appropriate size

planet_images = {}

known_planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "sun"]

# Directory to save screenshots
screenshot_directory = "../images/simulatedImages2"
if not os.path.exists(screenshot_directory):
    os.makedirs(screenshot_directory)


# Load planet images once, scaling them appropriately
def load_planet_images():
    for planet in known_planets:
        image_path = f'static/images/{planet}.webp'
        image = pygame.image.load(image_path)
        planet_images[planet] = pygame.transform.scale(image, (30, 30))  # Scale the image for planets

    # Special case for the moon
    moon_image_path = 'static/images/moon.webp'
    moon_image = pygame.image.load(moon_image_path)
    planet_images['moon'] = pygame.transform.scale(moon_image, (100, 100))  # Larger scale for the moon

    # Default image for unknown planets
    planet_images['default'] = pygame.transform.scale(pygame.image.load('static/images/planet.webp'), (30, 30))

load_planet_images()


def altaz_to_screen(alt, az, width=WIDTH, height=HEIGHT):
    """Convert altitude and azimuth to screen coordinates."""
    x = (az.deg % 360) / 360 * width  # Use modulo to wrap around the azimuth
    y = height - (alt.deg + 90) / 180 * height
    return int(x), int(y)


def draw_local_sky(observer_position, current_time, planet_list):
    WIN.fill(BLACK)  # Clear the entire screen

    altaz_frame = AltAz(obstime=current_time, location=observer_location)
    for star in stars:
        star_coord = SkyCoord(ra=star.ra * u.rad, dec=star.dec * u.rad, frame='icrs')
        star_altaz = star_coord.transform_to(altaz_frame)

        if star_altaz.alt > 0:
            x, y = altaz_to_screen(star_altaz.alt, star_altaz.az)
            WIN.blit(star_image, (x - star_image.get_width() // 2, y - star_image.get_height() // 2))

    filtered_planet_list = [planet for planet in planet_list if planet.name.lower() in known_planets]

    with solar_system_ephemeris.set('builtin'):
        for planet_fromData in filtered_planet_list:
          #  if planet_fromData.name.lower() == 'moon':
                    planet = get_body(planet_fromData.name, current_time, observer_location)
                    altaz = planet.transform_to(AltAz(obstime=current_time, location=observer_location))

                    if altaz.alt > 0:  # If above horizon
                        x, y = altaz_to_screen(altaz.alt, altaz.az)
                        planet_image = planet_images.get(planet_fromData.name.lower(), planet_images['default'])
                        WIN.blit(planet_image, (x - planet_image.get_width() // 2, y - planet_image.get_height() // 2))

              #  label = FONT.render(planet_fromData.name.capitalize(), 1, WHITE)
              #  WIN.blit(label, (x + 5, y + 5))

    WIN.blit(background_image, (0, HEIGHT // 2 + 10))


    pygame.display.update()

def main(planet_list):
    clock = pygame.time.Clock()
    simulation_speed = 60 * 60  # Each frame simulates one hour
    current_time = Time(datetime.datetime.now(timezone.utc))

    frame_count = 0  # Keep track of the number of frames
    running = True
    while running:
        clock.tick(30)  # Limit the frame rate to 30 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if frame_count == 1:  # After the first frame
            pygame.image.save(WIN, os.path.join(screenshot_directory, "screenshot_after_one_frame.png"))
        elif frame_count == 3:  # After the second frame
            pygame.image.save(WIN, os.path.join(screenshot_directory, "screenshot_after_three_frames.png"))

        current_time += datetime.timedelta(seconds=simulation_speed / 30)
        draw_local_sky(observer_location, current_time, planet_list)
        frame_count += 1  # Increment the frame count

    pygame.quit()

if __name__ == "__main__":
    planet_list = load_planets()
    main(planet_list)






