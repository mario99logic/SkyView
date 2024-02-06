import pygame
import numpy as np
import datetime
from datetime import timezone
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_body, solar_system_ephemeris
from astropy.time import Time
import astropy.units as u
from Objects.objects import all_planets

pygame.init()
WIDTH, HEIGHT = 800, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Night Sky Simulation")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LAND_COLOR = (34, 139, 34)  # Dark green, feel free to choose any color that represents land to you

# Observer's location: Haifa
observer_location = EarthLocation(lat=32.7940 * u.deg, lon=34.9896 * u.deg, height=0 * u.m)

# Initialize pygame font
pygame.font.init()
FONT = pygame.font.SysFont('arial', 15)


def altaz_to_screen(alt, az, width=WIDTH, height=HEIGHT):
    """Convert altitude and azimuth to screen coordinates."""
    x = (az.deg % 360) / 360 * width  # Use modulo to wrap around the azimuth
    y = height - (alt.deg + 90) / 180 * height
    return int(x), int(y)


def draw_local_sky(observer_position, current_time):
    WIN.fill(BLACK)  # Clear screen to black

    land_height = HEIGHT // 2  # Use half of the window height for the land
    land_rect = pygame.Rect(0, HEIGHT - land_height, WIDTH, land_height)
    pygame.draw.rect(WIN, LAND_COLOR, land_rect)

    with solar_system_ephemeris.set('builtin'):

        for planet_fromData in all_planets:
            planet = get_body(planet_fromData.name, current_time, observer_location)

            altaz = planet.transform_to(AltAz(obstime=current_time, location=observer_location))

            if altaz.alt > 0:  # If above horizon
                x, y = altaz_to_screen(altaz.alt, altaz.az)
                pygame.draw.circle(WIN, planet_fromData.color, (x, y), 8)

                label = FONT.render(planet_fromData.name.capitalize(), 1, WHITE)
                WIN.blit(label, (x + 5, y + 5))

    pygame.display.update()

def main():
    clock = pygame.time.Clock()

    # Time acceleration factor: how much time in the simulation passes per real-time second
    simulation_speed = 60 * 60   # Each frame simulates one hour
    current_time = Time(datetime.datetime.now(timezone.utc))

    running = True
    while running:
        clock.tick(30)  # Limit the frame rate to 30 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Advance the simulation time
        current_time += datetime.timedelta(seconds=simulation_speed / 30)  # 30 FPS

        # Recalculate planet positions with the updated time, accounting for Earth's orbit and rotation
        draw_local_sky(observer_location, current_time)

    pygame.quit()

if __name__ == "__main__":
    main()
