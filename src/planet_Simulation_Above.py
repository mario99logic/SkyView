import pygame
from motion_Calculations import simulate_motion
import sys
sys.path.append('/Users/mariorohana/Desktop/SkyView')

from Objects.objects import all_planets, sun
from Objects.stars import Star

pygame.init()

WIDTH, HEIGHT = 800, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet Simulation")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
Blue = (0, 0, 255)
SUN_RADIUS = 10
stars = [
    Star(name="Star1", magnitude=5, location=[2e11, 0, 0], color=(255, 255, 255)),
    Star(name="Star2", magnitude=4, location=[3e11, 0, 0], color=(255, 255, 255)),
    Star(name="star3", magnitude=3,location=[1e11,0,0],color=(255, 255, 0)),
    Star(name="star3", magnitude=3,location=[3e11,3e11,0],color=(0, 0, 255))
    # Add more stars as needed
]

# Scaling factor to fit the solar system within our screen
SCALE = 1e9
# change this if you want to see all the planets

def draw_orbit(planet):
    """Draw the orbit of a planet based on its initial position."""
    orbit_radius = int(planet.initial_location[0] / SCALE)
    pygame.draw.circle(WIN, WHITE, (WIDTH // 2, HEIGHT // 2), orbit_radius, 1)  # 1 is the line width
##############################3
def draw_stars(stars):
    for star in stars:
        # Calculate position on the screen based on the location
        x = WIDTH // 2 + int(star.location[0] / SCALE)
        y = HEIGHT // 2 + int(star.location[1] / SCALE)
        pygame.draw.circle(WIN, star.color, (x, y), 2)

##################################
def draw_window(planets,stars):
    WIN.fill(BLACK)  # Fill the screen with black color
    draw_stars(stars)
    # Draw the sun in the center
    pygame.draw.circle(WIN, sun.color, (WIDTH // 2, HEIGHT // 2), SUN_RADIUS)

    for planet in planets:
        if planet.name != "sun":  # We don't want to draw an orbit for the Sun
            draw_orbit(planet)

    for planet in planets:
        if planet.name != "sun":
            # Calculate the position of the planet relative to the sun
            x = WIDTH // 2 + int(planet.location[0] / SCALE)
            y = HEIGHT // 2 + int(planet.location[1] / SCALE)

            # Draw the planet as a circle
            pygame.draw.circle(WIN, planet.color, (x, y), 10)

            label = FONT.render(planet.name, 1, WHITE)  # 1 is for antialiasing, and WHITE is the color
            WIN.blit(label, (x - label.get_width() // 2, y - 20))  # Adjust the y-offset (-20) as needed

    pygame.display.update()


pygame.font.init()  # Initialize the font module
FONT = pygame.font.SysFont('arial', 15)  # Choose the 'arial' font with a size of 15


def main():
    run = True
    clock = pygame.time.Clock()
    dt = 86400  # Time step for each frame, representing one day

    while run:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Simulate the motion of the planets for the next time step
        simulate_motion(all_planets, dt, dt)  # Simulate just for the next time step

        draw_window(all_planets,stars)

    pygame.quit()


main()
