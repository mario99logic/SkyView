import pygame
from motion_Calculations import simulate_motion
import sys


from Objects.objects import all_planets, sun

pygame.init()

WIDTH, HEIGHT = 700, 700
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet Simulation")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
Blue = (0, 0, 255)
SUN_RADIUS = 10

sun_image = pygame.image.load('static/images/sun.webp')
sun_image = pygame.transform.smoothscale(sun_image, (80, 80))

# Scaling factor to fit the solar system within our screen
SCALE = 600000000  # change this if you want to see all the planets


# to see all planets use 12000000000
# to see the smaller part use 600000000

# Assuming you have a function to return the file path for each planet
def get_planet_image_path(planet_name):
    return f'static/images/{planet_name}.webp'


planet_images = {}
for planet in all_planets:
    if planet.name != 'sun' and planet.name != "moon":  # Skip the sun as it's already loaded
        image_path = get_planet_image_path(planet.name)
        image = pygame.image.load(image_path)
        image = pygame.transform.smoothscale(image, (40, 40))  # Adjust size as needed
        planet_images[planet.name] = image


def draw_orbit(planet):
    """Draw the orbit of a planet based on its initial position."""
    orbit_radius = int(planet.initial_location[0] / SCALE)
    pygame.draw.circle(WIN, WHITE, (WIDTH // 2, HEIGHT // 2), orbit_radius, 1)  # 1 is the line width


def draw_window(planets):
    WIN.fill(BLACK)  # Fill the screen with black color

    sun_size_factor = max(10, int(50000000000 / SCALE))  # Adjust base size not to get too small or too large
    resized_sun_image = pygame.transform.smoothscale(sun_image, (sun_size_factor, sun_size_factor))
    sun_rect = resized_sun_image.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    WIN.blit(resized_sun_image, sun_rect)

    for planet in planets:
        if planet.name not in ["sun", "moon"]:  # Skip drawing for sun and moon
            draw_orbit(planet)
            x = WIDTH // 2 + int(planet.location[0] / SCALE)
            y = HEIGHT // 2 + int(planet.location[1] / SCALE)

            # Calculate the dynamic size based on SCALE
            size_factor = max(5, int(21000000000 / SCALE))  # Adjust size base to not get too small or too large
            resized_image = pygame.transform.smoothscale(planet_images[planet.name], (size_factor, size_factor))

            planet_rect = resized_image.get_rect(center=(x, y))
            WIN.blit(resized_image, planet_rect)

            label = FONT.render(planet.name, 1, WHITE)
            WIN.blit(label, (x - label.get_width() // 2, y - 20))

    pygame.display.update()


pygame.font.init()  # Initialize the font module
FONT = pygame.font.SysFont('arial', 15)  # Choose the 'arial' font with a size of 15


def main():
    global SCALE  # Declare SCALE as global to modify the global variable inside this function

    run = True
    clock = pygame.time.Clock()
    dt = 86400  # Time step for each frame, representing one day

    while run:
        clock.tick(60)  # Control the frame rate to a reasonable number
        keys = pygame.key.get_pressed()  # Get the state of all keys

        if keys[pygame.K_RIGHT]:
            SCALE *= 1.05  # Slightly increase scale continuously while key is held
        if keys[pygame.K_LEFT]:
            SCALE /= 1.05  # Slightly decrease scale continuously while key is held
        if keys[pygame.K_UP]:
            dt *= 1.05  # Increase dt for faster simulation speed
            dt = min(dt, 86400 * 10)  # Limit maximum dt to avoid too fast simulation
        if keys[pygame.K_DOWN]:
            dt /= 1.05  # Decrease dt for slower simulation speed
            dt = max(dt, 1)  # Ensure dt does not go below 1 to prevent negative or zero delta time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        simulate_motion(all_planets, dt, dt)
        draw_window(all_planets)

    pygame.quit()




main()
