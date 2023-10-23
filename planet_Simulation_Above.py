import pygame
from motion_Calculations import simulate_motion
from Objects.objects import all_planets, sun

pygame.init()

WIDTH, HEIGHT = 800, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet Simulation")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
Blue = (0, 0, 255)
SUN_RADIUS = 10

# Scaling factor to fit the solar system within our screen
SCALE = 1e9     # change this if you want to see all the planets

def draw_orbit(planet):
    """Draw the orbit of a planet based on its initial position."""
    orbit_radius = int(planet.initial_location[0] / SCALE)
    pygame.draw.circle(WIN, WHITE, (WIDTH // 2, HEIGHT // 2), orbit_radius, 1)  # 1 is the line width


def draw_window(planets):
    WIN.fill(BLACK)  # Fill the screen with black color

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

        draw_window(all_planets)

    pygame.quit()


main()
