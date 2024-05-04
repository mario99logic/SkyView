import pygame
from motion_Calculations import simulate_motion
from Objects.objects import all_planets

pygame.init()

WIDTH, HEIGHT = 700, 700
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet Simulation")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

sun_image = pygame.image.load('static/images/sun.webp')
sun_image = pygame.transform.smoothscale(sun_image, (80, 80))

# Scaling factor to fit the solar system within our screen
SCALE = 4000000000

planet_images = {}
planet_paths = {}  # Dictionary to store the paths of each planet

for planet in all_planets:
    if planet.name != 'sun' and planet.name != "moon":
        image_path = f'static/images/{planet.name}.webp'
        image = pygame.image.load(image_path)
        image = pygame.transform.smoothscale(image, (40, 40))
        planet_images[planet.name] = image
        planet_paths[planet.name] = []  # Initialize an empty list for storing path points


def draw_orbit(planet):
    """Draw the orbit of a planet by connecting past positions transformed to screen coordinates."""
    if len(planet_paths[planet.name]) > 1:
        screen_path = [(WIDTH // 2 + int(pos[0] / SCALE), HEIGHT // 2 + int(pos[1] / SCALE)) for pos in
                       planet_paths[planet.name]]
        pygame.draw.lines(WIN, WHITE, False, screen_path, 1)


def draw_window(planets):
    WIN.fill(BLACK)

    """sun_present = any(planet.name.lower() == "sun" for planet in planets)

    if sun_present:
        sun_size_factor = max(10, int(50000000000 / SCALE))
        resized_sun_image = pygame.transform.smoothscale(sun_image, (sun_size_factor, sun_size_factor))
        sun_rect = resized_sun_image.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        WIN.blit(resized_sun_image, sun_rect)"""

    for p in planets:

        x = WIDTH // 2 + int(p.location[0] / SCALE)
        y = HEIGHT // 2 + int(p.location[1] / SCALE)
        if p.name not in ["sun", "moon"]:
            draw_orbit(p)
            planet_paths[p.name].append((p.location[0], p.location[1]))  # Store current position in path

            size_factor = max(5, int(21000000000 / SCALE))
            resized_image = pygame.transform.smoothscale(planet_images[p.name], (size_factor, size_factor))
            planet_rect = resized_image.get_rect(center=(x, y))
            WIN.blit(resized_image, planet_rect)

            label = FONT.render(p.name, 1, WHITE)
            WIN.blit(label, (x - label.get_width() // 2, y - 20))

        elif p.name == 'sun':
            sun_size_factor = max(10, int(50000000000 / SCALE))
            resized_sun_image = pygame.transform.smoothscale(sun_image, (sun_size_factor, sun_size_factor))
            sun_rect = resized_sun_image.get_rect(center=(x, y))
            WIN.blit(resized_sun_image, sun_rect)

    pygame.display.update()


pygame.font.init()
FONT = pygame.font.SysFont('arial', 15)


def main():
    global SCALE
    run = True
    clock = pygame.time.Clock()
    dt = 86400  # Time step for each frame, representing one day

    for planet in all_planets:
        if planet.name != 'sun' and planet.name != "moon":
            image_path = f'static/images/{planet.name}.webp'
            image = pygame.image.load(image_path)
            image = pygame.transform.smoothscale(image, (40, 40))
            planet_images[planet.name] = image
            planet_paths[planet.name] = []  # Initialize an empty list for storing path points

    while run:
        clock.tick(60)
        keys = pygame.key.get_pressed()

        if keys[pygame.K_RIGHT]:
            SCALE *= 1.05
        if keys[pygame.K_LEFT]:
            SCALE /= 1.05
        if keys[pygame.K_UP]:
            dt *= 1.05
        if keys[pygame.K_DOWN]:
            dt /= 1.05

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        simulate_motion(all_planets, dt, dt)
        draw_window(all_planets)

    pygame.quit()


main()
