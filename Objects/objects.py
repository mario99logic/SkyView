class CelestialObject:
    def __init__(self, name, object_type, mass, radius, location, speed, initial_speed, initial_location, color):
        self.name = name
        self.object_type = object_type  # Type of the object e.g. "Planet", "Star", etc.
        self.mass = mass  # Mass of the object in kilograms
        self.radius = radius  # Radius of the object in meters
        self.location = location  # Current x, y, z coordinates in meters
        self.speed = speed  # Current speed of the object in m/s
        self.initial_speed = initial_speed  # Initial speed of the object in m/s
        self.initial_location = initial_location  # Initial x, y, z coordinates in meters
        self.color = color

    def __repr__(self):
        return f"{self.name}: Location {self.location}, Speed {self.speed}"


# Create an instance for Sun
sun = CelestialObject(
    object_type="Star",
    name="sun",
    mass=1.989e30,
    radius=696340e3,
    location=[0, 0, 0],
    speed=[0, 0, 0],  # Sun is considered stationary for this model
    initial_speed=[0, 0, 0],
    initial_location=[0, 0, 0],
    color=(255, 255, 0)
)

# Create an instance for Earth
earth = CelestialObject(
    object_type="Planet",
    name="earth",
    mass=5.972e24,
    radius=6371e3,
    location=[1.496e11, 0, 0],
    speed=[0, 29.78e3, 0],  # Moving in the y-direction
    initial_speed=[0, 29.78e3, 0],
    initial_location=[1.496e11, 0, 0],
    color=(100, 149, 237)
)

mercury = CelestialObject(
    object_type="Planet",
    name="mercury",
    mass=3.301e23,
    radius=2440e3,
    location=[5.79e10, 0, 0],  # Average distance from Sun
    speed=[0, 47.87e3, 0],
    initial_speed=[0, 47.87e3, 0],
    initial_location=[5.79e10, 0, 0],
    color=(26, 26, 26)
)

venus = CelestialObject(
    object_type="Planet",
    name="venus",
    mass=4.867e24,
    radius=6052e3,
    location=[1.082e11, 0, 0],  # Average distance from Sun
    speed=[0, 35.02e3, 0],
    initial_speed=[0, 35.02e3, 0],
    initial_location=[1.082e11, 0, 0],
    color=(230, 230, 230)
)

mars = CelestialObject(
    object_type="Planet",
    name="mars",
    mass=6.39e23,
    radius=3389.5e3,
    location=[2.279e11, 0, 0],  # Average distance from Sun
    speed=[0, 24.077e3, 0],
    initial_speed=[0, 24.077e3, 0],
    initial_location=[2.279e11, 0, 0],
    color=(153, 61, 0)
)

jupiter = CelestialObject(
    object_type="Planet",
    name="jupiter",
    mass=1.898e27,
    radius=69911e3,
    location=[7.785e11, 0, 0],  # Average distance from Sun
    speed=[0, 13.07e3, 0],
    initial_speed=[0, 13.07e3, 0],
    initial_location=[7.785e11, 0, 0],
    color=(176, 127, 53)
)

saturn = CelestialObject(
    object_type="Planet",
    name="saturn",
    mass=5.683e26,
    radius=58232e3,
    location=[1.433e12, 0, 0],  # Average distance from Sun
    speed=[0, 9.68e3, 0],
    initial_speed=[0, 9.68e3, 0],
    initial_location=[1.433e12, 0, 0],
    color=(176, 143, 54)
)

uranus = CelestialObject(
    object_type="Planet",
    name="uranus",
    mass=8.681e25,
    radius=25362e3,
    location=[2.877e12, 0, 0],  # Average distance from Sun
    speed=[0, 6.80e3, 0],
    initial_speed=[0, 6.80e3, 0],
    initial_location=[2.877e12, 0, 0],
    color=(85, 128, 170)
)

neptune = CelestialObject(
    object_type="Planet",
    name="neptune",
    mass=1.024e26,
    radius=24622e3,
    location=[4.503e12, 0, 0],  # Average distance from Sun
    speed=[0, 5.43e3, 0],
    initial_speed=[0, 5.43e3, 0],
    initial_location=[4.503e12, 0, 0],
    color=(54, 104, 150)
)

all_planets = [sun, earth, mercury, venus, mars, jupiter, saturn, uranus, neptune]

# Predefined locations on Earth
locations = {
    "Equator": [earth.location[0], earth.location[1], earth.location[2] + earth.radius],
    "North Pole": [earth.location[0], earth.location[1] + earth.radius, earth.location[2]],
    "South Pole": [earth.location[0], earth.location[1] - earth.radius, earth.location[2]],
    # ... add more locations as needed
}

# Directions based on cardinal directions
directions = {
    "North": [0, 1, 0],
    "South": [0, -1, 0],
    "East": [1, 0, 0],
    "West": [-1, 0, 0],
    "Up": [0, 0, 1],  # Zenith
    "Down": [0, 0, -1]  # Nadir
}
