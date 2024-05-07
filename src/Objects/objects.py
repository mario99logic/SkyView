import json


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


def parse_data_to_objects(data):
    stars = []
    planets = []
    for item in data:
        name = item['name']
        position = [float(x) for x in item['position'].split(',')]  # Parse position

        if item['type'].lower() == 'planet':  # Include Sun in planets for typical simulations
            # For planets, gather speed, mass, and radius
            speed = [float(x) for x in item['speed'].split(',')]  # Parse speed
            mass = float(item['mass'])  # Mass as float
            radius = float(item['radius'])  # Radius as float

            # Create planet object
            planet = CelestialObject(
                name=name,
                object_type='planet',
                mass=mass,
                radius=radius,
                location=position,
                speed=speed,
                initial_speed=speed,
                initial_location=position,
                color=(255, 255, 255)  # Assume white color for simplicity
            )
            planets.append(planet)

        elif item['name'].lower() == 'sun':
            planet = CelestialObject(
                object_type="star",
                name="sun",
                mass=1.989e30,
                radius=696340e3,
                location=position,
                speed=[0, 0, 0],  # Sun is considered stationary for this model
                initial_speed=[0, 0, 0],
                initial_location=position,
                color=(255, 255, 0)
            )
            planets.append(planet)

        elif item['type'].lower() == 'star' and name != 'sun':
            # For stars, just the name and position
            star = CelestialObject(
                object_type="Star",
                name="sun",
                mass=1.989e30,
                radius=696340e3,
                location=position,
                speed=[0, 0, 0],
                initial_speed=[0, 0, 0],
                initial_location=position,
                color=(255, 255, 0)
            )
            stars.append(star)

    return stars, planets


moon_distance_from_earth = 384400e3  # in meters from Earth
moon_location_relative_to_earth = [1.496e11 + moon_distance_from_earth, 0, 0]  # Simplified model


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

moon = CelestialObject(
    object_type="Moon",
    name="moon",
    mass=7.342e22,
    radius=1737.4e3,
    location=moon_location_relative_to_earth,
    speed=[0, 1.022e3, 0],  # Simplified, relative to Earth's movement
    initial_speed=[0, 1.022e3, 0],
    initial_location=moon_location_relative_to_earth,
    color=(190, 190, 190)
)


all_planets = [sun, earth, mercury, venus, mars, jupiter, saturn, uranus, neptune, moon]

#all_planets = [sun, earth]

def celestial_object_to_dict(obj):
    return {
        "name": obj.name,
        "type": obj.object_type,
        "mass": obj.mass,
        "radius": obj.radius,
        "location": obj.location,
        "speed": obj.speed,
        "color": obj.color
    }

all_planets_json = json.dumps([celestial_object_to_dict(planet) for planet in all_planets])






# Predefined locations on Earth
locations = {
    "Equator": [earth.location[0], earth.location[1], earth.location[2] + earth.radius],
    "North Pole": [earth.location[0], earth.location[1] + earth.radius, earth.location[2]],
    "South Pole": [earth.location[0], earth.location[1] - earth.radius, earth.location[2]],
    # ... add more locations as needed
}

