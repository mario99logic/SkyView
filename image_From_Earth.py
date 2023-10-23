import numpy as np
from Objects.objects import all_planets, earth
from Objects.stars import stars
import matplotlib.pyplot as plt
from motion_Calculations import simulate_motion
from matplotlib.animation import FuncAnimation



def is_in_line_of_sight(observer_position, observer_direction, celestial_object):

    if celestial_object == earth:
        return False

    # Compute the vector from the observer to the celestial object
    direction_to_object = np.array(celestial_object.location) - np.array(observer_position)

    # Normalize both direction vectors for comparison
    normalized_observer_direction = observer_direction / np.linalg.norm(observer_direction)
    normalized_direction_to_object = direction_to_object / np.linalg.norm(direction_to_object)

    # Compute the angle between the observer's direction and the direction to the object
    dot_product = np.dot(normalized_observer_direction, normalized_direction_to_object)
    angle = np.arccos(dot_product)

    # For now, let's assume a very wide cone of vision, say 90 degrees (pi/2 radians).
    # You can adjust this value as needed.
    if angle < np.pi / 2:
        return True
    else:
        return False

def normalize_color(color):
    return (color[0]/255, color[1]/255, color[2]/255)


def direction_to_2d_coordinates(observer_position, object_position):
    direction_vector = np.array(object_position) - np.array(observer_position)
    normalized_vector = direction_vector / np.linalg.norm(direction_vector)

    # Scale the normalized vector to fit within the plot's bounds
    scale_factor = 0.8  # Adjust this value as needed
    scaled_vector = normalized_vector * scale_factor

    return scaled_vector[0], scaled_vector[1]

simulate_motion(all_planets, dt=60000, total_time=31536000)


""" this function is for making the skyview image """
def update_sky(frame, ax, observer_position, stars):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X direction')
    ax.set_ylabel('Y direction')
    ax.set_title('2D Sky Representation at timestep ' + str(frame))
    ax.set_facecolor('black')  # Set background to black

    # Simulate motion for this frame
    simulate_motion(all_planets, dt=60*60*24, total_time=60*60*24)  # 1 day per frame

    visible_planets = [planet for planet in all_planets if is_in_line_of_sight(observer_position, observer_direction, planet)]
    for obj in visible_planets:
        x, y = direction_to_2d_coordinates(observer_position, obj.location)
        ax.plot(x, y, 'o', color=normalize_color(obj.color), label=obj.name)  # Planets as white dots

    #for star in stars:
        #x, y = direction_to_2d_coordinates(observer_position, star.location)
        #ax.plot(x, y, '*', color='white')  # Stars as white stars

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Setting up the animation
fig, ax = plt.subplots()
observer_position = [earth.location[0], earth.location[1], earth.location[2] + earth.radius]
observer_direction = [-1, 0, 0]  # Pointing directly along the negative x-axis towards the sun

ani = FuncAnimation(fig, update_sky, frames=365, fargs=(ax, observer_position, stars), repeat=True)
plt.tight_layout()
plt.show()




""""
# observer on Earth's surface,specific point for now
observer_position = [earth.location[0], earth.location[1], earth.location[2] + earth.radius]
observer_direction = [-1, 0, 0]  # Pointing directly along the negative x-axis towards the sun




visible_planets = [planet for planet in all_planets if is_in_line_of_sight(observer_position, observer_direction, planet)]

print(visible_planets)
plot_sky(observer_position, visible_planets, stars)

"""