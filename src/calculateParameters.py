from src.image_processing.Simulation_images import get_object_locations


def compute_velocity_and_position(x1, y1, x2, y2, dt):
    """
    Compute the velocity and new position of an object in space.

    Parameters:
    x1, y1: Coordinates of the object at time t1.
    x2, y2: Coordinates of the object at time t2.
    dt: Time difference between t1 and t2 in seconds.

    Returns:
    velocity: A tuple (vx, vy) representing the velocity components in x and y directions.
    new_position: A tuple (new_x, new_y) representing the estimated position at time t2 + dt.
    """
    # Calculate velocity components
    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt

    # Calculate new position assuming constant velocity
    new_x = x2 + vx * dt
    new_y = y2 + vy * dt

    return (vx, vy), (new_x, new_y)

def compute_parameters():
    directory = '../images/uploadedImages'  # Adjust the directory as needed
    positions = get_object_locations(directory)

    if positions and len(positions) >= 2:
        # Use the first two positions returned by get_object_locations
        (x1, y1), (x2, y2) = positions[:2]
        dt = 60 * 60 * 2  # Time difference in seconds, adjust as needed based on frame capture interval
        velocity, new_position = compute_velocity_and_position(x1, y1, x2, y2, dt)
        print("Velocity (vx, vy):", velocity)
        print("New Position (x, y):", new_position)
        return {'velocity': velocity, 'position': new_position}
    else:
        print("Not enough data to calculate velocity and position.")




