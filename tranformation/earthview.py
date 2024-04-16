import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.coordinates import get_body, EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u
from datetime import datetime, timedelta

# Define observation location
observer_location = EarthLocation(lat=51.4769 * u.deg, lon=0.0005 * u.deg)

# Define observation start time and end time
start_time = Time(datetime.utcnow())
end_time = start_time + timedelta(hours=12)  # For a span of 12 hours

# Generate a list of observation times from start to end
times = [start_time + timedelta(minutes=10) * i for i in range(int((end_time - start_time).to(u.hour) / u.hour * 6))]

# Prepare figure for animation
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Initialize objects for animation
moon, = ax.plot([], [], 'o', label='Moon', color='gray')
mars, = ax.plot([], [], 'o', label='Mars', color='red')
sirius, = ax.plot([], [], 'o', label='Sirius', color='blue')

# Set up the plot limits, labels, and other aesthetics
ax.set_ylim(0, 90)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_yticks(range(0, 91, 30))
ax.set_yticklabels(['Zenith', '60°', '30°', 'Horizon'])
ax.set_xticks([angle * u.deg.to(u.rad) for angle in range(0, 360, 45)])
ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
ax.legend()


# Function to update the positions of celestial objects for each frame
def update(frame):
    obs_time = Time(times[frame])
    altaz_frame = AltAz(obstime=obs_time, location=observer_location)

    # Get current positions and transform to AltAz frame
    moon_coord = get_body('moon', obs_time, observer_location).transform_to(altaz_frame)
    mars_coord = get_body('mars', obs_time, observer_location).transform_to(altaz_frame)
    sirius_coord = SkyCoord.from_name('Sirius').transform_to(altaz_frame)

    # Update data for each object
    moon.set_data(moon_coord.az.radian, 90 - moon_coord.alt.degree)
    mars.set_data(mars_coord.az.radian, 90 - mars_coord.alt.degree)
    sirius.set_data(sirius_coord.az.radian, 90 - sirius_coord.alt.degree)

    return moon, mars, sirius


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(times), blit=True, interval=200)

plt.show()