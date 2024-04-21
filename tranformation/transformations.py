import math
import numpy as np

# #the functions takes the right ascension (RA) and Declination parameters of an obejct (equatorial system)
# and converts them into X_Y_Z locations
def convert2_x_y_z(distance_ly,ra_hours,ra_minutes,ra_seconds,dec_degrees,
                  dec_minutes,dec_seconds):
    ra_rad = (ra_hours * 15 + ra_minutes * 0.25 + ra_seconds * 1 / 240) * (math.pi / 180)
    dec_rad = (dec_degrees + dec_minutes / 60 + dec_seconds / 3600) * (math.pi / 180)

    # Convert spherical coordinates to Cartesian coordinates
    x = distance_ly * math.cos(dec_rad) * math.cos(ra_rad)
    y = distance_ly * math.cos(dec_rad) * math.sin(ra_rad)
    z = distance_ly * math.sin(dec_rad)

    # Scaling factor
    SCALE = 1e9

    # Scale the coordinates
    x_scaled = x / SCALE
    y_scaled = y / SCALE
    z_scaled = z / SCALE

    return x_scaled,y_scaled,z_scaled

#converts the right ascension from H M S to radians
def hms_to_radians(hours, minutes, seconds):
    # Convert hours, minutes, and seconds to degrees
    total_degrees = hours * 15 + minutes / 4 + seconds / 240
    # Convert degrees to radians
    radians = total_degrees * (np.pi / 180)
    return radians

#converts the declination attribute to radians
def degrees_to_radians(dec_degrees,dec_minutes,dec_seconds):
    dec_rad = (dec_degrees + dec_minutes / 60 + dec_seconds / 3600) * (math.pi / 180)
    return dec_rad

#the functions exectutes a transformation from equatorial coordinate system
#to ecliptic coordinates system
#note:aplha is the Rigth Ascension and delta is Declination attribute
#it works on array of locations (multiple objects) its crusial to  have alpha and delta with same length
#retruned values: lambda represents the longitude attribute in the ecliptic system, beta represents the latitude attribute
#alpha and... are in radians
def equat2eclip(alpha, delta):
    # Earth's axial tilt in radians
    epsilon = (23 + 26/60 + 21.448/3600) * np.pi / 180

    # Ensure input vectors have the same length
    if len(delta) != len(alpha):
        raise ValueError('Inputs must be the same length.')

    # Convert to column vectors
    delta = np.array(delta).reshape(-1, 1)
    alpha = np.array(alpha).reshape(-1, 1)

    # Calculate trigonometric combinations of coordinates
    sb = np.sin(delta) * np.cos(epsilon) - np.cos(delta) * np.sin(epsilon) * np.sin(alpha)
    cbcl = np.cos(delta) * np.cos(alpha)
    cbsl = np.sin(delta) * np.sin(epsilon) + np.cos(delta) * np.cos(epsilon) * np.sin(alpha)

    # Calculate coordinates
    lambda_val = np.arctan2(cbsl, cbcl)
    r = np.sqrt(cbsl**2 + cbcl**2)
    beta = np.arctan2(sb, r)
    r2 = np.sqrt(sb**2 + r**2)

    # Sanity check: r2 should be 1
    if np.sum(np.abs(r2 - 1)) > 1e-12:
        print('Warning: Latitude conversion radius is not uniformly 1.')

    return lambda_val, beta




#conversion from ecliptic system to equatorial system
def eclip2equat(lambda_ecliptic, beta_ecliptic):
    # J2000 epoch Earth's axial tilt in radians
    epsilon = (23 + 26/60 + 21.448/3600) * np.pi / 180

    # Ensure input vectors have the same length
    if len(lambda_ecliptic) != len(beta_ecliptic):
        raise ValueError('Inputs must be the same length.')

    # Convert to column vectors
    lambda_ecliptic = np.array(lambda_ecliptic).reshape(-1, 1)
    beta_ecliptic = np.array(beta_ecliptic).reshape(-1, 1)

    # Calculate trigonometric combinations of coordinates
    sd = np.sin(epsilon) * np.sin(lambda_ecliptic) * np.cos(beta_ecliptic) + \
         np.cos(epsilon) * np.sin(beta_ecliptic)
    cacd = np.cos(lambda_ecliptic) * np.cos(beta_ecliptic)
    sacd = np.cos(epsilon) * np.sin(lambda_ecliptic) * np.cos(beta_ecliptic) - \
           np.sin(epsilon) * np.sin(beta_ecliptic)

    # Calculate coordinates
    alpha = np.arctan2(sacd, cacd)
    r = np.sqrt(cacd**2 + sacd**2)
    delta = np.arctan2(sd, r)
    r2 = np.sqrt(sd**2 + r**2)

    # Sanity check: r2 should be 1
    if np.sum(np.abs(r2 - 1)) > 1e-12:
        print('Warning: Latitude conversion radius is not uniformly 1.')

    return alpha, delta

#the function executes a transformation from ecliptic system to heliocentric system which centered on Sun
def eclip2helio(llambda_val,bet):
    pass

 # #the function converts from equatorial coordinate system centered on earth to galactic coordinate system centered on sun
# #where alpha is the right ascension in radians and delta is declination in radians
def equatorial_to_galactic(alpha, delta):
    # Galactic center coordinates in equatorial system (centered on Sun)
    alpha0 = 0.0
    delta0 = 0.0

    # Convert coordinates to radians
    alpha = np.radians(alpha)
    delta = np.radians(delta)
    alpha0 = np.radians(alpha0)
    delta0 = np.radians(delta0)

    # Calculate differences in coordinates
    d_alpha = alpha - alpha0
    d_delta = delta - delta0

    # Calculate galactic coordinates
    tan_l = np.cos(delta) * np.sin(d_alpha) / \
            (np.sin(delta) * np.cos(delta0) - np.cos(delta) * np.sin(delta0) * np.cos(d_alpha))

    sin_b = np.sin(delta) * np.sin(delta0) + np.cos(delta) * np.cos(delta0) * np.cos(d_alpha)

    # Calculate actual galactic coordinates
    l = np.arctan2(tan_l, 1)
    b = np.arcsin(sin_b)

    # Convert back to degrees
    l = np.degrees(l)
    b = np.degrees(b)

    return l, b

# the function converts the galactic coordinates(longitude,altitude) to xyz format
# l and b given in radians
def galactic_to_cartesian(l, b):
    # Assuming radial distance is 1 (normalized coordinates)
    r = 1.0

    x = r * np.cos(b) * np.cos(l)
    y = r * np.cos(b) * np.sin(l)
    z = r * np.sin(b)

    return x, y, z
