import os

import cv2 as cv
import numpy as np


# Function to load an image
def load_image(file_path):
    """Load an image from the specified file path in grayscale mode and check for errors."""
    image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: No image found at path {file_path}. Check the file path and ensure it points to a valid image.")
        return None
    return image

# Function to preprocess the image
def preprocess_image(image, threshold=200):
    # Apply a threshold to get a binary image (set the image to black and white)
    _, thresh = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    return thresh


# Function to find contours in the image
def find_contours(thresh):
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


# Function to classify shapes based on circularity and size
def classify_shapes(contours, circularity_threshold=0.7, size_threshold=5000):
    object_info = []

    for cnt in contours:
        # Calculate area and perimeter of the contour
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)

        # Calculate circularity
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0

        # Calculate the centroid of the contour
        M = cv.moments(cnt)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x, centroid_y = 0, 0  # Assign some default value or handle the case as needed

        # Classify based on circularity and size
        if circularity > circularity_threshold:
            shape_type = "large circle" if area > size_threshold else "small circle"
        else:
            shape_type = "not circle"  # Or use another classification as needed

        object_info.append((shape_type, (centroid_x, centroid_y)))

    return object_info

# this function identify the biggest circle, made to track the targeted object
def find_biggest_circle(contours, circularity_threshold=0.7):
    max_area = 0
    biggest_circle_center = None

    for cnt in contours:
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0

        if circularity > circularity_threshold and area > max_area:
            max_area = area
            M = cv.moments(cnt)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                biggest_circle_center = (centroid_x, centroid_y)

    return biggest_circle_center

# Function to process the image and classify shapes
def process_and_classify(file_path):
    # Load, preprocess, and find contours
    image = load_image(file_path)
    thresh = preprocess_image(image)
    contours = find_contours(thresh)

    # Classify shapes
    object_info = classify_shapes(contours)

    return object_info

def process_and_find_biggest_circle(file_path):
    image = load_image(file_path)
    thresh = preprocess_image(image)
  #  cv.imshow('Preprocessed Image', thresh)  # Display the image
  #  cv.waitKey(0)  # Wait for a key press to close the window
  #  cv.destroyAllWindows()  # Close all OpenCV windows
    contours = find_contours(thresh)

    biggest_circle_center = find_biggest_circle(contours)
    return biggest_circle_center

def process_multiple_images(image_paths):
    # Initialize an empty list to hold the location of the biggest circle for each image
    locations = []

    # Iterate over each image path provided
    for path in image_paths:
        # Use the process_and_find_biggest_circle function to find the biggest circle in the current image
        location = process_and_find_biggest_circle(path)

        # Append the location to the list; None if no circle was found
        locations.append(location)

    # Return the list of locations
    return locations


# Example usage: for identifying the location of the biggest function
"""
file_path = '/Users/mariorohana/Desktop/Sky View notes/WhatsApp Image 2024-02-12 at 08.52.52.jpeg'  # Update this to the path of your image
biggest_circle_location = process_and_find_biggest_circle(file_path)
if biggest_circle_location:
    print(f"Biggest circle found at {biggest_circle_location}")
else:
    print("No circles found")


# Example usage: for identifying each object

file_path = '/Users/mariorohana/Desktop/image.png'  # Update this to the path of your image
objects_info = process_and_classify(file_path)
for obj_type, (x, y) in objects_info:
    print(f"{obj_type} found at ({x}, {y})")

"""

def get_object_locations():
    directory_path = '../uploadedImages'  # Path to the directory containing images
    image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if
                   f.endswith(('.jpeg', '.png', '.jpg'))]
    object_locations = []

    for path in image_paths:
        location = process_and_find_biggest_circle(path)
        object_locations.append(location)

    print(object_locations)


if __name__ == "__main__":
    get_object_locations()

