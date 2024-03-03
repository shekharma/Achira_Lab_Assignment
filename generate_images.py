import cv2
import numpy as np
import random

# Function to check if a given position is within the non-overlapping region
def is_non_overlapping(position, positions, shape_size):
    x, y = position
    for pos in positions:
        px, py = pos
        if (x + shape_size[1] > px and x < px + shape_size[1]) and (y + shape_size[0] > py and y < py + shape_size[0]):
            return False
    return True

# Function to generate random non-overlapping positions for shapes
def generate_random_positions(image_size, num_shapes, shape_size):
    positions = []
    while len(positions) < num_shapes:
        x = random.randint(0, image_size[1] - shape_size[1])
        y = random.randint(0, image_size[0] - shape_size[0])
        position = (x, y)
        if is_non_overlapping(position, positions, shape_size):
            positions.append(position)
    return positions

# Function to place shapes randomly on the image
def place_shapes(image, shapes, positions):
    for shape, position in zip(shapes, positions):
        x, y = position
        image[y:y+shape.shape[0], x:x+shape.shape[1]] = shape
    return image

# Function to classify shapes based on contours and moments
def classify_shapes(shapes):
    classified_shapes = []
    for shape in shapes:
        contours, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue  # Skip contour with zero perimeter
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.6:
                classified_shapes.append('Circle')
            elif 0.5 >= circularity >= 0.2:
                classified_shapes.append('Square')
            elif 0.9 > circularity >= 0.5:
                classified_shapes.append('Hexagon')
            else:
                classified_shapes.append('Gear')
    return classified_shapes

# Function to create bounding boxes around shapes and assign names
def create_bounding_boxes_with_names(image, positions, shape_size, shape_names):
    for idx, position in enumerate(positions):
        x, y = position
        shape_name = shape_names[idx]  # Get the predicted shape name
        cv2.rectangle(image, (x, y), (x + shape_size[1], y + shape_size[0]), (0, 255, 0), 2)
        cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Function to apply sharpening filter to the image
def sharpen_image(image):
    sharpened = cv2.filter2D(image, -1, np.array([[-1, -1, -1],
                                               [-1, 9, -1],
                                               [-1, -1, -1]]))
    return sharpened

circle = cv2.imread("C:/Users/Administrator/Downloads/Achira lab/input_images/input_images/disc.tif", cv2.IMREAD_GRAYSCALE)
hexagon = cv2.imread('C:/Users/Administrator/Downloads/Achira lab/input_images/input_images/hexagon.tif', cv2.IMREAD_GRAYSCALE)
square = cv2.imread('C:/Users/Administrator/Downloads/Achira lab/input_images/input_images/square.tif', cv2.IMREAD_GRAYSCALE)
gear = cv2.imread('C:/Users/Administrator/Downloads/Achira lab/input_images/input_images/gear.tif', cv2.IMREAD_GRAYSCALE)

# Define image size
image_size = (724, 724)

# Define the number of shapes
k = 4  # Number of different shapes
N = 5  # Number of each shape (repetitions)
num_shapes = N * k  # Total number of shapes

# Define shape sizes
shape_size = (64, 64)  # You can adjust this according to your shapes

# Generate random positions for shapes
positions = generate_random_positions(image_size, num_shapes, shape_size)

# Randomly shuffle the shapes
shapes = [circle, hexagon, square, gear] * N
random.shuffle(shapes)

# Apply sharpening filter to the image
sharpened_image = sharpen_image(np.zeros(image_size, dtype=np.uint8))

# Place shapes on the sharpened image
image_with_shapes = place_shapes(sharpened_image, shapes, positions)

# Display the image with shapes, bounding boxes, and names
cv2.imshow('Image with Shapes, Bounding Boxes, and Names', image_with_shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
