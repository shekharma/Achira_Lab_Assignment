import os
import cv2
import numpy as np
import random
import argparse

# Function to check if a given position is within the non-overlapping region
def is_non_overlapping(position, positions, shape_size):
    x, y = position
    for pos in positions:
        px, py = pos
        if (x + shape_size[1] > px and x < px + shape_size[1]) and (y + shape_size[0] > py and y < py + shape_size[0]):
            return False
    return True

def generate_random_positions(image_size, num_shapes, shape_size):
    positions = []
    while len(positions) < num_shapes:
        x = random.randint(0, image_size[1] - shape_size[1])
        y = random.randint(0, image_size[0] - shape_size[0])
        position = (x, y)
        if is_non_overlapping(position, positions, shape_size):
            positions.append(position)
    return positions

def place_shapes_with_transformations(image, shapes, positions, shape_size):
    shape_annotations = []
    corner_points = []
    shape_names = ['circle', 'hexagon', 'square', 'gear']
    for index, (shape, position) in enumerate(zip(shapes, positions)):
        shape = shapes[index % len(shapes)]  
        shape_name = shape_names[index % len(shape_names)]  
        
        scale_factor = random.uniform(0.75, 1.0)
        angle = random.uniform(0, 90)
        
        resized_shape = cv2.resize(shape, None, fx=scale_factor, fy=scale_factor)
        
        rows, cols = resized_shape.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated_shape = cv2.warpAffine(resized_shape, rotation_matrix, (cols, rows))
        
        new_shape_size = rotated_shape.shape[::-1]

        x = position[0] + (shape_size[1] - new_shape_size[1]) // 2
        y = position[1] + (shape_size[0] - new_shape_size[0]) // 2
        
        image[y:y+new_shape_size[1], x:x+new_shape_size[0]] = rotated_shape
        
        centre_x = (x + x + new_shape_size[0]) // 2
        centre_y = (y + y + new_shape_size[1]) // 2
        centre_x = centre_x / output_dims
        centre_y = centre_y / output_dims
        
        width = new_shape_size[1]
        height = new_shape_size[0]
        width = width / output_dims
        height = height / output_dims
        
        shape_annotations.append((index % len(shape_names), [centre_x, centre_y, width, height]))
        corner_points.append((index % len(shape_names), [x, y, x+new_shape_size[0], y+new_shape_size[1]]))
        
    return image, shape_annotations, corner_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images with shapes and transformations')
    parser.add_argument('--input', type=str, help='Input directory containing shape images')
    parser.add_argument('--out-dims', type=int, nargs=2, help='Output image dimensions (width and height)')
    parser.add_argument('--nout', type=int, help='Number of output images to generate')
    args = parser.parse_args()

    input_images = args.input
    output_dims = tuple(args.out_dims)
    n_output_images = args.nout

    circle = cv2.imread(os.path.join(input_images, "disc.tif"), cv2.IMREAD_GRAYSCALE)
    hexagon = cv2.imread(os.path.join(input_images, "hexagon.tif"), cv2.IMREAD_GRAYSCALE)
    square = cv2.imread(os.path.join(input_images, "square.tif"), cv2.IMREAD_GRAYSCALE)
    gear = cv2.imread(os.path.join(input_images, "gear.tif"), cv2.IMREAD_GRAYSCALE)

    shape_name = [circle, hexagon, square, gear]
    k = len(shape_name)

    shape_size = (64, 64)

    output_folder = '../generated_image_folder/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(n_output_images):
        num_shapes = int(output_dims[0]**2 / (k * 2.5 * shape_size[0]**2)) * k
        
        positions = generate_random_positions(output_dims, num_shapes, shape_size)
        
        shapes = [circle, hexagon, square, gear] * num_shapes
        random.shuffle(shapes)
        
        image = np.zeros(output_dims, dtype=np.uint8)
        
        generated_image = place_shapes_with_transformations(image, shapes, positions, shape_size)
        
        file_path = os.path.join(output_folder, f'generated_image_{i+1}.png')
        
        cv2.imwrite
