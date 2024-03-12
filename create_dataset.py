################ Note ########################
######## Run after gen_images.py #############
##############################################

import os

# Directory to save images
output_directory = "../Annotated_data/"
# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through 200 iterations to create images
for i in range(200):
    # Generate random positions for shapes
    positions = generate_random_positions(image_size, num_shapes, shape_size)

    # Create a blank white image
    image = np.zeros(image_size, dtype=np.uint8)

    # Place shapes on the image with transformations and get shape annotations
    image_with_shapes_transformed, shape_annotations, corner_points = place_shapes_with_transformations(image, shapes, positions, shape_size)

    # Draw bounding boxes on the image
    for annotation in corner_points:
        label, [x1, y1, x2, y2] = annotation
        cv2.rectangle(image_with_shapes_transformed, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image_with_shapes_transformed, str(label), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)

    # Save the image with annotations
    file_path = os.path.join(output_directory+'Annotated_Images/', f"image_{i}.png")
    cv2.imwrite(file_path, image_with_shapes_transformed)
    
    # Write annotations to a text file
    annotations_file_path = os.path.join(output_directory + 'Annotated_label/', f"image_{i}.txt")
    with open(annotations_file_path, "w") as file:
        for annotation in shape_annotations:
            label, [x1, y1, x2, y2] = annotation
            file.write(f"{label},{x1},{y1},{x2},{y2}\n")
print('Annotation Done...')
#print(f"Image saved with annotations at {image_file_path}. Annotations saved at {annotations_file_path}.")
    
