########## Note ################
### Run after gen_images.py ###
###############################


import os
# Define the directories
annotation_dir = "/provide/path"  # Directory containing original annotation files
output_dir = "/provide/path" # Directory to store modified annotation files
output_image_dir = "/provide/path"  # Directory to save images

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)

# Loop through iterations to create N images
for i in range(N):
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
    file_path = os.path.join(output_image_dir+'/train/', f"image_{i}.png")
    cv2.imwrite(file_path, image_with_shapes_transformed)
    
    # Write annotations to a text file
    annotations_file_path = os.path.join(output_dir+'/train/', f"image_{i}.txt")
    with open(annotations_file_path, "w") as file:
        for annotation in shape_annotations:
            label, (x1, y1, x2, y2) = annotation
            file.write(f"{label} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")


    # Remove commas from annotations and rewrite the annotation file
    with open(annotations_file_path, "r") as file:
        lines = file.readlines()
    with open(annotations_file_path, "w") as file:
        for line in lines:
            line = line.replace(",", " ")  # Replace commas with spaces
            file.write(line)

print('Annotation Done...')
