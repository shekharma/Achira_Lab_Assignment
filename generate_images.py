
import cv2
import numpy as np
#img="C:/Users/Administrator/Downloads/Achira lab/image_with_shapes.png"
img ='C:/Users/Administrator/Downloads/Achira lab/image_with_shapes_transformed_100.png' ## directory for image generated with 25 shapes each
img=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
_, threshold=cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)  
# Apply morphological operation (dilation) to smooth the edges
#kernel = np.ones((2,2), np.uint8)
#smoothed_img = cv2.dilate(threshold, kernel, iterations=5)

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_img = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing contours
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)  # Draw all contours with green color and thickness 3

circle = cv2.imread("C:/Users/Administrator/Downloads/Achira lab/input_images/input_images/disc.tif", cv2.IMREAD_GRAYSCALE)
hexagon = cv2.imread('C:/Users/Administrator/Downloads/Achira lab/input_images/input_images/hexagon.tif', cv2.IMREAD_GRAYSCALE)
square = cv2.imread('C:/Users/Administrator/Downloads/Achira lab/input_images/input_images/square.tif', cv2.IMREAD_GRAYSCALE)
gear = cv2.imread('C:/Users/Administrator/Downloads/Achira lab/input_images/input_images/gear.tif', cv2.IMREAD_GRAYSCALE)

# Initialize lists to store shape labels
shape_labels = [square, gear, hexagon, circle]
N=10   ## each shape repeatation
k=4 ## no .of shapes
# Loop over contours to label shapes and draw labels
for contour in contours:
    # Approximate the contour with polygonal curves
    epsilon = 0.038* cv2.arcLength(contour, True)  # Adjust the epsilon value as needed
    approx = cv2.approxPolyDP(contour, epsilon, True)
    #num_vertices = len(contour)
    # Determine the number of vertices in the approximated contour
    num_vertices = len(approx)
    #print(num_vertices)
    # Label shapes based on the number of vertices
    if 4>= num_vertices:
        current_shape_label='Square'
        
    elif 6>=num_vertices >4:
        current_shape_label='Hexagon'
        
    elif 8>=num_vertices > 6:
        current_shape_label='Gear'
        
    else:
        current_shape_label='Circle'  # Assuming all other contours are circular
        
    # Draw the approximated contour on the image
    cv2.polylines(contour_img, [approx], True, (0, 0, 255), 1)  # Draw the approximated contour in red with thickness 1
    
    # Get the centroid of the contour
    M = cv2.moments(approx)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    # Calculate the position for writing the label
    label_pos = (cX-10 , cY)  # Adjust the offset (10 pixels to the right of the centroid)
    
    # Write the label on the image
    cv2.putText(contour_img, current_shape_label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
# Display the image with contours and labeled shapes
cv2.imshow('Threshold', threshold)
cv2.imshow('Image with Contours and Labeled Shapes', contour_img)
cv2.imwrite('before_cotour.png', threshold)
cv2.imwrite('after_contour.png', contour_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


