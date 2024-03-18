# Achira_Lab_Assignment
### Question:
#### You have given a 4 type of shapes(square, hexagon, gear and circel) so you have to use these shapes and randomly sample it on the blank black sheet so it's like now the shapes presents on blank black sheet.Once you have a such image use algorithm/Deep learning approach to detect/identify these shapes (you can bound it) and label them according to their shapes.
It should do the following: 
1. Read N images from the input folder. These are the N shapes you need to identify.
2. Generate M x M pixel output images which have a distribution of all the shapes in it.
3. A given output image should contain k images of each type. Compute k based on the input image sizes and M. (Or make a reasonable assumption.) 
4. The shapes should not overlap with each other. 
5. Each shape should be placed at a random position within the image, without getting cut off at the boundary. 
6. Each shape should be scaled randomly by a factor or 1 to 0.75. 
7. Each shape should be rotated randomly by 0 to 90 degrees. 
8. Support required command line arguments (see below) for the program.

You should be able to run your Python program from the command line as shown in a sample run below       python3 gen_images.py --input input_images/ --out-dims 1024 1024 --nout 1000
#### Notes 
1. Follow professional coding guidelines for your program. 
2. Include error checking in your code. 
3. Think about how you will test for correctness of your program. (No need to implement unit testing.) 
4. You are welcome to use common Python libraries like numpy, scipy, PIL, etc.
5. The images required for the test are in the Google Drive folder shared with you over email.
6. Upload your results to a public github repo and send it to us. Please document what you ##did in a top level README.md

Might the problem creates little confusion.This is small picture will give you idea about the difference between image classification, localization and detection.
![cat](https://github.com/shekharma/Achira_Lab_Assignment/assets/122733304/3dbcb61d-43b8-4be7-b3bd-72256a9e8b00)

## Part 1
#### In this part I used inbuilt python and OpenCV libraries and functions which generate images and some manual analysis like thresholding and contour base approach to detect the shape.
generate_images.py file shows the basic approach to solve the problem of no prior available data.

## Image Generation
Constraints :
1.  Shapes should be non-overlapping.
2.  Shapes should randomly distrubuted.
3.  Shapes are come from the random transformation such as scale factor=0.7 to 1.0(Image Scaling) and  angle=0 to 90 degree (Random rotation)

so to follow this constraints, is_non_overlapping() function to check non-overlapping positions for shapes, after checking overlapping the generate_random_positions() function generates the shapes and place_shapes_with_transformations() function for image transformations.

This functions combinely generate the image for the given instructions/constraints.
The arguments used are the:
1.  image_size=(1024,1024)
2.  k= 4  the nmber of given shapes
3.  shape_size=(64,64)
4.  num_shapes(calculated)=N=int(image_size[0]**2/(k*A*shape_size[0]**2)) the A is adjusting parameter to avoid the crowd of shapes in our image i.e. more the A value less the number of shapes, less the crowd (I took A=2.5 which is gave 25 shapes for each shape type)

## Shape Detection/ Image Segmentation
Image Detection is offen we call it as image localization + classification. The main task in image detection is to identify the boundary. So to detect the boundary i used thresholding technique
#### Image Thresholding:
In Image thresholding we find the optimized pixel value which separates the shape from the background and highlight it. Adjusting the pixel values affect on our shape separation.
To explain more:
   -   _, threshold=cv2.threshold(img, 25, 255, cv2.THRESH_BINARY):

This line of code applies the thresholding operation to the input image(img) using a threshold value of 25. Pixels with intensities greater than or equal to 25 will become white (255), and pixels with intensities below 25 will become black (0).
The thresholded image is stored in the variable threshold, while the threshold value used is stored in the dummy variable _ (underscore), which is ignored. Once we separate our shapes from background next we have to find the category of shape and that can be done by seeing the boudary of shape and for that we are drawing contours.


#### Contour Drawing 
Contour drawing is nothing but the boundary drawing of our shapes. Let's understand through code lines

1. **cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE):**
   - `cv2.findContours()` is a function used to find contours in a binary image.
   - It takes three arguments:
     - `threshold`: This is the binary image on which contours will be found. It should be a single-channel image with pixels either 0 or 255 (white or black).
     - `cv2.RETR_TREE`: This argument specifies the retrieval mode for contours. `cv2.RETR_TREE` retrieves all of the contours and reconstructs a full hierarchy of nested contours.
     - `cv2.CHAIN_APPROX_SIMPLE`: This argument specifies the contour approximation method. `cv2.CHAIN_APPROX_SIMPLE` compresses horizontal, vertical, and diagonal segments and leaves only their end points.

2. **contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE):**
   - This line of code calls `cv2.findContours()` with the specified arguments and stores the result in the `contours` variable.
   - `contours` is a list of contours(disc,gear,square,hexagon) found in the binary image.

3. **contour_img = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR):**
   - `cv2.cvtColor()` is a function used to convert an image from one color space to another.
   - Here, it converts the binary image `threshold` from grayscale (`cv2.COLOR_GRAY2BGR`) to BGR color space.
   - The reason for this conversion is that `cv2.drawContours()` requires a color image (BGR) to draw contours on.

4. **cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1):**
   - `cv2.drawContours()` is a function used to draw contours on an image.
   - It takes several arguments:
     - `contour_img`: This is the color image on which contours will be drawn.
     - `contours`: This is the list of contours to be drawn.
     - `-1`: This argument specifies to draw all contours found in the image.
     - `(0, 255, 0)`: This argument specifies the color of the contours. In BGR format, it represents green (0 for blue, 255 for green, and 0 for red).
     - `1`: This argument specifies the thickness of the contour lines in pixels.
   - After this function call, `contour_img` will have the contours drawn on it.

Finally, `contour_img` will contain the original image with the contours drawn on it. You can then use this image for visualization or further processing.
Now this 'contour_img' is used for the further calculation for labeling the image.If check the codelines in Detect_Shape.py the for loop use to iterate over each contours and performe this two main task 

1.  cv2.arcLength(contour, True): This function calculates the arc length or perimeter of a contour. The contour argument is the input contour, and the second argument (True) specifies whether the contour is a closed curve or not (in this case, it's set to True).

2.  epsilon = 0.038 * cv2.arcLength(contour, True): This line calculates a value (epsilon) which will be used as a parameter for the approximation of the contour. The value of epsilon is set to 3.8% (0.038) of the contour's arc length. This value determines the maximum distance from the original contour to the approximated contour. Adjusting this value will change the level of approximation; smaller values will result in more accurate contours, while larger values will result in less accurate contours.
3.  cv2.approxPolyDP(contour, epsilon, True): This function approximates a polygonal curve to the contour. It takes the original contour (contour), the epsilon value calculated previously, and a Boolean parameter specifying whether the curve is closed or not (in this case, it's set to True). The function then returns the approximated polygonal curve.

These two lines of code are used to approximate a contour with a polygonal curve, where the accuracy of the approximation is determined by the epsilon parameter, which is calculated as a fraction of the contour's arc length. Adjusting epsilon allows you to control the level of detail in the approximation.

### Result
You can see the result of this part in file before_contour.png, after_contour.png.
 Image After the contour drwaing and detection. | Image before the contour drawing. | Original Image
 :---------------------------------------------:|:--------------------------------: | :-----------------------------: |
![after_contour](https://github.com/shekharma/Achira_Lab_Assignment/assets/122733304/d35a4fd8-91bb-46f3-8b85-dcbed367155c) | ![before_cotour](https://github.com/shekharma/Achira_Lab_Assignment/assets/122733304/9a13fc56-cfb8-4e3b-a0dc-7d392790c432) | ![image_with_shapes_transformed_100](https://github.com/shekharma/Achira_Lab_Assignment/assets/122733304/5c7778d4-4d7a-4b78-b2e9-14dd1441c602)





## Part 2
### Deep learning approach
In this section of code we create our own dataset and train on YOLO(yolo5n.pt and yolo8n.pt) no any special reason to use this model.

### Image Generation
In this part, I used same code skeleton to generate the non-overlapping, scaled and rotated images. Python script is in gen_images.py
The user arguments for the script are the directory of input shapes (circle, hexagon, square, gear), size of the image that you want to generate like 1024x1024 and the number of images that you want eg. 100 images.
   -   Command line for this is python3 gen_images.py --input input_images/ --out-dims 1024 1024 --nout 1000

Generated Image 1 | Generated Image 2 
 :---------------------------------------------:|:--------------------------------: |
 ![image](https://github.com/shekharma/Achira_Lab_Assignment/assets/122733304/404087ef-4ab3-48ef-b659-9663fae5da39) | ![image](https://github.com/shekharma/Achira_Lab_Assignment/assets/122733304/ddc21e6c-d3a8-4fb8-b7a1-21b7d2636f2a)


### Data labeling
To detect the shape in an image we want a dataset where we can give the label data with their position. The general YOLO format for the such data is <label> <centre_x> <centre_y> <width_of_object> <height_of_object>. There are other formats are also for such a task which depends on the what model you are using. This position of label is normalized using image size. For centre_x and width_of_object we divided by the width of image and for centre_y and height_of_object we divided by the height of image. You can see the results in image_994.txt file.
   -   Note:- while creating labels for YOLO save the images and labels with same names like for images image_994.jpg and label and annotation image_994.txt.

### Model training
Here I tried the pre-trained YOLO models yolov5 and yolo8 there is no any special reason to use this models, but the trainig and prediction is pretty easy no need to write number of code lines.
code line is
   -   !yolo task=detect mode=train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640

Here task is the detection you can change it for segmentation and mode is for the training and prediction if you want to pass more arguments like batch_size and any other you can extend the code line with that argument.

This is the whole about my work.
I don't have any result to show, the reason behind the low computational power (CPU). If you want perform such task make sure you have GPU support.

## YOLOv8 Architecture
![239739723-57391d0f-1848-4388-9f30-88c2fb79233f](https://github.com/shekharma/Achira_Lab_Assignment/assets/122733304/91947436-4d00-474c-8507-5d30c7eb084a)


#References
   -   https://docs.ultralytics.com
   -   https://learnopencv.com/contour-detection-using-opencv-python-c/

