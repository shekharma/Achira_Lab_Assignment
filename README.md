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

## Part 1
#### In this part I used inbuilt python and OpenCV libraries and functions which generate images and some manual analysis like thresholding and contour base approach to detect the shape.
generate_images.py file shows the basic approach to solve the problem of no data available.
