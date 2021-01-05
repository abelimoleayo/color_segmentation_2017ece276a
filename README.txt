FILES AND FOLDERS AND ASSUMPTIONS
====================================
label.m         : MATLAB file used for generating data for labeling color
                     - Assumes "trainset/" folder exists for reading images for labelling
                     - Creates "mats_files/" folder and stores therein ".mats" files of 
                       RGB values for each color class

mats_files/     : contains all ".mats" files created using "label.m"

means.p         : Pickle file with means of color class Gaussians
covs.p          : Pickle file with covariances of color class Gaussians
w_MLE.p         : Pickle file with linear regression parameter for distance approximation

segmentation.py : Main python script
                     - Assumes "means.p" and "covs.p" exists in the same folder. If not, it
                       creates them using ".mats" files in "mats_files/" folder which is
                       assumed to exist
                     - Assumes "w_MLE.p" exists. Uses this for approximating distance from
                       barrel
                     - Assumes "testset/" exists with images to be tested. You can modify 
                       "folder" variable on line 85 to use test images from a different folder
README.txt      : This file
                     - contains instructions



EXECUTION INSTRUCTIONS
========================
Make sure all requirements above are met, then run "python segmentation.py" from terminal

WHAT PYTHON SCRIPT DOES AND OUTPUTS
-----------------------------------
- Reads test images (.png or .jpg) from "testset/" folder. 
     - If you need to use images from a different folder, modify the "folder" variable in 
       line 85 of "segmentation.py" 

- For each image 
     - Launches one window with the following four images
          - Color segmented image
          - Mask of red barrel before morphological operations
          - Mask of red barrel after morphological operations
          - yellow bounding box around red barrels in original image
     - Prints to terminal
          - For each barrel in image, one line of text 
                - Image number
                - Coordinates of bounding box as specified in problem 4 of HW
                - Approximate distance from camera
     - CLOSE DISPLAY IMAGE FOR EXECUTION TO CONTINUE TO NEXT IMAGE
         



