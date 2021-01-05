#========================================================#
#     Author: Imoleayo Abel                              #
#     Course: Sensing and Estimation Robotics (ECE 276A) #
#    Quarter: Fall 2017                                  #
# Instructor: Nikolay Atanasov                           #
#    Project: 01 - Color Segementation                   #
#       File: segmentation.py                            #
#       Date: Oct-30-2017                                #
#========================================================#
import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt
import cv2, os, pickle


#================================================================================#
# Function to eliminate contours that are unlikely to correspond to barrels      #
#                                                                                #
#  Input: list of contours                                                       #
# Output: list of bounding rectangles for contours that are likely to be barrels #
#================================================================================#
def barrelness(cntrs):
    max_aspect_ratio = 3  

    # get bounding rectangles for all contours
    rects = [cv2.minAreaRect(cnt) for cnt in cntrs]

    if len(cntrs) <= 1: return rects

    # select only rectangles with aspect ratio less than the maximum allowed aspect ratio
    rects = [rect for rect in rects if (1.0*max(rect[1])/min(rect[1]) < max_aspect_ratio)]
    if len(rects) <= 1: return rects

    # return rectangles with area greater than quarter the area of largest rectangle
    max_area = max([rect[1][0]*rect[1][1] for rect in rects])
    return [rect for rect in rects if 1.0*rect[1][0]*rect[1][1]/max_area > 0.25]


#=================================#
#           MAIN SCRIPT           #
#=================================#

# compute mean and variance for the color classes if they don't already exist
try:    
    means = pickle.load( open( "means.p", "rb" ) )
    covs = pickle.load( open( "covs.p", "rb" ))
except:
    # folder from which to read MATLAB-generated .mat files
    mats_folder = "mat_files"   
    means = {}
    covs = {}

    # compute and coallate means and covariances of all classes in dictionaries
    for filename in os.listdir(mats_folder):
        if filename.lower().endswith(".mat"):
            # load data from .mat file
            RGB_file = sp.loadmat(os.path.join(mats_folder,filename))
            RGB = RGB_file['data']
            n = RGB.shape[1]

            # get color name and output information
            color_str = filename.split('.')[0]
            print "computing mean and variance for color: " + color_str

            # compute and cache mean and covariance 
            mean = np.array( [ 1.0*np.sum(RGB, axis=1)/n ] ).reshape(3,1)
            means[color_str] = mean
            offset = RGB - mean
            covs[color_str] = offset.dot(offset.T)/n

    # save dictionaries of means and covariances
    pickle.dump( means, open( "means.p", "wb" ))
    pickle.dump( covs, open( "covs.p", "wb" ))

# see if we need to train distance estimating parameter: w_MLE
trainDistanceEstimator = False
try:
    w_MLE = pickle.load( open( "w_MLE.p", "rb" ) )
    w_MLE = w_MLE[0]
except:
    trainDistanceEstimator = True

# folder to read images from. If we need to train distance estimator, set to training set.
# Otherwise set to test set.
folder = "testset"

# if we're training distance estimator, initialize arrays for computing w_MLE
if trainDistanceEstimator:
    X = []
    y = []

# process each image: segment, mask, process mask, draw bouding box, train/predict distance
for filename in os.listdir(folder):
    # only consider .png and .jpg images
    if filename.lower().endswith(".png") or filename.lower().endswith(".jpg"):
        # read image and retrieve dimensions
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rows,cols,chanls = img.shape
        min_dim =  min(rows,cols)      # for making linear regression resolution independent
        n_pixels = rows*cols
        print "Processing image " + filename
        
        # compute the log-probabilities of all pixel belonging to each class
        index_dict = {}
        probs_Vals = np.zeros((len(means),n_pixels))
        means_arry = np.zeros((len(means),chanls))

        # iterate over color classes
        for index,key in enumerate(means.keys()):
            # reshape image vertically and get offset from mean
            offset = img.reshape((n_pixels, chanls)).T - means[key]

            # normalizer for class's Gaussian pdf
            normalizer = 1/(np.sqrt(((2*np.pi)**3)*np.linalg.det(covs[key])))
            
            # compute log probabilities of all pixels belonging to current class
            probs_Vals[index] = (np.log(normalizer) - \
                            (0.5*np.sum(offset * np.linalg.inv(covs[key]).dot(offset), axis=0)))
            
            # save mean color of class in array and the class's index in the array 
            means_arry[index] = means[key].T
            index_dict[key] = index
            
        # get index of class with highest log-probability value for all pixels
        max_prob_indices = np.argmax(probs_Vals, axis=0)

        # label all pixels with the mean color of their respective color classes. This is the 
        # segmented image
        img_sgmt = means_arry[max_prob_indices, :].reshape((rows,cols,chanls)).astype(np.uint8)

        # get the maximum log-probability value for each pixel
        maxcols = probs_Vals[max_prob_indices, np.arange(n_pixels)].reshape(n_pixels,1)

        # create mask of pixels where log-probability of bred (barrel-red) class is 
        # equal to or greater than the maximum log-probability of all the classes
        bredmask = probs_Vals[index_dict['bred'], :].reshape(n_pixels,1) >= maxcols

        # reshape bred (barrel-red) mask image appropriately and prepare for display. This is the 
        # pre-processed mask image
        bredmask_thresh = bredmask.reshape((rows,cols)).astype(np.uint8)*255

        # perform morphological operations to coallate neigboring pixels and to remove isolated 
        # noisy pixels
        kernel = np.ones((5,5),np.uint8)        
        bredmask_thresh_morph = cv2.dilate(bredmask_thresh, kernel, iterations = 3)
        bredmask_thresh_morph = cv2.morphologyEx(bredmask_thresh_morph, cv2.MORPH_OPEN, kernel)
        bredmask_thresh_morph = cv2.morphologyEx(bredmask_thresh_morph, cv2.MORPH_CLOSE, kernel)
        bredmask_thresh_morph = cv2.dilate(bredmask_thresh_morph,kernel,iterations = 7)
        bredmask_thresh_morph = cv2.erode(bredmask_thresh_morph,kernel,iterations = 24)
        bredmask_thresh_morph = cv2.dilate(bredmask_thresh_morph,kernel,iterations = 14)

        # get contours in processed mask
        im2, contours, hierarchy = cv2.findContours(np.copy(bredmask_thresh_morph),\
                                                    cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # if morphological operation removed all contours, revert to original mask before 
        # morphological operations were performed
        if len(contours) == 0:
            im2, contours, hierarchy = cv2.findContours(np.copy(bredmask_thresh),\
                                                        cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # sort contours based on area and remove contours with 0 area
        contours = [cnt for cnt in contours if cv2.contourArea(cnt)>0]

        # compute bounding rectangle for all contours and remove rectangles unlikely to be barrels
        rects = barrelness(contours)

        # sort remaining rectangles based on the x-coordinate of their centers so they go from left
        # to right
        rects.sort(key = lambda elm : elm[0][0])

        # initialize array of distances of barrels if we're training
        if trainDistanceEstimator:
            distances_str = filename.split(".")[0].split("_")

        # copy of original image for displaing bounding rectangles around barrels
        boxed_img = np.copy(img)

        # if we already have w_MLE (we are testing), draw bounding rectangles around barrels and 
        # approximate distances. If we don't yet have w_MLE (we are training), collect areas and 
        # distances for computing w_MLE
        for i,rect in enumerate(rects):
            # get coordinates of vertices of bounding rectangle
            boxfloat = cv2.boxPoints(rect)
            box = np.int0(boxfloat)

            # draw bounding rectange on barrel
            cv2.drawContours(boxed_img,[box],0,(255,255,0),5)

            # if training, save off data for computing w_MLE, otherwise, print info about 
            # bounding rectangle
            if trainDistanceEstimator:
                X.append([1,min_dim/max(rect[1])])
                y.append([int(distances_str[i])])
            else:
                # compute distance and retrieve bottomLeft & topRight points of bounding rectangle
                distance = np.array([1,min_dim/max(rect[1])]).dot(w_MLE)[0]

                # sort vertices of bounding rectangle based on vertical coordinates
                boxfloat = sorted(boxfloat, key = lambda elm : elm[1])

                # bottom left vertex is the leftmost vertex of the two lowest vertices
                bottomLeft = sorted(boxfloat[2:], key = lambda elm : elm[0])[0] 

                # top right vertex is the rightmost of the two higest vertices
                topRight = sorted(boxfloat[:2], key = lambda elm : elm[0])[1]

                # print distance and vertices
                print "Image: " + filename + ", BottomLeftX: " + str(bottomLeft[0]) + \
                                             ", BottomLeftY: " + str(bottomLeft[1]) + \
                                             ", TopRightX: " + str(topRight[0]) + \
                                             ", TopRightY: " + str(topRight[1]) + \
                                             ", Distance: " + str(distance)     
            
        # show segmented image, bred (barrel-red) mask, processed mask, and bounding box image
        f, axes = plt.subplots(2,2)
        f.canvas.set_window_title(filename)
        f.set_size_inches(12, 8, forward=True)
        axes[0,0].set_title("Segmented")
        axes[0,1].set_title("Barrel-red mask")
        axes[1,0].set_title("Processed mask")
        axes[1,1].set_title("Bounding box")
        axes[0,0].imshow(img_sgmt)
        axes[0,1].imshow(bredmask_thresh, cmap="Greys_r")
        axes[1,0].imshow(bredmask_thresh_morph, cmap="Greys_r")
        axes[1,1].imshow(boxed_img)
        plt.show()

        # save images for report
        # cv2.imwrite(os.path.join("IMG_boxed",filename), cv2.cvtColor(boxed_img,cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join("IMG_mask_post",filename), bredmask_thresh_morph)
        # cv2.imwrite(os.path.join("IMG_mask_pre",filename), bredmask_thresh)
        # cv2.imwrite(os.path.join("IMG_segmented",filename), cv2.cvtColor(img_sgmt,cv2.COLOR_RGB2BGR))


# if training distance estimator, compute w_MLE
if trainDistanceEstimator:
    V = np.eye(len(X))
    V_inv = np.linalg.inv(V)

    X = np.array(X)
    y = np.array(y)

    # compute and save off w_MLE to disk
    w_MLE = {0: np.linalg.inv(X.T.dot(V_inv).dot(X)).dot(X.T.dot(V_inv.dot(y)))}
    pickle.dump( w_MLE, open( "w_MLE.p", "wb" ))

    # plot data point and linear estimate
    f, ax = plt.subplots(1,2)
    f.set_size_inches(12, 6, forward=True)
    ax[0].plot(1/X[:,1],y,'bo',label="Actual distances")
    ax[0].legend(loc='upper center', shadow=True)
    ax[0].set_title("Distance vs. Length")
    ax[0].set_xlabel("Length")
    ax[0].set_ylabel("Distance")
    ax[1].plot(X[:,1],y,'bo',label="Actual distances")
    ax[1].plot(X[:,1],X.dot(w_MLE[0]),'ro',label="Linear regression estimates")
    ax[1].legend(loc='upper center', shadow=True)
    ax[1].set_title("Distance vs. 1/Length")
    ax[1].set_xlabel("1/Length")
    ax[1].set_ylabel("Distance")
    plt.show()
