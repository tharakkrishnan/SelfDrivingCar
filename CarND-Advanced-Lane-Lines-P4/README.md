
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1a]: ./camera_cal/calibration1.jpg "Distorted"
[image1b]: ./output_images/undist_calibration1.jpg "Undistorted"
[image2a]: ./test_images/test1.jpg "Road Transformed"
[image2b]: ./output_images/undist_test1.jpg
[image3a]: ./output_images/cmb_test1.jpg "Binary Example"
[image3b]: ./output_images/masked_cmb_test1.jpg "Masked Binary Example"
[image4a]: ./output_images/perspective_original.jpg "Unwarped Example"
[image4b]: ./output_images/perspective_warped.jpg "Warp Example"
[image5]: ./output_images/warped_test1.jpg "Bird's eye-view"
[image6]: ./output_images/detect_lane_test1.jpg "lane detected"
[video1]: ./project_video.mp4 "Video"


### Camera Calibration

#### 1. Compute the camera matrix and distortion coefficients

The code for this step is contained in the first code cell of the IPython notebook located in "./pipline.ipynb" in the functions
_compute\_camera\_calibration\_matrix(save_images=save_images)_ and _undistort\_camera\_image(objpoints, imgpoints, image, fname=fname, save\_images=save\_images)_

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Figure 1a: Distorted Chessboard
![alt text][image1a]
Figure 1b: Undistorted Chessboard
![alt text][image1b]

### Pipeline (single images)

#### 2. Distortion-correction
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

Figure 2a: Distorted Test image
![alt text][image2a]
Figure 2b: Undistorted Test image
![alt text][image2b]

#### 3a. Use color transforms, gradients or other methods to create a thresholded binary image
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in function: _generate\_threshold\_binary\_image(undist, s\_thresh=(170, 255), sx\_thresh=(20, 100), fname=fname, save\_images=save\_images)_ in _pipeline.ipynb_).  Here's an example of my output for this step.  (note: this is not actually for the original test image in Figure 2a.)
In particular, I transformed the image into the HSV space and used only the S parameter. I then run a Sobel gradient in the x direction to capture vertical artifacts like lanes. I combine both the S and Sx parameters to obtain the combined binary.

Figure 3a: Combined binary image
![alt text][image3a]

#### 3b. Generate a region of interest (ROI) mask and mask out unimportant artifacts
I genereated a ROI mask (mask generation in function _region\_of\_interest(cmb,region\_of\_interest\_vertices, fname=fname, save\_images=save\_images))using the following points:
| ROI points    |
|:-------------:| 
| 585, 460      | 
| 203, 720      |
| 1127, 720     |
| 695, 460      |

The mask zeroes out all points the combined binary image that are outside the ROI.

Figure 3b: Masked Combined binary image
![alt text][image3b]


#### 4. Generate Perspective transform and Inverse transform matrices based on an image with straight lanes

The code for my perspective transform is included in a function called _get\_warp\_matrix(image, src, dst, save\_images=save\_images)_, which appears in the file `pipeline.ipynb` (`output_images/perspective_original.jpg` and `output_images/perspective_warped.jpg`)  The function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  
I chose the hardcode the source and destination points in the following manner:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Figure 4a: Original Perspective
![alt text][image4a]

Figure 4b: Bird's eye-view Warped Perspective
![alt text][image4b]

#### 5. Apply the perpective transform on the combined binary image to obtain the bird's eyeview perspective
I applied the perpective transform using the function _apply\_perspective\_transform(img, warp\_matrix, fname='', save\_images=0)_.

Figure 5: Bird's Eye view perspective of the masked combined binary image
![alt text][image5]

#### 6. Identified lane-line pixels and fitted their positions with a polynomial
I identified the lane lines using a brute-force windowing technique. Initially, I take a histogram of the lower half of the combined binary image to identify peaks that 
may repersent lanes. Using the histogram peaks to center my inital 100 pixel width window I search for the center of the lane pixels. The window height is about 72 pixels requiring
me to search using sliding windws along 10 horizontal  strips covering the image from top to bottom. 

The code is in the function _detect\_lanes\_from\_scratch(binary\_warped, fname='', save\_images=0)_ 

Figure 6: Lanes detected on warped binary image
![alt text][image6]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

