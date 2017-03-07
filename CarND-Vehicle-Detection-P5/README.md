**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1a]: ./output_images/Training_images_for_vehicles.png
[image1b]: ./output_images/Training_images_for_non-vehicles.png
[image2a]: ./output_images/HOG_Features_Generated_for_Vehicle_Images.png
[image2b]: ./output_images/HOG_Features_Generated_for_Non-Vehicle_Images.png
[image3]: ./output_images/slide_windows_test1.png
[image4]: ./output_images/bboxes_test1.png
[image5]: ./output_images/heatmap_test1.png
[image6]: ./output_images/final_test1.png
[video1]: ./output_images/output.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Extracted HOG features from the training images.

The code for this step is contained in cell 4 of the IPython notebook pipline.ipynb.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1a] 

![alt text][image1b] 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2a]

![alt text][image2b]


#### 2. Final choice of Features.

I used the YCbCr color space to extract the features.

I tried various values of parameters and then settled at:
| Feature extraction parameters |
|:-----------------------------:| 
| orientation           9       | 
| pix_per_cell          (8,8)   |
| cell_per_block        (2,2)   |
| spatial_size          (32,32) |
| hist_bins             32      |

I also augemented the HOG features with Color features by downsampling the image to 32,32 image and flattening it.
I also added a color histogram with 32 bins. These three together form the featur vector to the SVM classifier.


#### 3. Trained a classifier using your selected HOG features and color features.

I trained a linear Support vector classifier by splitting the data into training and test set as shown in `classifier.py`.

### Sliding Window Search
I conducted on optimized sliding window search as shown in the `find_cars` function in `pipeline.py`.
The search was restricted between rows 400 to 680 since this is the reegion of interest on the road where other vehicls will most likely be detected.

![alt text][image3]

Ultimately I searched on scale=1.5 using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

The vehicle is detected multiple times and each detection generates a new bounding box. The final bounding box around the vehicle is drawn by taking the entire width and height of all the boxes combined.

#### 4. Filter for false positives and some method for combining overlapping bounding boxes.
Heat-maps are used to remove both false positives and multiple detectors. 
The heat-map image is initialized with dimensions equals to that of the input images.
Add "heat" (+=1) for all pixels within windows where a positive detection is reported by the classifier.
The hot parts of the map are where the cars are, and by imposing a threshold, I rejected areas affected by false positives.
The `add_heat(heatmap, bbox_list)` and `apply_threshold(heatmap, threshold)` methods in the `heat_map.py` file encapsulate the functionalities described in above three items. 
Based on the output of the `apply_threshold(heatmap, threshold)` methods, I drew bounding boxes around each detected car.

In addition to `add_heat(heatmap, bbox_list)` and `apply_threshold(heatmap, threshold)` I maintained a data-structure as a list of lists: each element in the primary list contains a secondary list of the bounding boxes found in a heatmap frame. In in order to improve the smoothness of the predicted bounding boxes I store a (configurable 25 ) number of heat-maps. When it comes to predicting the thresholded heat-map, we calculated the sum of last N heat-maps and that calculated heat-map passed to the apply_threshold(heatmap, threshold) method.
I recorded the positive detection and created bounding boxes in each frame of video. I saved the positive detection of a vehicle results from over 25 previous frames. As I traverse each frame I keep a count of each new detection of a vehicle and discard vehicles that do not have positive detections over a threshhold. This is performed in lines 180-212 of `pipeline.py`

### Here are six frames, and the multiple bounding boxes that were detected by the classifier:

<img src="./test_images/test1.jpg" width="300"/> <img src="./output_images/bboxes_test1.png" width="300"/>

<img src="./test_images/test2.jpg" width="300"/> <img src="./output_images/bboxes_test2.png" width="300"/>

<img src="./test_images/test3.jpg" width="300"/> <img src="./output_images/bboxes_test3.png" width="300"/>

<img src="./test_images/test4.jpg" width="300"/> <img src="./output_images/bboxes_test4.png" width="300"/>

<img src="./test_images/test5.jpg" width="300"/>  <img src="./output_images/bboxes_test5.png" width="300"/>

<img src="./test_images/test6.jpg" width="300"/> <img src="./output_images/bboxes_test6.png" width="300"/>

### Here are six frames, their corresponding heatmaps and final bounding boxes:
<img src="./test_images/test1.jpg" width="200"/> <img src="./output_images/heatmap_test1.png" width="200"/> <img src="./output_images/final_test1.png" width="200"/>

<img src="./test_images/test2.jpg" width="200"/> <img src="./output_images/heatmap_test2.png" width="200"/> <img src="./output_images/final_test2.png" width="200"/>

<img src="./test_images/test3.jpg" width="200"/> <img src="./output_images/heatmap_test3.png" width="200"/> <img src="./output_images/final_test3.png" width="200"/>

<img src="./test_images/test4.jpg" width="200"/> <img src="./output_images/heatmap_test4.png" width="200"/> <img src="./output_images/final_test4.png" width="200"/>

<img src="./test_images/test5.jpg" width="200"/> <img src="./output_images/heatmap_test5.png" width="200"/> <img src="./output_images/final_test5.png" width="200"/>

<img src="./test_images/test6.jpg" width="200"/> <img src="./output_images/heatmap_test6.png" width="200"/> <img src="./output_images/final_test6.png" width="200"/>


---

### Video Implementation

Here's a [link to my video result](https://youtu.be/ixeJaC10Rl8)

---

### Discussion
Shortcomings:
Much of the computations performed on the pipeline are computationally heavy and are not performant in real-time. 

Improvements:
 1. use principal component analysis to reduce the dimensionality of the feature vector
 2. Use multiple scales while searching using sliding windows to improve the vehicle detection quality in more challenging videos.
 3. Recently, new deep learning based method such ash YOLO and SSD have been recorded very good performance on Detection and Tracking benchmark datasets.
 4. In order to improve real-time performance, benchmark/profile the functions and try to optimize the performance by focusing on improving performance for the largest offenders or implement them in CUDA or on an FPGA. 
