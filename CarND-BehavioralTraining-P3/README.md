**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./loss.png "Model loss function"

My project includes the following files:
* `train_steering_model.ipynb` containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* run1.avi video of simulator in autonomous mode 

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The `train_steering_model.ipynb` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### A. Model Architecture and Training Strategy


My model consists of a convolution neural network implemented at `train_steering_model.ipynb` cell 2.


The model includes ELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Since the output is not a classifier ouput, the loss function used was Mean squared error(MSE).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

#### 5. Solution Design Approach

The overall strategy for deriving a model architecture was to use a convolution network similar to LENET to identify features on the track.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I gathered a lot more training data that would show recoveries. I also added functions to:
1. shear the image
2. flip the image to balance out the effects of steering in a single direction around the track
3. add random gamma to counteract the effects of lighting
4. I cropped the image to focus the network and not get distracted by the sky and other unnecessary features.
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded images that showed the 
model how to recover, i.e turn the wheels back onto the road and head back to the center.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### B. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

|Layer (type)                   |  Output Shape     |     Param #  |   Connected to         | 
|:-----------------------------:|:-----------------:|:------------:|:----------------------:|
|lambda_1 (Lambda)              |  (None, 64, 64, 3)|     0        |   lambda_input_1[0][0] |            
|convolution2d_1 (Convolution2D)| (None, 16, 16, 16)|    3088      |   lambda_1[0][0]       |           
|elu_1 (ELU)                    | (None, 16, 16, 16)|    0         |   convolution2d_1[0][0]|            
|convolution2d_2 (Convolution2D)|  (None, 8, 8, 32) |     12832    |   elu_1[0][0]          |            
|elu_2 (ELU)                    | (None, 8, 8, 32)  |    0         |   convolution2d_2[0][0]|            
|convolution2d_3 (Convolution2D)|  (None, 4, 4, 64) |     51264    |   elu_2[0][0]          |            
|flatten_1 (Flatten)            |  (None, 1024)     |     0        |   convolution2d_3[0][0]|            
|dropout_1 (Dropout)            |  (None, 1024)     |     0        |   flatten_1[0][0]      |            
|elu_3 (ELU)                    |  (None, 1024)     |     0        |   dropout_1[0][0]      |            
|dense_1 (Dense)                |  (None, 512)      |     524800   |   elu_3[0][0]          |            
|dropout_2 (Dropout)            |  (None, 512)      |     0        |   dense_1[0][0]        |            
|elu_4 (ELU)                    |  (None, 512)      |     0        |   dropout_2[0][0]      |            
|dense_2 (Dense)                |  (None, 1)        |     513      |   elu_4[0][0]          |            

Total params: 592,497
Trainable params: 592,497
Non-trainable params: 0
Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

### C. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from these positions. 

After the collection process, I had roughly number of data points. 
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used a generator to train the model that would automatically augment the data by flipping, shearing or adding random gamma to the image. The generator allowed me to create and work with larger data sets than I normally could have.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 (with 128 sample batches) as evidenced by the flattening of validation loss curve from this epoch onward.


### D. Results

The model was trained for 8 epochs with 200 batches each, with each batch consisting of 256 samples.
The training loss converged quickly but the validation loss decreased slowly indicating that model continued to learn over 8 epochs. 

![alt text][image2]

The model was then hooked up to drive the simulator by running:

python drive.py model.h5 

The video of the autonomous driving is at run1.avi also on youtube at:

