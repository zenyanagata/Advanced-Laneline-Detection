# Advanced Lane Finding Project
---

<video controls autoplay loop width="520" height="340">
  <source src="project_video_myoutput.mp4" type="video/mp4">
  <p>unable to play the video. Try other brawser.</p>
</video>

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

[image1]: ./output_images/undistort_and_warped.jpg "Undistorted"
[image2]: ./output_images/undistort_and_crop.png "Road Transformed"
[image3]: ./output_images/binary_lanedetect.png "Binary Example"
[image4]: ./output_images/birds_eye_view.png "Warp Example"
[image5]: ./output_images/sliding_window_and_curvature.png "Fit Visual"
[image6]: ./output_images/unwarp.png "Output"
[video1]: ./test_videos_output/project_video_myoutput.mp4 "Video"

### Here I will describe step by step of how I implemented the lane detection algorithm. All codes for this project are contained in the IPython notebook located in "./examples/summery.ipynb"

---

### Camera Calibration
I used all the 20 images in "./camera_cal" to find each corner of chessboard and created objpoints, imgpoints.
Next, use cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None) to calibrate all the images and get mtx and dist coefficient so that I can undistort the image using cv2.undistort.
![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

First, I used the cv2.undistort to undistort my image (used mtx and dist are already derived by chessboard calibration step).
And then, get rid of the bottom of the image where the car bumper is. Cope is simple ```crop_undist = undist[:670,:]```.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at function ```binary_detection``` which appears the 4th code cell of the IPython notebook).  Here's an example of my output for this step.

![alt text][image3]

I used hls transform before applying sobel matrix instead of gray-scale. This enabled the sobel transform to detect yellow lines and white lines more vividy. I detail, I used "saturation" to detect yellow, and "lightness" to detect white line more precisely (in code it looks like ```grad_combined[ (gradx_l==1) & (grady_l==1) | (gradx_s==1) & (grady_s==1) ] = 1```).
And then, combined magnitude & direatcion detection as well but here I used half saturation half lightness transform beforehand (the code looks like this```magdir_combined[ (mag_lshalf==1) & (dir_lshalf==1) ] = 1```).

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
The code for my perspective transform includes a function called `perspective_transform`, which appears the 5th code cell of the IPython notebook.  This function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([ [593, 450], [685, 450], [1045, 670], [270, 670] ])
dst = np.float32([ [320, 0], [960, 0], [960, 670], [320, 670] ])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
Also, calibration ratio for pixel to meter is desctibed in the following image.
![alt text][image4]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Then I used the sliding window method using convolution window to get the polynomial fit.
The code appears in the 6th code cell of the IPython notebook.
I did some sanity check here as well.
1. check if both curvature is within +- 10% of each other.
2. check if the detected lane width is within +- 10% of 640 pixel (3.7 meters).
 -->  if not, then line with less detection points are deleted and line with more detection points are drawn horizontially in +- 640 pixel (or 3.7 meters).

The result looks like this.
![alt text][image5]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
I did this in the same line as 4. The radius of curvature and center offset are written in the previous image.
Since the lane width and lane length after bird's eye view transform is roughly 640 pixel and 100 pixel, conversion rate of x, y axis are 3.7/100 meter/pixel, 3.0/640 meter/pixel. I don't know why but the result is not as expected. I need some help for this section.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I implemented this step in the 7th code cell of the IPython notebook. Here is an example of my result on a image.
I did the perspective transform as same before but the src and dst are flipped this time to warp it back.

![alt text][image6]

---

### Pipeline (video)
In addition to the pipeline I've mentioned so far, some other features are added to the video pipelline.
Code for this section is in the last code cell of the IPython notebook.
New features:
1. Sanity check by comparing previous detected lane line and current.
   Is the detected lines reasonable enough when compared to the previous one?
   --> If not, use the previous lane line for current detection.
2. Smoothening.
   By taking the average fitted polynomial coefficients of 5 previous images, I've managed to make a smooth detection.
   (Still, leaves much room for improvement.)

Here's a [link to my video result](./test_videos_output/project_video_myoutput.mp4)

---

### Discussion

#### 1. Issues I faced in this project.
My pipeline will likely fail when the road color is similar to white. Since my lane detection using gradient is based on lightness transform, distinguishing white road from lane is tough. (lightness is for detecting white color region.)
Here I'll talk about the approach I took. I made a following sanity check for fitted polynomial coefficients.
1. Is the previous and current fitted polynomial coefficients close to each other?
   Code looks like this where a_thresh, b_thresh, c_thresh are the threshold I defined: 
   ```if right.diffs[0]/rightfit_co[0] > a_thresh or right.diffs[1]/rightfit_co[1] > b_thresh or right.diffs[2]/rightfit_co[2] > c_thresh:```

To make it more robust, sanity check thresholds (i.e., threshold for fitted polynomial coefficients) may be varied in respect to road color.
To pursue this project further, I would make the pipeline more simple so that real time lane line detection can be achieved.
One idea is using previous detected lane line to predict the next line.
