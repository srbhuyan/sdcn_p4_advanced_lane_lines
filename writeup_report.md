##Advanced Lane Line Detection

[//]: # (Image References)

[image1]: ./output_images/chess_undistorted.png "Chess Distort"
[image2]: ./output_images/original.jpg "Road Distorted"
[image3]: ./output_images/undist.jpg "Road Undistorted"
[image4]: ./output_images/binary.jpg "Road Binary"
[image5]: ./output_images/undist_warped.png "Road Warped"
[image6]: ./output_images/sliding_window.jpg "Sliding Window"
[image7]: ./output_images/targeted_search.jpg "Targeted Search"
[image8]: ./output_images/image_final.jpg "Draw To Road"

[video1]: ./project_video_out.mp4 "Video"

###Camera Calibration

The code for this step is contained in lines 18 through 53 of the file called `camera_calibration.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the following result. The image on the left is a distorted chessboeard image and the one on the right is the result after applying undistort.

![alt text][image1]

###Pipeline (single images)
####Distortion Correction
The following images provide an example of how undistortion is applied to road images. The image on the left is a distorted road image and the image on the right has distortion correction applied.
![alt text][image2]
![alt text][image3]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 8 through 48 in `image_processor.py`).  Here's an example of my output for this step.

![alt text][image4]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in a function called `perspective_transform()`, which appears in lines 72 through 90 in the file `camera_calibration.py`.  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. I used the source and destination points in the following manner:

```
src = np.float32([[580,460], [710,460], [1150,720], [150,720]])
offset = 200
dst = np.float32([ [offset, 0],
                   [test_img_size[0]-offset, 0],
                   [test_img_size[0]-offset, test_img_size[1]-0],
                   [offset, test_img_size[1]-0]
                 ])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 580, 460      | 200, 0        |
| 710, 460      | 1080, 0       |
| 1150, 720     | 1080, 720     |
| 150, 720      | 200, 720      |

I verified that my perspective transform was working as expected comparing the test image and its warped counterpart to verify that the lane lines appear parallel in the warped image.

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code that identifies the lane line pixels using the sliding window search on a histogram is in the function names `histogram_search()` (lines 30 through 98) in the file `find_lane_lines.py`. Once the lines are extablished by sliding window the subsequent frames use the code in function `targeted_search()` (lines 148 through 169) in file `file_lane_lines.py`. The result of fitting a line to the lane line pixels using the sliding window method is as show below:

![alt text][image6]

The result of the targeted search is as show below:

![alt text][image7]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I code used to calculate the radius of curvature is in the function `curvature()` in lines 233 through 246 in the file `find_lane_lines.py`. Lines 275 through 283 in the file `find_lane_lines.py` is used to compute the position of the vehicle with respect to center.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 249 through 293 in my code in `find_lane_lines.py` in the function `draw_to_road()`.  Here is an example of my result on a test image:

![alt text][image8]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

Outlier Rejection:
The curve fitting I am using is very sensitive to presence of outliers in the identified pixels. The fitted deviates from the lane lines significcantly when ther are outlier pixels in the identified pixel set. This can be improved by implementing outlier identification and rejection techniques.

Challenge Video:
My pipeline does not work very well on the `challenge_video.mp4`. It looks like the thresholding parameters I am using does not effectively identify the lane lines in the challenge video. To identify the lane lines I will have to try different combinations of color spaces (HLS, HLV etc) or a combination of mutiple color spaces. Using low pass and high pass image filters in combination with color spaces will give a better result.
