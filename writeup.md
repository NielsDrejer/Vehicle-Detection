
## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/CarExamples.png
[image2]: ./output_images/NonCarExamples.png
[image3]: ./output_images/HOGChannel2.png
[image4]: ./output_images/FirstSetSlidingWindows.png
[image5]: ./output_images/FirstTestResult.png
[image6]: ./output_images/SecondTestResult.png
[image7]: ./output_images/HeatMap.png
[image8]: ./output_images/HeatMapThreshold.png
[image9]: ./output_images/HeatMapResult.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in code cells 2 - 7 of the jupyter notebook CarND-Vehicle-Detection.ipynp.  

First I read in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

![alt text][image2]

It is worthwhile noticing that there are 8792 car images and 8968 non car images, so approximately the same number. I therefore decided not to amend the training set.

I used standard get_hog_features() function from the lesson material. This accepts among others `orientations`, `pixels_per_cell`, and `cells_per_block` as parameters. Below is an example of the HOG channel 2 features extracted from an RGB image of a car.

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

The selection of HOG parameters were basically governed by the accuracy I could achieve with my Linear classifier, so you could say it is an empirical choice.

In addition to HOG I also used binned color and color histogram features. I tried quite a few combinations and in the end settled for:

* color_space = 'YCrCb'
* orient = 9  (HOG orientations)
* pix_per_cell = 8 (HOG pixels per cell)
* cell_per_block = 2 (HOG cells per block)
* hog_channel = "ALL" (Can be 0, 1, 2, or "ALL")
* spatial_size = (32, 32) (Spatial binning dimensions)
* hist_bins = 32 (Number of histogram bins)

These are pretty much the starting point parameters proposed by in the lesson material. I did not have time to conduct a complete and thorough test of every combination of the parameters, but the above combination yielded a pretty good accuracy of my classifier (98.54%)

I tried other color spaces, such as HLS and YUV. I also tried to increase the number of HOG orientations, HOG pixels per cell as well as using not all HOG channels. Finally I tried to use solely HOG features, i.e. disabled the spatial binning and color histogram features. With no combination I achieved a better accuracy than with the above combination.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In code cell 11 you can see the code I used to train a linear SVM classifier. Before training I created training and test data sets by splitting my features using the train_test_split() function and a ration 80% training and 20% test data

I further used the sklearn StandardScaler to ensure my training and test data have been scaled so all features carry approximately the same weight.

The output of the accuracy test of my trained classifier was:

Test Accuracy of SVC =  0.9854

My SVC predicts:  [0. 1. 1. 0. 1. 0. 0. 0. 0. 0.]

For these 10 labels:  [0. 1. 1. 0. 1. 0. 0. 0. 0. 0.]

0.00176 Seconds to predict 10 labels with SVC

As mentioned above no other classifier yielded a better result. The second best candidate seemed to be the YUV color space, which is hardly surprising given the similarity of YCrCb and YUV. The main difference was the time it took to extract the features and traing the classifier. Disabling the spatial binning and color histogram features provided a huge time saving, in particular in the sliding window step mentioned below. As my code in no case was anywhere near real time I decided to forget about this aspect and go for precision.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In my notebook you will see 2 sliding windows implementations. Both were taken from the lesson material.

The first simple version is called search_windows(). Testing it I found it could make sense to apply it 3 times using the following x and y limits:

* x_limits = [[None, None], [312, None], [412, None]]
* y_limits = [[400, 640], [400, 600], [390, 540]]
* window_size = [(128, 128), (96, 96), (80, 80)]
* overlap = [(0.5, 0.5), (0.5, 0.5), (0.75, 0.75)]

This looks like this:

![alt text][image4]

The disadvantage of this code is that it is ver inefficient. I noticed this when applying it in my video pipeline. I could process about 1 video frame per second only with my admittedly not very fast Mac Mini.

Therefore I used the function advanced_find_cars() which implements HOG subsampling. I also took this over from the lesson material, but modified it to do the color conversion itself, and to not return the image with the identified windows drawn.

Also here I found that it made sense to apply the function several times to each image to maximize the number of found windows. I ended up calling this function 5 times for each image with the following parameters:

1. ystart = 416, ystop = 480, scale = 1.0

2. ystart = 432, ystop = 528, scale = 1.5

3. ystart = 400, ystop = 528, scale = 2.0

4. ystart = 400, ystop = 596 ,scale = 3.5

5. ystart = 464, ystop = 640, scale = 3.5

These parameters were found empirically, essentially looking at the results. The results of both sliding windows implementations are shown in the next section below.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In both the simple and the advanced case I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, using exactly the same parameters as when extracting the features for training of my linear classifier.

Results for the simple case:

![alt text][image5]

Results for the advanced case:

![alt text][image6]

It is quite obvious that the advanced search gives slightly better results, although also these are not perfect. Most notable are the 2 false positives in test image 5, where something has been detected in the shades of the road middle separation. As discussed below I ended up adding a specific mechanism to remove these kinds of false positives in my final pipeline.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

The mp4 file is also included in the github project.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The first and main method for filtering out false positives and for combining overlapping boxes is the heatmapping and labelling technique from the course material.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap of a test image. You will also see the starting point of the heatmap, which is the result of my 5 HOG subsampling sliding windows searches:

![alt text][image7]

After applying the threshold function (in this case using a threshold of 1) I get:

![alt text][image8]

Finally using the label function and the draw_labeled_bboxes() function the end result looks like this:

![alt text][image9]

But I also implemented a couple of other things to help. First thing to mention is my class Detected_Cars, which I use to store a list of previously detected windows from previous frames. I ended up storing windows detected in the last 15 frames. This helps to average the detected cars, as each bounding box in a given frame really is an average of the bounding boxes of the last 15 frames. The number 15 is highly tunable.

In Detected_Cars I check that each new window really is on the actual road. It turned out that my sliding windows search gave some false positives in the middle road separation. By inspecting an image I decided that the equation x = -2.25*y + 1800 is a reasonable approximation of a left most line for my pipeline. I compared the lower right corner of each detected window with this line, and only windows in which this corner is located to the right of the line are used. All others are thrown away.

I also had a problem with sporadic false positives in the middle of the road. To an extend this problem could be removed by adjusting the threshold applied to the heatmaps of the averages windows, but not completely. I therefore also added a check that I only draw labelled boxes wider that 50 pixels (in the x direction obviously).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As I mentioned above removing false positives appears to be a major challenge. I used hot windows from the last 15 detections (ideally frames), and therefore I had to adjust the threshold I apply to the resulting heatmap. It took some experimentation to get to the result:

apply_threshold(heat_image, 2 + len(detected_cars.prev_windows)//2)

You will notice in my video that at one point in time the frame around the white car disappears completely for a couple of video frames. If I change the threshold to 1 + len(detected_cars.prev_windows)//2, the frame does not disappear, but instead I get a very short false detection at another time in the video. That's how sensitive the whole thing is.

Another important thing I did was to remove all detections to the left of an imaginary line I defined. I had quite a few false detections in the middle plank of the road, and using this crude line got rid of those. This is obviously not usable in practice. Just image if the car would be driving in the rightmost lane of the road instead of the leftmost. But it did the job in the time I have left for the project. A much better implementation could make use of the lane detections from the 1st or the 4th project and discard based on these.

I am also not completely happy with the resulting bounding boxes. You can see in my video that these are not very good at precisely surrounding the detected cars. Partly I think this could become better by using more images for training the classifier. In particular the white car is difficult for my classifier. Another thing that could be done would be to work more on the function to display the bounding boxes. For example if I had an estimate of the distance to each identified car, I could define a minimum size of the bounding box which would fit much more precisely with a car in that distance. I am sure there are many more tricks that could be applied.

Lastly my classifier seems to be quite sensitive to the lightness of the camera images. It produces more false positives where there are shades (therefore the problem with the middle plank), or when the road surface is darker. Much more work should be going into taking care of this problem. Either via more images for training, or perhaps also by making more use of the techniques we applied in the 1st and 4th projects for lane detection.
