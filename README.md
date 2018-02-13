# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Additional features from color histogram and spatial binning are also used as part of feature vector in training. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./images/car.png
[image2]: ./images/notcar.png
[image3]: ./images/hog_grad.png
[image4]: ./images/heat1.png
[image5]: ./images/heat2.png
[image6]: ./images/heat3.png
[image7]: ./images/heat4.png




Dataset
---

The dataset used in this project is the provided dataset of samples of vehicles and non-vehicle images. It is composed from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video. 

Here are some example images labeled as 'vehicle', followed by those labeled as 'non-vehicle'

![car][image1]
![notcar][image2]



Feature Extraction and Classifier Training
---

The code for this step is contained in the code cells under the "Feature Extraction and Classifier Training" section of the Ipython notebook.


The data images were read in and processed to extract the following features:

- Color histogram features 
- Spatial binning features
- Hisgotram of Oriented Gradient (HOG) features


#### Color Histogram and Spatial Binning Features

Color histogram of color values embeds the distribution of color values and allows for object detection without worrying about orientation and image size (unlike template matching, which depends heavily on raw color values layed out in specific order.) Once an image is loaded (and converted if requested), it is split up into its three channels, and a histogram of each channel is computed. These histograms are then concatenated and included as features.

While template matching is not robust, it still holds useful information. On the other hand, maintaining a full resolution image is not very effective either. Thus, spatial binning in performed on each image channel to resize the original image to a smaller resolution. 


#### HOG Features

HOG works over cells within an image, computing the the gradient magnitudes and directions at each pixel within cells. It produces a histogram of the magnitudes over different orientaton bins.

We utilize the [HOG implementation](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog) from the [scikit-image](http://scikit-image.org/) package.

The color space chosen for the features __'YCrCb'__, as it seemed to provide the best accuracy. YCrCb space provides channels for luminescence, improving the performance on areas with shadows. The following sample visualization demonstrates the gradient orientation and magnitude found with HOG, for a single channel of the car image and a non-car image.

![notcar][image3]

The parameters that were eventually used by HOG are as follows, which provided good accuracy empirically:

HOG features:
- __orient__ : 18
- __pix_per_cell__ : 8
- __cell_per_block__ (the number of HOG cells per block): 2 
- __hog_channel__ (which channels to consier: "ALL"
- __spatial_size__ (dimensions for spatial binning): (16, 16)
- __hist_bins__ (number of histogram bins): 16 

Color Space: __'YCrCb'__



#### Feature Pre-Processing

Once the features were extracted from all of the dataset images, the features and labels were __randomized__ in order and __split into training and test sets__ at 8:2 ratio, using [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Then, the data was normalized to zero mean and unit variance using scikit-learn's [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).


#### Classifier

The Linear SVM implementation in [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) was used. I trained a linear SVM using the features obtained above, with the aforementioned parameter values.

The color space made a significant difference in accuracy, achieving accuracy of 0.9924 on the test set. Since this accuracy seemed good for the purposes of the project, the penalty parameter C of the SVM was not tuned. 





Vehicle Detection
---

The code for this sliding window search is contained in the cells under the "Vehicle Detection (test images)" section of the Ipython notebook.


### Sliding Window

The sliding window search, as the name suggests, 'slides a window' that selects a region within the image, extracts necessary features (i.e, spatial binning, color histogram, and HOG features, in the same ordering used in the classifier training), and feeds the feature representation of that patch to the classifier. If the classifier identifies that feature vector as a car, then the corresponding window location that generated the image patch / feature vector gets marked as a location of interest. This is repeated across the rows and columns of the image with certain strides in both directions.

Window overlap was chosen empirically at 2 steps. While other step sizes were considered and tested, it seemed to be a tradeoff between how fast the algorithm runs vs the accuracy of the model. 2 steps (or 75% overlap in the case of 8x8 cells) seemed to provide the best balance. 

In addition, the sliding window search is repeated for windows of various scales. The scales were chosen empirically as well, based 

A shortcut done to reduce computation time is to limit the region that sliding window gets operated on to the bottom half of the image only, where the cars would be found.

![alt text][image3]


### Heatmap 

A heatmap was employed to identify "hot spots" in the image in which one or more bounding boxes which were classified to have cars. This provides a relative sense of the area of the image in which there is a higher probability of a car. 

The pixels of the heatmaps were then compared against a threshold to remove those spots that did not receive enough votes. This was a method to remove false positives and make the classifier more reliable. 

The heatmaps also serves as a way of combining together overlapping windows when there are duplicate detections. This is because all of the bounding boxes are reduced to regions of "heat". Furthermore, when combined with multi-scale windows, it helps maintain a more accurate bounding box around the car, by having reinforced voting to areas of overlapping windows. In these ways, the heatmap helped the classifier be more reliable. 


### Examples of Sliding Window + Heatmap on Test Images

Below are examples illustrating the detection by the multi-scale sliding window combined with heatmaps for thresholding. Multiple scale sizes were employed to ensure that a vehicle does indeed get detected when there is one. On the other hand, since this increases more chances of false positives, the thresholding was tuned empirically to throw out candidate bounding boxes without enough votes. It can be seen that this classifier does a decent job of false positive detection on the test images.

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]




Video Implementation
---

#### Video Output
Here's a [link to my video result](./project_video_output.mp4). The pipeline performs reasonably well on the entire project video, but there remain a few unstable bounding boxes around the end. 


### False Positive Filtering

Again, false positive filtering is done as explained in the previous section, via the use of Heatmaps. In addition to the aforementioned discussion, the video pipeline introduces an averaging over N heatmaps of the the previous N frames of the video to obtain a smoother bounding box around the vehicles. This averaging also be interpreted a way of false positive filtering, as it attempts to average out any bad detections that may occur.




Issues / Areas for Improvement
---

- Robustness: The bounding box is not always accurate and may represent the car to be larger or smaller in the image then they actually are. In addition, there are corner cases of two cars being close to each other. As well as the cars on the other side of the side of the highway - this might be removable by providing some more training images of such situations.

- Parameter tuning: The tuning of various heatmap thresholding and window sizes and count were somewhat cumbersome and a lot more difficult than finding a good set of parameters for the SVC. A lot of trial-and-error search. I'd like to go back and fine-tune these to obtain better bounding boxes a fewer false positives.

- Speed: Even with the sub-sampling window search in which we only extract the HOG features once, it still takes a decent amount of time for detection. In real scenario, this would have to be done in real-time and would require faster proessing 

- This can further be combined with the lane detection project to compose a pipeline for simultaneous lane and vehicle detection. I'd like to put this together once I reduce some of the errors in this module.





