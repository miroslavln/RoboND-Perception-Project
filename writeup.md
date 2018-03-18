## Project: Perception Pick & Place
#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
The pipeline consists of the following steps:
* Statistical outliers filter with number of neigbour points equal to 10 and threshold of 0.1
* Voxel Grid Downsampling using leaf size of 0.005
* Passhthrough filter for z and y
* RANSAC Plane Segmentations
    
#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
The clustering piple consists of the following steps.
* Euclidian Clusterning with cluster tolerance of 0.05 and minumum cluster size of 50 and max size 20000
# 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
* I created a color histogram with 32 bins and I used 16 bins for the normal histogram. I trained the classifier with 100 spawns for each object.

![demo-1](https://github.com/miroslavln/RoboND-Perception-Project/blob/master/images/figure_1.png)
![demo-2](https://github.com/miroslavln/RoboND-Perception-Project/blob/master/images/figure_2.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.


![test-1](https://github.com/miroslavln/RoboND-Perception-Project/blob/master/images/scene_1.png)
![test-2](https://github.com/miroslavln/RoboND-Perception-Project/blob/master/images/scene_2.png)
![test-3](https://github.com/miroslavln/RoboND-Perception-Project/blob/master/images/scene_3.png)

I spent a lot of time tinkering with the parameters of the individual filters. I was able to get incrementally better results,
with each adjustment. I also tryed different kernel when training the svm but the linear kernel seems to perform best.



