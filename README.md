# EyeQ-Tracking-Challenge-V2
This repository is intended for EyeQ's AI Challenge for person detection and tracking based on classical computer vision approaches.

## Available Input
Input video can be downloaded from here.
https://bit.ly/2L8hFGE


### System Configuration
Tensorflow's Object Detection API has been used to train the desired architecture. I have installed the requirements for the API and configure the machine with CUDA, Cudnn libraries for GPU acceleration. To get more details, please have a look at [Tensorflow's Object Detection API Instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). 


## Developed Approach
### Detection
I have used Faster RCNN Resnet-50 as base detection algorithm that can recognzie persons in a video. I have given it preference over Yolo and SSD because latter were not that accurate to detect the persons in the video and I was not allowed to fine-tune or retrain the system based on the required scenario. I have used "Non Maximum Suppresion" to clean the detection results, in case where multiple bounding boxes have been detected for a single person.
[Detection Results](https://drive.google.com/open?id=1cRYyxYhNgU-85h3J7fAHHhnUhxl16asq)

Download FasterRCNN [Resnet-50](download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) and provide the path in `person_tracker.py` to get above results.

### Tracking
The tracker used in the repository is "Centroid Based Tracking". It assigns a unique number to each identified person and started to track it onwards based on centroid estimation. It has capability to tack the person if detector fails for some consective frames. It is working best so far.

The limitations of the systems are:

1. We need to specify a threshold for missing person that in how many frames it can be out of frame. If within that period of time he doesn't caome back in the frame. The person will be dropped and counted as new person if found in the frame.

2. If two persons come infront of each other or cross each, at a certain point their bounding boxes will totally overlap each other and that can cause exchange of IDs.

3. Detecting only head and legs as person by detectors is not a consistent thing. This can cuase a significant increment in IDs (Person-2 in our case). This problem should be remove either by finetuning or retraing the system but i was not allowed here.  


### Video Inference
Run inference on the input video as:

	`python3 person_tracker.py -i INPUT_VIDEO_PATH.mp4 -o OUTPUT_VIDEO_PATH.mp4`
	
	OUTPUT_VIDEO_PATH.mp4 (contains visualization of detection + tracking)
 
## Inference Video
Have a look at tracking results [online]().

### Other Trackers and Approaches Tried
I have tried to use different detectors both from classical computer vison and deep learning. But didn't get satisfieable results.

The list of trackers that I have tried were.

1. DeepSort with Yolo & Resnet. Results can be found for reference [here](https://drive.google.com/open?id=12aT5E7FwsvUO0m2L0AiaTyRLnExPW-s3)

2. Channel and Spatial Reliability Tracker (CSRT), Kernelized Correlation Filters (KCF)

3. CSRT with KNN Tracking

4. Dlib Correlation Tracker

